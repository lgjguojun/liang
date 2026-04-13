from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Callable

import pandas as pd

APP_TITLE = "A股量化限价撮合工作站 v5.3"
PRICE_DIVISOR = 100.0
DATE_FORMAT = "%Y-%m-%d"

DETAIL_COLUMNS = [
    "日期",
    "开盘",
    "最高",
    "最低",
    "收盘",
    "信号标识",
    "仓位",
    "成本价",
    "待离场",
    "次日指令",
    "净值",
    "交易记录",
]

SIGNAL_FILTER_BUY = "buy"
SIGNAL_FILTER_BUY_N = "buy_n"
SIGNAL_FILTER_SELL = "sell"
SIGNAL_FILTER_SELL_N = "sell_n"

SIGNAL_FILTER_LABELS = {
    SIGNAL_FILTER_BUY: "买入",
    SIGNAL_FILTER_BUY_N: "继续N天买入",
    SIGNAL_FILTER_SELL: "卖出",
    SIGNAL_FILTER_SELL_N: "继续N天卖出",
}

INSTRUCTIONS_TEXT = (
    "【信号】收盘价 > 近N日最高收盘(含当日) * (1-缓冲区)，产生买入信号。\n"
    "【买入】次日挂单=昨日收盘*买入比例；若低开则按开盘价成交，若盘中触及则按挂单价成交。\n"
    "【持有】不设固定止盈，只要收盘信号仍在就继续持有。\n"
    "【卖出】收盘信号消失后，下一个交易日先挂单[昨日收盘*离场比例]；若高开成交则按开盘价，"
    "若盘中触及则按挂单价，未触及则按收盘价离场。\n"
    "【时间段】可选全时间段或自定义开始/结束日期，区间会自动对齐到该股在段内的实际交易日，"
    "并从区间首个交易日重新开始回测。\n"
    "【筛选】批量扫描按所选时间段最后一个交易日的 raw_signal 连续天数筛选，"
    "“买入/卖出”分别等价于连续1天买入/卖出。"
)


@dataclass(frozen=True)
class StrategyParams:
    n_days: int = 3
    buffer_pct: float = 0.01
    buy_limit: float = 0.97
    sell_limit: float = 1.03
    fee: float = 0.0005


@dataclass
class BacktestResult:
    analysis_df: pd.DataFrame
    details_df: pd.DataFrame
    state: dict[str, Any]


@dataclass(frozen=True)
class CompositeScanPreset:
    label: str
    n_days: int
    buy_limit: float
    sell_limit: float
    window_months: int = 0
    window_days: int = 0


COMPOSITE_SCAN_TOP_N = 10
COMPOSITE_SCAN_PRESETS = (
    CompositeScanPreset("5天 1/1 1个月", n_days=5, buy_limit=1.0, sell_limit=1.0, window_months=1),
    CompositeScanPreset("3天 0.97/1.03 1.5个月", n_days=3, buy_limit=0.97, sell_limit=1.03, window_days=45),
    CompositeScanPreset("5天 0.97/1.03 1个月", n_days=5, buy_limit=0.97, sell_limit=1.03, window_months=1),
    CompositeScanPreset("3天 1/1 1.5个月", n_days=3, buy_limit=1.0, sell_limit=1.0, window_days=45),
    CompositeScanPreset("5天 1/1 1.5个月", n_days=5, buy_limit=1.0, sell_limit=1.0, window_days=45),
    CompositeScanPreset("3天 0.97/1.03 1个月", n_days=3, buy_limit=0.97, sell_limit=1.03, window_months=1),
    CompositeScanPreset("5天 0.97/1.03 1.5个月", n_days=5, buy_limit=0.97, sell_limit=1.03, window_days=45),
    CompositeScanPreset("3天 1/1 1个月", n_days=3, buy_limit=1.0, sell_limit=1.0, window_months=1),
)

COMPOSITE_SUMMARY_COLUMNS = [
    "composite_rank",
    "symbol",
    "total_points",
    "hit_count",
    "first_place_count",
    "best_rank",
    "avg_rank",
    "details",
]

COMPOSITE_DETAIL_COLUMNS = [
    "preset_label",
    "window_start",
    "window_end",
    "n_days",
    "buy_limit",
    "sell_limit",
    "rank",
    "points",
    "symbol",
    "total_return",
    "max_drawdown",
    "equity",
    "signal_label",
    "signal_streak",
    "aligned_end",
]


def list_symbols(data_dir: str | Path = "data") -> list[str]:
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted(file.stem for file in path.glob("*.day"))


def load_day_data(
    symbol: str,
    data_dir: str | Path = "data",
    price_divisor: float = PRICE_DIVISOR,
) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}.day"
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with open(path, "rb") as handle:
        while chunk := handle.read(32):
            dt, open_px, high_px, low_px, close_px, _, _, _ = struct.unpack("<IIIIIfII", chunk)
            rows.append(
                {
                    "date": pd.to_datetime(str(dt)),
                    "open": open_px / price_divisor,
                    "high": high_px / price_divisor,
                    "low": low_px / price_divisor,
                    "close": close_px / price_divisor,
                }
            )

    if not rows:
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        empty.index.name = "date"
        return empty

    return pd.DataFrame(rows).set_index("date").sort_index()


def normalize_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.normalize()


def format_date(value: Any) -> str | None:
    ts = normalize_date(value)
    if ts is None:
        return None
    return ts.strftime(DATE_FORMAT)


def build_signal_streaks(signals: pd.Series) -> pd.Series:
    streaks: list[int] = []
    last_signal: int | None = None
    streak = 0

    for signal in signals.astype(int).tolist():
        if signal == last_signal:
            streak += 1
        else:
            streak = 1
            last_signal = signal
        streaks.append(streak)

    return pd.Series(streaks, index=signals.index, dtype=int)


def build_signal_filter_text(signal_value: int, streak_days: int) -> str:
    streak_days = max(int(streak_days), 1)
    if int(signal_value) == 1:
        return "买入" if streak_days == 1 else f"继续{streak_days}天买入"
    return "卖出" if streak_days == 1 else f"继续{streak_days}天卖出"


def describe_signal_filter(filter_mode: str, streak_days: int) -> str:
    streak_days = max(int(streak_days), 1)
    if filter_mode == SIGNAL_FILTER_BUY:
        return "买入"
    if filter_mode == SIGNAL_FILTER_BUY_N:
        return f"继续{streak_days}天买入"
    if filter_mode == SIGNAL_FILTER_SELL:
        return "卖出"
    if filter_mode == SIGNAL_FILTER_SELL_N:
        return f"继续{streak_days}天卖出"
    raise ValueError(f"未知的信号筛选类型: {filter_mode}")


def matches_signal_filter(state: dict[str, Any], filter_mode: str, streak_days: int) -> bool:
    aligned_end = state.get("aligned_end")
    if not aligned_end:
        return False

    last_signal = int(state.get("last_signal", 0))
    signal_streak = int(state.get("signal_streak", 0))
    streak_days = max(int(streak_days), 1)

    if filter_mode == SIGNAL_FILTER_BUY:
        return last_signal == 1 and signal_streak == 1
    if filter_mode == SIGNAL_FILTER_BUY_N:
        return last_signal == 1 and signal_streak == streak_days
    if filter_mode == SIGNAL_FILTER_SELL:
        return last_signal == 0 and signal_streak == 1
    if filter_mode == SIGNAL_FILTER_SELL_N:
        return last_signal == 0 and signal_streak == streak_days
    raise ValueError(f"未知的信号筛选类型: {filter_mode}")


def format_window_label(state: dict[str, Any]) -> str:
    aligned_start = state.get("aligned_start")
    aligned_end = state.get("aligned_end")
    if not aligned_start or not aligned_end:
        return "所选时间段内无交易数据"
    if state.get("is_full_range", False):
        return f"全时间段（{aligned_start} 至 {aligned_end}）"
    return f"{aligned_start} 至 {aligned_end}"


def build_base_state(
    selected_start: Any = None,
    selected_end: Any = None,
    aligned_start: Any = None,
    aligned_end: Any = None,
) -> dict[str, Any]:
    is_full_range = normalize_date(selected_start) is None and normalize_date(selected_end) is None
    return {
        "position": 0,
        "cash": 1.0,
        "units": 0.0,
        "entry_price": 0.0,
        "pending_exit": False,
        "last_signal": 0,
        "last_close": 0.0,
        "buy_order": None,
        "sell_order": None,
        "signal_streak": 0,
        "signal_filter_label": "",
        "selected_start": format_date(selected_start),
        "selected_end": format_date(selected_end),
        "aligned_start": format_date(aligned_start),
        "aligned_end": format_date(aligned_end),
        "is_full_range": is_full_range,
    }


def empty_result(
    df: pd.DataFrame | None = None,
    *,
    selected_start: Any = None,
    selected_end: Any = None,
    aligned_start: Any = None,
    aligned_end: Any = None,
) -> BacktestResult:
    base = df.copy() if df is not None else pd.DataFrame(columns=["open", "high", "low", "close"])
    if "rolling_max" not in base.columns:
        base["rolling_max"] = pd.Series(dtype=float)
    if "raw_signal" not in base.columns:
        base["raw_signal"] = pd.Series(dtype=int)
    if "signal_streak" not in base.columns:
        base["signal_streak"] = pd.Series(dtype=int)
    if "equity_curve" not in base.columns:
        base["equity_curve"] = pd.Series(dtype=float)

    state = build_base_state(
        selected_start=selected_start,
        selected_end=selected_end,
        aligned_start=aligned_start,
        aligned_end=aligned_end,
    )
    return BacktestResult(base, pd.DataFrame(columns=DETAIL_COLUMNS), state)


def slice_data_by_window(
    df: pd.DataFrame,
    start_date: Any = None,
    end_date: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ts = normalize_date(start_date)
    end_ts = normalize_date(end_date)

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("开始日期不能晚于结束日期")

    window_df = df.copy()
    if start_ts is not None:
        window_df = window_df.loc[window_df.index >= start_ts]
    if end_ts is not None:
        window_df = window_df.loc[window_df.index <= end_ts]

    aligned_start = window_df.index[0] if not window_df.empty else None
    aligned_end = window_df.index[-1] if not window_df.empty else None
    state = build_base_state(
        selected_start=start_ts,
        selected_end=end_ts,
        aligned_start=aligned_start,
        aligned_end=aligned_end,
    )
    return window_df, state


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    start_date: Any = None,
    end_date: Any = None,
) -> BacktestResult:
    if df.empty:
        return empty_result(df, selected_start=start_date, selected_end=end_date)

    work_df, window_state = slice_data_by_window(df, start_date, end_date)
    if work_df.empty:
        return empty_result(
            work_df,
            selected_start=start_date,
            selected_end=end_date,
            aligned_start=window_state.get("aligned_start"),
            aligned_end=window_state.get("aligned_end"),
        )

    work_df = work_df.copy()
    work_df["rolling_max"] = work_df["close"].rolling(params.n_days).max()
    work_df["raw_signal"] = (
        work_df["close"] > work_df["rolling_max"] * (1 - params.buffer_pct)
    ).astype(int)
    work_df["signal_streak"] = build_signal_streaks(work_df["raw_signal"])

    equity = [1.0] * len(work_df)
    pos = 0
    cash = 1.0
    units = 0.0
    entry_price = 0.0
    pending_exit = False
    details_rows: list[dict[str, Any]] = []

    for i in range(1, len(work_df)):
        dt = work_df.index[i].date()
        prev_close = float(work_df["close"].iloc[i - 1])
        open_px = float(work_df["open"].iloc[i])
        high_px = float(work_df["high"].iloc[i])
        low_px = float(work_df["low"].iloc[i])
        close_px = float(work_df["close"].iloc[i])

        trade_msg = ""
        signal_tag = "买入信号" if int(work_df["raw_signal"].iloc[i]) == 1 else "卖出信号"
        next_action = ""

        if pos == 0:
            if int(work_df["raw_signal"].iloc[i - 1]) == 1:
                buy_order = prev_close * params.buy_limit
                next_action = f"买入挂单@{buy_order:.3f}"
                if open_px <= buy_order:
                    entry_price = open_px
                    units = (cash * (1 - params.fee)) / open_px
                    cash = 0.0
                    pos = 1
                    trade_msg = f"买入(低开):{open_px:.3f}"
                elif low_px <= buy_order:
                    entry_price = buy_order
                    units = (cash * (1 - params.fee)) / buy_order
                    cash = 0.0
                    pos = 1
                    trade_msg = f"买入(挂单):{buy_order:.3f}"

                if pos == 1 and int(work_df["raw_signal"].iloc[i]) == 0:
                    pending_exit = True
                    next_action = f"离场挂单@{close_px * params.sell_limit:.3f}"
                    trade_msg += " -> 卖出信号(T+1次日处理)"
        else:
            if pending_exit:
                sell_order = prev_close * params.sell_limit
                next_action = f"离场挂单@{sell_order:.3f}"
                if open_px >= sell_order:
                    cash = units * open_px * (1 - params.fee)
                    trade_msg = f"卖出(高开):{open_px:.3f}"
                    units = 0.0
                    pos = 0
                    entry_price = 0.0
                    pending_exit = False
                elif high_px >= sell_order:
                    cash = units * sell_order * (1 - params.fee)
                    trade_msg = f"卖出(挂单):{sell_order:.3f}"
                    units = 0.0
                    pos = 0
                    entry_price = 0.0
                    pending_exit = False
                else:
                    cash = units * close_px * (1 - params.fee)
                    trade_msg = f"卖出(收盘):{close_px:.3f}"
                    units = 0.0
                    pos = 0
                    entry_price = 0.0
                    pending_exit = False
            elif int(work_df["raw_signal"].iloc[i]) == 0:
                pending_exit = True
                next_action = f"离场挂单@{close_px * params.sell_limit:.3f}"
                trade_msg = "卖出信号(T+1次日执行)"
            else:
                next_action = "继续持有"

        equity[i] = cash if pos == 0 else units * close_px
        details_rows.append(
            {
                "日期": dt,
                "开盘": open_px,
                "最高": high_px,
                "最低": low_px,
                "收盘": close_px,
                "信号标识": signal_tag,
                "仓位": "持仓" if pos == 1 else "空仓",
                "成本价": round(entry_price, 3) if pos == 1 else None,
                "待离场": "是" if pending_exit else "",
                "次日指令": next_action,
                "净值": round(equity[i], 4),
                "交易记录": trade_msg,
            }
        )

    work_df["equity_curve"] = equity
    last_signal = int(work_df["raw_signal"].iloc[-1])
    last_close = float(work_df["close"].iloc[-1])
    signal_streak = int(work_df["signal_streak"].iloc[-1])
    state = {
        **window_state,
        "position": pos,
        "cash": cash,
        "units": units,
        "entry_price": float(entry_price) if pos == 1 else 0.0,
        "pending_exit": pending_exit,
        "last_signal": last_signal,
        "last_close": last_close,
        "buy_order": last_close * params.buy_limit if pos == 0 and last_signal == 1 else None,
        "sell_order": last_close * params.sell_limit if pos == 1 and pending_exit else None,
        "signal_streak": signal_streak,
        "signal_filter_label": build_signal_filter_text(last_signal, signal_streak),
    }
    details_df = pd.DataFrame(details_rows, columns=DETAIL_COLUMNS)
    return BacktestResult(work_df, details_df, state)


def compute_metrics(analysis_df: pd.DataFrame) -> tuple[float, float]:
    if analysis_df.empty or "equity_curve" not in analysis_df.columns:
        return 0.0, 0.0
    total_return = float(analysis_df["equity_curve"].iloc[-1] - 1)
    max_drawdown = float((analysis_df["equity_curve"] / analysis_df["equity_curve"].cummax() - 1).min())
    return total_return, max_drawdown


def build_composite_end_date_label(end_date: Any) -> str:
    end_ts = normalize_date(end_date)
    if end_ts is None:
        raise ValueError("综合排名需要指定结束日期")
    return end_ts.strftime(DATE_FORMAT)


def resolve_relative_window_from_end(
    end_date: Any,
    *,
    months: int = 0,
    days: int = 0,
) -> tuple[str, str]:
    end_ts = normalize_date(end_date)
    if end_ts is None:
        raise ValueError("综合排名需要指定结束日期")

    start_ts = end_ts
    if months:
        start_ts = start_ts - pd.DateOffset(months=months)
    if days:
        start_ts = start_ts - pd.Timedelta(days=days)
    return start_ts.strftime(DATE_FORMAT), end_ts.strftime(DATE_FORMAT)


def build_composite_strategy_params(base_params: StrategyParams, preset: CompositeScanPreset) -> StrategyParams:
    return StrategyParams(
        n_days=preset.n_days,
        buffer_pct=base_params.buffer_pct,
        buy_limit=preset.buy_limit,
        sell_limit=preset.sell_limit,
        fee=base_params.fee,
    )


def build_batch_scan_row(symbol: str, result: BacktestResult) -> dict[str, Any]:
    total_return, max_drawdown = compute_metrics(result.analysis_df)
    return {
        "symbol": symbol,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "equity": float(result.analysis_df["equity_curve"].iloc[-1]),
        "signal_label": result.state["signal_filter_label"],
        "signal_streak": int(result.state["signal_streak"]),
        "aligned_end": result.state["aligned_end"],
    }


def sort_batch_scan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (-float(item["total_return"]), -float(item["max_drawdown"]), str(item["symbol"])),
    )


def run_composite_batch_ranking(
    symbols: list[str],
    load_data: Callable[[str], pd.DataFrame],
    base_params: StrategyParams,
    end_date: Any,
    filter_mode: str | None = None,
    streak_days: int = 1,
    *,
    top_n: int = COMPOSITE_SCAN_TOP_N,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_label = build_composite_end_date_label(end_date)
    safe_top_n = max(int(top_n), 1)
    total_steps = len(symbols) * len(COMPOSITE_SCAN_PRESETS)
    completed_steps = 0
    detail_rows: list[dict[str, Any]] = []

    for preset in COMPOSITE_SCAN_PRESETS:
        start_date, preset_end_date = resolve_relative_window_from_end(
            end_label,
            months=preset.window_months,
            days=preset.window_days,
        )
        params = build_composite_strategy_params(base_params, preset)
        preset_rows: list[dict[str, Any]] = []

        for symbol in symbols:
            try:
                result = run_backtest(load_data(symbol), params, start_date=start_date, end_date=preset_end_date)
                if result.analysis_df.empty:
                    continue
                if filter_mode is not None and not matches_signal_filter(result.state, filter_mode, streak_days):
                    continue
                preset_rows.append(build_batch_scan_row(symbol, result))
            except Exception:
                continue
            finally:
                completed_steps += 1
                if progress_callback is not None:
                    progress_callback(completed_steps, total_steps, preset.label)

        for rank, row in enumerate(sort_batch_scan_rows(preset_rows)[:safe_top_n], start=1):
            detail_rows.append(
                {
                    "preset_label": preset.label,
                    "window_start": start_date,
                    "window_end": preset_end_date,
                    "n_days": preset.n_days,
                    "buy_limit": preset.buy_limit,
                    "sell_limit": preset.sell_limit,
                    "rank": rank,
                    "points": safe_top_n - rank + 1,
                    **row,
                }
            )

    if not detail_rows:
        return (
            pd.DataFrame(columns=COMPOSITE_SUMMARY_COLUMNS),
            pd.DataFrame(columns=COMPOSITE_DETAIL_COLUMNS),
        )

    detail_df = pd.DataFrame(detail_rows, columns=COMPOSITE_DETAIL_COLUMNS)
    summary_rows: list[dict[str, Any]] = []

    for symbol, group in detail_df.groupby("symbol", sort=False):
        ordered_group = group.sort_values(["rank", "preset_label"], ascending=[True, True])
        summary_rows.append(
            {
                "symbol": symbol,
                "total_points": int(group["points"].sum()),
                "hit_count": int(len(group)),
                "first_place_count": int((group["rank"] == 1).sum()),
                "best_rank": int(group["rank"].min()),
                "avg_rank": float(group["rank"].mean()),
                "details": "；".join(
                    f"{row.preset_label}#{int(row.rank)}({row.total_return:.2%})"
                    for row in ordered_group.itertuples()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["total_points", "hit_count", "first_place_count", "best_rank", "avg_rank", "symbol"],
        ascending=[False, False, False, True, True, True],
    )
    summary_df = summary_df.reset_index(drop=True)
    summary_df.insert(0, "composite_rank", range(1, len(summary_df) + 1))
    summary_df = summary_df[COMPOSITE_SUMMARY_COLUMNS]
    detail_df = detail_df.sort_values(["rank", "preset_label", "symbol"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    return summary_df, detail_df


def build_next_day_instructions(state: dict[str, Any]) -> list[str]:
    if not state.get("aligned_end"):
        return ["所选时间段内无交易数据"]

    position = int(state.get("position", 0))
    pending_exit = bool(state.get("pending_exit", False))
    last_signal = int(state.get("last_signal", 0))

    if position == 1:
        lines = [f"当前持仓成本: {state['entry_price']:.3f}"]
        if pending_exit and state.get("sell_order") is not None:
            lines.append(f"明日离场挂单: {state['sell_order']:.3f}")
            lines.append("若明日未触及离场挂单，则按明日收盘价卖出")
        else:
            lines.append("明日继续持有，若收盘出现卖出信号则下一个交易日离场")
        return lines

    if last_signal == 1 and state.get("buy_order") is not None:
        return [
            f"当前空仓，建议买入挂单: {state['buy_order']:.3f}",
            "买入后不设止盈，只在卖出信号出现后的下一交易日离场",
        ]

    return ["当前空仓，明日无挂单指令"]
