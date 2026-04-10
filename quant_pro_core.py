from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import pandas as pd

APP_TITLE = "A股量化限价撮合工作站 v5.3"
PRICE_DIVISOR = 100.0
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
INSTRUCTIONS_TEXT = (
    "【信号】收盘价 > 近N日最高收盘(含当天) * (1-缓冲区)，产生买入信号。\n"
    "【买入】次日挂单[昨日收盘×买入比例]。若低开于挂单价按开盘价成交；若盘中触及则按挂单价成交。\n"
    "【持有】不设固定止盈，只要收盘信号仍在就继续持有。\n"
    "【卖出】收盘信号消失后，下一交易日先挂单[昨日收盘×离场比例]；若高开成交则按开盘价，若盘中触及则按挂单价，未触及则按收盘价离场。"
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


def empty_result(df: pd.DataFrame | None = None) -> BacktestResult:
    base = df.copy() if df is not None else pd.DataFrame(columns=["open", "high", "low", "close"])
    if "rolling_max" not in base.columns:
        base["rolling_max"] = pd.Series(dtype=float)
    if "raw_signal" not in base.columns:
        base["raw_signal"] = pd.Series(dtype=int)
    if "equity_curve" not in base.columns:
        base["equity_curve"] = pd.Series(dtype=float)
    state = {
        "position": 0,
        "cash": 1.0,
        "units": 0.0,
        "entry_price": 0.0,
        "pending_exit": False,
        "last_signal": 0,
        "last_close": 0.0,
        "buy_order": None,
        "sell_order": None,
    }
    return BacktestResult(base, pd.DataFrame(columns=DETAIL_COLUMNS), state)


def run_backtest(df: pd.DataFrame, params: StrategyParams) -> BacktestResult:
    if df.empty:
        return empty_result(df)

    work_df = df.copy()
    work_df["rolling_max"] = work_df["close"].rolling(params.n_days).max()
    work_df["raw_signal"] = (
        work_df["close"] > work_df["rolling_max"] * (1 - params.buffer_pct)
    ).astype(int)

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
    state = {
        "position": pos,
        "cash": cash,
        "units": units,
        "entry_price": float(entry_price) if pos == 1 else 0.0,
        "pending_exit": pending_exit,
        "last_signal": last_signal,
        "last_close": last_close,
        "buy_order": last_close * params.buy_limit if pos == 0 and last_signal == 1 else None,
        "sell_order": last_close * params.sell_limit if pos == 1 and pending_exit else None,
    }
    details_df = pd.DataFrame(details_rows, columns=DETAIL_COLUMNS)
    return BacktestResult(work_df, details_df, state)


def compute_metrics(analysis_df: pd.DataFrame) -> tuple[float, float]:
    if analysis_df.empty or "equity_curve" not in analysis_df.columns:
        return 0.0, 0.0
    total_return = float(analysis_df["equity_curve"].iloc[-1] - 1)
    max_drawdown = float((analysis_df["equity_curve"] / analysis_df["equity_curve"].cummax() - 1).min())
    return total_return, max_drawdown


def build_next_day_instructions(state: dict[str, Any]) -> list[str]:
    position = int(state.get("position", 0))
    pending_exit = bool(state.get("pending_exit", False))
    last_signal = int(state.get("last_signal", 0))

    if position == 1:
        lines = [f"当前持仓成本: {state['entry_price']:.3f}"]
        if pending_exit and state.get("sell_order") is not None:
            lines.append(f"明日离场挂单: {state['sell_order']:.3f}")
            lines.append("若明日未触及离场挂单，则按明日收盘价卖出")
        else:
            lines.append("明日继续持有，若收盘出现卖出信号则下一交易日离场")
        return lines

    if last_signal == 1 and state.get("buy_order") is not None:
        return [
            f"当前空仓，建议买入挂单: {state['buy_order']:.3f}",
            "买入后不设止盈，只在卖出信号出现后的下一交易日离场",
        ]

    return ["当前空仓，明日无挂单指令"]
