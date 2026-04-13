import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quant_pro_core import (
    APP_TITLE,
    COMPOSITE_SCAN_PRESETS,
    COMPOSITE_SCAN_TOP_N,
    INSTRUCTIONS_TEXT,
    SIGNAL_FILTER_BUY,
    SIGNAL_FILTER_BUY_N,
    SIGNAL_FILTER_LABELS,
    SIGNAL_FILTER_SELL,
    SIGNAL_FILTER_SELL_N,
    StrategyParams,
    build_next_day_instructions,
    compute_metrics,
    describe_signal_filter,
    format_window_label,
    list_symbols,
    load_day_data,
    matches_signal_filter,
    run_backtest,
    run_composite_batch_ranking,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FILTER_MODES = [
    SIGNAL_FILTER_BUY,
    SIGNAL_FILTER_BUY_N,
    SIGNAL_FILTER_SELL,
    SIGNAL_FILTER_SELL_N,
]


@st.cache_data(show_spinner=False)
def load_symbol_data(symbol: str) -> pd.DataFrame:
    return load_day_data(symbol, "data")


@st.cache_data(show_spinner=False)
def analyze_symbol(
    symbol: str,
    n_days: int,
    buffer_pct: float,
    buy_limit: float,
    sell_limit: float,
    fee: float,
    start_date: str | None,
    end_date: str | None,
):
    params = StrategyParams(
        n_days=n_days,
        buffer_pct=buffer_pct,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        fee=fee,
    )
    return run_backtest(load_symbol_data(symbol), params, start_date=start_date, end_date=end_date)


def render_instruction_block(state: dict):
    st.subheader("明日交易指令")
    for line in build_next_day_instructions(state):
        st.write(f"- {line}")


def render_single_stock_tab(
    params: StrategyParams,
    symbols: list[str],
    start_date: str | None,
    end_date: str | None,
):
    st.subheader("单股分析")
    if not symbols:
        st.warning("未找到 `data/*.day` 数据文件。")
        return

    symbol = st.selectbox("选择股票", symbols)
    result = analyze_symbol(
        symbol,
        params.n_days,
        params.buffer_pct,
        params.buy_limit,
        params.sell_limit,
        params.fee,
        start_date,
        end_date,
    )

    if result.analysis_df.empty:
        st.warning("所选时间段内没有交易数据，请调整时间范围后重试。")
        return

    analysis_df = result.analysis_df
    details_df = result.details_df
    total_ret, mdd = compute_metrics(analysis_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", f"{total_ret:.2%}")
    c2.metric("最大回撤", f"{mdd:.2%}")
    c3.metric("最新净值", f"{analysis_df['equity_curve'].iloc[-1]:.4f}")
    c4.metric("最后信号", result.state["signal_filter_label"])

    st.caption(f"回测区间: {format_window_label(result.state)}")
    render_instruction_block(result.state)

    st.subheader("净值曲线")
    st.line_chart(analysis_df["equity_curve"])

    with st.expander("信号校对", expanded=True):
        review_df = analysis_df[
            ["open", "high", "low", "close", "rolling_max", "raw_signal", "signal_streak", "equity_curve"]
        ].copy()
        review_df = review_df.rename(
            columns={
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "rolling_max": f"近{params.n_days}日最高收盘",
                "raw_signal": "raw_signal",
                "signal_streak": "连续天数",
                "equity_curve": "净值",
            }
        )
        st.dataframe(review_df.tail(60), use_container_width=True)

    st.subheader("交易明细")
    st.dataframe(details_df, use_container_width=True, hide_index=True)


def render_batch_tab(
    params: StrategyParams,
    symbols: list[str],
    start_date: str | None,
    end_date: str | None,
    filter_mode: str,
    streak_days: int,
):
    st.subheader("全量排行")
    if not symbols:
        st.warning("未找到 `data/*.day` 数据文件。")
        return

    if not st.button("开始全量回测", type="primary"):
        return

    rows = []
    progress = st.progress(0.0)
    total = len(symbols)

    for idx, symbol in enumerate(symbols, start=1):
        result = analyze_symbol(
            symbol,
            params.n_days,
            params.buffer_pct,
            params.buy_limit,
            params.sell_limit,
            params.fee,
            start_date,
            end_date,
        )
        if result.analysis_df.empty:
            progress.progress(idx / total)
            continue
        if not matches_signal_filter(result.state, filter_mode, streak_days):
            progress.progress(idx / total)
            continue

        total_ret, mdd = compute_metrics(result.analysis_df)
        rows.append(
            {
                "代码": symbol,
                "累计收益": total_ret,
                "最大回撤": mdd,
                "最新净值": result.analysis_df["equity_curve"].iloc[-1],
                "最后信号": result.state["signal_filter_label"],
                "连续天数": result.state["signal_streak"],
                "结束交易日": result.state["aligned_end"],
            }
        )
        progress.progress(idx / total)

    filter_text = describe_signal_filter(filter_mode, streak_days)
    st.caption(f"筛选条件: {filter_text}")
    st.caption(
        "回测区间: "
        + ("全时间段" if start_date is None and end_date is None else f"{start_date or '最早'} 至 {end_date or '最晚'}")
    )

    if not rows:
        st.warning("所选时间段和信号筛选下没有匹配股票。")
        return

    ranking_df = pd.DataFrame(rows).sort_values(["累计收益", "最大回撤", "代码"], ascending=[False, False, True])
    ranking_df = ranking_df.reset_index(drop=True)
    st.dataframe(
        ranking_df.style.format(
            {
                "累计收益": "{:.2%}",
                "最大回撤": "{:.2%}",
                "最新净值": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_composite_batch_tab(
    params: StrategyParams,
    symbols: list[str],
    composite_end_date: str | None,
    filter_mode: str,
    streak_days: int,
):
    st.subheader("综合排名")
    st.caption(
        f"固定使用 {len(COMPOSITE_SCAN_PRESETS)} 组参数组合；每组按累计收益取前{COMPOSITE_SCAN_TOP_N}名，再按名次积分汇总。"
    )
    st.caption("当前实现中，1.5个月按45个自然日回推。")

    if not symbols:
        st.warning("未找到 `data/*.day` 数据文件。")
        return

    if not composite_end_date:
        st.info("请输入综合排名结束日期，格式为 YYYY-MM-DD。")
        return

    if not st.button("开始综合排名", type="primary"):
        return

    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(completed: int, total: int, preset_label: str):
        ratio = 1.0 if total == 0 else completed / total
        progress.progress(ratio)
        status.caption(f"扫描进度: {completed}/{total} | 当前组合: {preset_label}")

    try:
        summary_df, detail_df = run_composite_batch_ranking(
            symbols=symbols,
            load_data=load_symbol_data,
            base_params=params,
            end_date=composite_end_date,
            filter_mode=filter_mode,
            streak_days=streak_days,
            top_n=COMPOSITE_SCAN_TOP_N,
            progress_callback=on_progress,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    progress.progress(1.0)
    status.caption(f"综合排名结束日期: {composite_end_date}")

    if summary_df.empty:
        st.warning("没有股票进入综合排名，请检查结束日期或筛选条件。")
        return

    summary_display_df = summary_df.rename(
        columns={
            "composite_rank": "综合排名",
            "symbol": "代码",
            "total_points": "综合积分",
            "hit_count": "入围次数",
            "first_place_count": "冠军次数",
            "best_rank": "最佳名次",
            "avg_rank": "平均名次",
            "details": "入围明细",
        }
    )
    st.dataframe(
        summary_display_df.style.format({"平均名次": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("入围明细", expanded=False):
        detail_display_df = detail_df.rename(
            columns={
                "preset_label": "参数组合",
                "window_start": "开始日期",
                "window_end": "结束日期",
                "n_days": "回看天数",
                "buy_limit": "买入比例",
                "sell_limit": "离场比例",
                "rank": "组内名次",
                "points": "积分",
                "symbol": "代码",
                "total_return": "累计收益",
                "max_drawdown": "最大回撤",
                "equity": "最新净值",
                "signal_label": "最后信号",
                "signal_streak": "连续天数",
                "aligned_end": "实际结束交易日",
            }
        )
        st.dataframe(
            detail_display_df.style.format(
                {
                    "买入比例": "{:.2f}",
                    "离场比例": "{:.2f}",
                    "累计收益": "{:.2%}",
                    "最大回撤": "{:.2%}",
                    "最新净值": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("桌面版与网页版共用同一份回测核心 `quant_pro_core.py`。")

    symbols = list_symbols("data")

    st.sidebar.header("策略参数")
    n_days = st.sidebar.slider("回看天数 (N)", 1, 20, 3)
    buffer_pct = st.sidebar.slider("信号缓冲区 (%)", 0.0, 5.0, 1.0) / 100
    buy_limit = st.sidebar.number_input("买入挂单比例", value=0.97, step=0.01, format="%.2f")
    sell_limit = st.sidebar.number_input("离场挂单比例", value=1.03, step=0.01, format="%.2f")
    fee = st.sidebar.number_input("交易规费 (单边)", value=0.0005, format="%.4f")

    st.sidebar.subheader("数据时间段")
    full_range = st.sidebar.checkbox("全时间段", value=True)
    start_date = None
    end_date = None
    if not full_range:
        start_date = st.sidebar.text_input("开始日期", value="", placeholder="YYYY-MM-DD").strip() or None
        end_date = st.sidebar.text_input("结束日期", value="", placeholder="YYYY-MM-DD").strip() or None
        st.sidebar.caption("日期会自动对齐到该股在区间内的实际交易日。")

    st.sidebar.subheader("最后一天信号筛选")
    composite_end_date = st.sidebar.text_input(
        "综合排名结束日期",
        value=end_date or "",
        placeholder="YYYY-MM-DD",
    ).strip() or None
    st.sidebar.caption("综合排名只使用结束日期，开始日期会按 1个月 或 45天 自动回推。")
    filter_mode = st.sidebar.selectbox(
        "筛选类型",
        options=FILTER_MODES,
        format_func=lambda mode: SIGNAL_FILTER_LABELS[mode],
        index=0,
    )
    default_streak = 2 if filter_mode in {SIGNAL_FILTER_BUY_N, SIGNAL_FILTER_SELL_N} else 1
    streak_days = int(st.sidebar.number_input("连续天数 N", min_value=1, value=default_streak, step=1))
    st.sidebar.caption("“买入/卖出”等价于连续1天买入/卖出。")
    st.sidebar.text_area("策略逻辑说明", INSTRUCTIONS_TEXT, height=260)

    params = StrategyParams(
        n_days=n_days,
        buffer_pct=buffer_pct,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        fee=fee,
    )

    tab1, tab2, tab3 = st.tabs(["单股分析", "全量排行", "综合排名"])
    with tab1:
        render_single_stock_tab(params, symbols, start_date, end_date)
    with tab2:
        render_batch_tab(params, symbols, start_date, end_date, filter_mode, streak_days)
    with tab3:
        render_composite_batch_tab(params, symbols, composite_end_date, filter_mode, streak_days)


if __name__ == "__main__":
    main()
