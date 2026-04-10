import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quant_pro_core import (
    APP_TITLE,
    INSTRUCTIONS_TEXT,
    StrategyParams,
    build_next_day_instructions,
    compute_metrics,
    list_symbols,
    load_day_data,
    run_backtest,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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
):
    params = StrategyParams(
        n_days=n_days,
        buffer_pct=buffer_pct,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        fee=fee,
    )
    return run_backtest(load_symbol_data(symbol), params)


def render_instruction_block(state: dict):
    st.subheader("明日交易指令")
    for line in build_next_day_instructions(state):
        st.write(f"• {line}")


def render_single_stock_tab(params: StrategyParams, symbols: list[str]):
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
    )
    analysis_df = result.analysis_df
    details_df = result.details_df
    total_ret, mdd = compute_metrics(analysis_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("累计收益率", f"{total_ret:.2%}")
    c2.metric("最大回撤", f"{mdd:.2%}")
    c3.metric("最新净值", f"{analysis_df['equity_curve'].iloc[-1]:.4f}")

    render_instruction_block(result.state)

    st.subheader("净值曲线")
    st.line_chart(analysis_df["equity_curve"])

    with st.expander("信号核对", expanded=True):
        review_df = analysis_df[["open", "high", "low", "close", "rolling_max", "raw_signal", "equity_curve"]].copy()
        review_df = review_df.rename(
            columns={
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "rolling_max": "近N日最高收盘",
                "raw_signal": "信号值",
                "equity_curve": "净值",
            }
        )
        st.dataframe(review_df.tail(60), use_container_width=True)

    st.subheader("交易明细")
    st.dataframe(details_df, use_container_width=True, hide_index=True)


def render_batch_tab(params: StrategyParams, symbols: list[str]):
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
        )
        total_ret, mdd = compute_metrics(result.analysis_df)
        rows.append(
            {
                "代码": symbol,
                "总收益": total_ret,
                "最大回撤": mdd,
                "最新净值": result.analysis_df["equity_curve"].iloc[-1],
                "最新信号": "买入信号" if result.state["last_signal"] == 1 else "卖出信号",
                "当前仓位": "持仓" if result.state["position"] == 1 else "空仓",
            }
        )
        progress.progress(idx / total)

    ranking_df = pd.DataFrame(rows).sort_values("总收益", ascending=False).reset_index(drop=True)
    st.dataframe(
        ranking_df.style.format(
            {
                "总收益": "{:.2%}",
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
    st.caption("桌面版与网页版共用同一份回测引擎 `quant_pro_core.py`。")

    symbols = list_symbols("data")

    st.sidebar.header("策略参数")
    n_days = st.sidebar.slider("回顾天数 (N)", 1, 20, 3)
    buffer_pct = st.sidebar.slider("信号缓冲区 (%)", 0.0, 5.0, 1.0) / 100
    buy_limit = st.sidebar.number_input("买入挂单比例", value=0.97, step=0.01, format="%.2f")
    sell_limit = st.sidebar.number_input("离场挂单比例", value=1.03, step=0.01, format="%.2f")
    fee = st.sidebar.number_input("交易规费 (单边)", value=0.0005, format="%.4f")
    st.sidebar.text_area("撮合逻辑说明", INSTRUCTIONS_TEXT, height=180)

    params = StrategyParams(
        n_days=n_days,
        buffer_pct=buffer_pct,
        buy_limit=buy_limit,
        sell_limit=sell_limit,
        fee=fee,
    )

    tab1, tab2 = st.tabs(["单股分析", "全量排行"])
    with tab1:
        render_single_stock_tab(params, symbols)
    with tab2:
        render_batch_tab(params, symbols)


if __name__ == "__main__":
    main()
