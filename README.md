# Quant Pro Web/Desktop

这个仓库现在同时包含桌面版和网页版，两者共用同一份交易引擎 `quant_pro_core.py`，保证信号、撮合、交易明细、明日指令完全一致。

## 主要文件

- `quant_pro_core.py`: 共享回测核心
- `quant_pro_v5_2_pro.py`: Tk 桌面版
- `streamlit_app.py`: Streamlit 网页版入口
- `sjb.py`: 兼容旧入口，转到 `streamlit_app.py`
- `data/*.day`: 回测数据

## 本地运行

桌面版:

```bash
python quant_pro_v5_2_pro.py
```

网页版:

```bash
streamlit run streamlit_app.py
```

## GitHub 部署建议

这个项目的网页版是 Python + Streamlit，不适合直接用 GitHub Pages 托管运行时。

推荐方式:

1. 把仓库推到 GitHub。
2. 保留 `streamlit_app.py`、`quant_pro_core.py`、`requirements.txt` 和 `data/`。
3. 用支持 Python/Streamlit 的托管平台从 GitHub 仓库直接部署。
4. 如果使用 Streamlit Community Cloud，入口文件填写 `streamlit_app.py`。

## 一致性原则

后续如果你要改交易逻辑，只改 `quant_pro_core.py`。

不要分别改桌面版和网页版里的策略逻辑，否则结果会漂移。
