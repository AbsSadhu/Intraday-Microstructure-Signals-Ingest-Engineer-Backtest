"""
Interactive Streamlit Dashboard for the Microstructure Research Platform.

Run with: streamlit run src/dashboard.py

Features:
- Data source & symbol picker
- One-click pipeline execution
- Interactive Plotly charts (equity, features, predictions)
- Model comparison table with color-coded metrics
- Real-time signal indicator
- Downloadable report
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "Dashboard requires streamlit and plotly. Install with:\n"
        "  pip install streamlit plotly"
    )


# ── Page Config ──
st.set_page_config(
    page_title="Microstructure Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .signal-long { color: #3fb950; font-size: 2em; font-weight: bold; }
    .signal-short { color: #f85149; font-size: 2em; font-weight: bold; }
    .signal-flat { color: #8b949e; font-size: 2em; font-weight: bold; }
    h1 { color: #58a6ff !important; }
    .stSidebar { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Controls ──
st.sidebar.title("⚙️ Configuration")

source = st.sidebar.selectbox(
    "Data Source",
    ["sample", "synthetic", "yahoo", "binance", "coinbase"],
    index=0,
    help="Choose where to load trade data from"
)

symbol_defaults = {
    "sample": "BTCUSDT",
    "synthetic": "BTCUSDT",
    "yahoo": "AAPL",
    "binance": "BTCUSDT",
    "coinbase": "BTC-USD",
}
symbol = st.sidebar.text_input("Symbol", value=symbol_defaults.get(source, "BTCUSDT"))

horizon = st.sidebar.slider("Forecast Horizon (bars)", 1, 30, 5)
rows = st.sidebar.slider("Synthetic Rows", 5000, 50000, 10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Settings")
fee_bps = st.sidebar.number_input("Commission (bps)", 0.0, 10.0, 1.0, 0.5)
slippage_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 10.0, 0.5, 0.25)
position_sizing = st.sidebar.selectbox("Position Sizing", ["fixed", "kelly", "volatility"])
use_walk_forward = st.sidebar.checkbox("Walk-Forward Validation", value=True)

run_clicked = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)


# ── Main Content ──
st.title("📊 Microstructure Research Platform")
st.markdown("*Intraday signal analysis, ML prediction, and strategy backtesting*")

if run_clicked:
    with st.spinner("Running pipeline... this may take a minute."):
        try:
            from src.main import run_pipeline

            start_dt = datetime.now(tz=timezone.utc) - timedelta(hours=24)
            end_dt = datetime.now(tz=timezone.utc)

            report, best_name, best_res, extras = run_pipeline(
                source=source,
                symbol=symbol,
                start=start_dt,
                end=end_dt,
                horizon=horizon,
                synthetic_rows=rows,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                position_sizing=position_sizing,
                use_walk_forward=use_walk_forward,
            )

            st.session_state["report"] = report
            st.session_state["best_name"] = best_name
            st.session_state["best_res"] = best_res
            st.session_state["extras"] = extras
            st.success(f"✅ Pipeline complete! Best model: **{best_name.upper()}**")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Display Results ──
if "report" in st.session_state:
    report = st.session_state["report"]
    best_name = st.session_state["best_name"]
    best_res = st.session_state["best_res"]
    extras = st.session_state["extras"]

    # ── Signal Indicator ──
    col1, col2, col3, col4 = st.columns(4)

    signal = report["latest_signal"]
    with col1:
        if signal > 0:
            st.markdown(f'<div class="signal-long">🟢 LONG</div>', unsafe_allow_html=True)
        elif signal < 0:
            st.markdown(f'<div class="signal-short">🔴 SHORT</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-flat">⚪ FLAT</div>', unsafe_allow_html=True)
        st.metric("Predicted Return", f"{signal:.6f}")

    with col2:
        best_perf = report[best_name]["perf"]
        st.metric("Sharpe Ratio", f"{best_perf['sharpe']:.2f}")

    with col3:
        st.metric("Hit Rate", f"{best_perf['hit_rate']:.1%}")

    with col4:
        st.metric("Max Drawdown", f"{best_perf['max_drawdown']:.2%}")

    st.markdown("---")

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Equity & Performance",
        "🔬 Features & Signals",
        "🎯 Model Evaluation",
        "🌊 Market Regimes",
        "📋 Full Report",
    ])

    # ── Tab 1: Equity & Performance ──
    with tab1:
        best_perf_df = extras["best_perf"]

        # Equity curve with drawdown
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("Equity Curve", "Drawdown"),
        )

        fig.add_trace(
            go.Scatter(x=best_perf_df.index, y=best_perf_df["equity"],
                       name="Equity", line=dict(color="#58a6ff", width=2)),
            row=1, col=1,
        )

        # Drawdown
        peak = best_perf_df["equity"].cummax()
        dd = (best_perf_df["equity"] / peak) - 1.0
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                       name="Drawdown", line=dict(color="#f85149", width=1),
                       fillcolor="rgba(248, 81, 73, 0.2)"),
            row=2, col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            showlegend=True,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model comparison table
        st.subheader("Model Comparison")
        model_names = [k for k in report if k in ("lasso", "xgboost", "random_forest")]
        comparison_data = []
        for name in model_names:
            row = {"Model": name.upper()}
            row.update(report[name]["metrics"])
            row.update(report[name]["perf"])
            comparison_data.append(row)

        comp_df = pd.DataFrame(comparison_data).set_index("Model")
        st.dataframe(
            comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#1a4d2e")
                   .highlight_min(axis=0, color="#4d1a1a"),
            use_container_width=True,
        )

    # ── Tab 2: Features & Signals ──
    with tab2:
        features = extras["features"]

        # Feature correlation heatmap
        st.subheader("Feature Correlation Matrix")
        corr = features.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", zmin=-1, zmax=1,
        )
        fig_corr.update_layout(
            template="plotly_dark", height=600,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Feature importance
        importance = extras.get("best_importance")
        if importance is not None and not importance.empty:
            st.subheader(f"Feature Importance ({best_name})")
            fig_imp = px.bar(
                x=importance.values, y=importance.index,
                orientation="h", color=importance.values,
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(
                template="plotly_dark", height=400,
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance",
                showlegend=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── Tab 3: Model Evaluation ──
    with tab3:
        eval_report = extras.get("eval_report", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            ic = eval_report.get("ic", 0)
            st.metric("Information Coefficient", f"{ic:.4f}")
        with col2:
            icir = eval_report.get("icir", 0)
            st.metric("IC Information Ratio", f"{icir:.4f}")
        with col3:
            mono = eval_report.get("calibration_monotonicity", 0)
            if isinstance(mono, float) and not np.isnan(mono):
                cal_table = eval_report.get("calibration_table")
                if cal_table is not None:
                    mono = cal_table.attrs.get("monotonicity_score", mono)
            st.metric("Calibration Monotonicity", f"{mono:.0%}" if isinstance(mono, float) else "N/A")

        # Prediction scatter
        st.subheader("Predicted vs Actual Returns")
        scatter_df = pd.DataFrame({"Actual": best_res.y_test, "Predicted": best_res.preds})
        fig_scatter = px.scatter(
            scatter_df, x="Actual", y="Predicted",
            opacity=0.4, trendline="ols",
            color_discrete_sequence=["#58a6ff"],
        )
        fig_scatter.add_shape(type="line", x0=scatter_df["Actual"].min(),
                              y0=scatter_df["Actual"].min(),
                              x1=scatter_df["Actual"].max(),
                              y1=scatter_df["Actual"].max(),
                              line=dict(dash="dash", color="#3fb950"))
        fig_scatter.update_layout(
            template="plotly_dark", height=500,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Calibration table
        cal_table = eval_report.get("calibration_table")
        if cal_table is not None:
            st.subheader("Signal Calibration (by Prediction Quintile)")
            st.dataframe(cal_table.style.format("{:.6f}"), use_container_width=True)

    # ── Tab 4: Market Regimes ──
    with tab4:
        regimes = extras.get("regimes")
        bars = extras.get("bars")
        reg_summary = extras.get("regime_summary")

        if regimes is not None and bars is not None:
            st.subheader("Price with Regime Overlay")

            price = bars["close"]
            regime_aligned = regimes.reindex(price.index).ffill()

            fig_regime = go.Figure()
            regime_colors = {0: "rgba(248,81,73,0.15)", 1: "rgba(210,153,34,0.15)", 2: "rgba(63,185,80,0.15)"}
            regime_names = {0: "Bearish", 1: "Normal", 2: "Bullish"}

            for rv, color in regime_colors.items():
                mask = regime_aligned == rv
                if mask.any():
                    fig_regime.add_trace(go.Scatter(
                        x=price.index, y=price.where(mask),
                        fill="tozeroy", fillcolor=color,
                        line=dict(width=0), name=regime_names.get(rv, f"Regime {rv}"),
                        showlegend=True,
                    ))

            fig_regime.add_trace(go.Scatter(
                x=price.index, y=price,
                line=dict(color="#58a6ff", width=1.5), name="Price",
            ))
            fig_regime.update_layout(
                template="plotly_dark", height=400,
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            )
            st.plotly_chart(fig_regime, use_container_width=True)

            if reg_summary is not None:
                st.subheader("Regime Statistics")
                st.dataframe(reg_summary.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("Regime detection was not available for this run.")

    # ── Tab 5: Full Report JSON ──
    with tab5:
        st.subheader("Raw Report Data")
        st.json(report)
        st.download_button(
            label="📥 Download Report JSON",
            data=json.dumps(report, indent=2, default=str),
            file_name="microstructure_report.json",
            mime="application/json",
        )

else:
    # Landing / instructions
    st.markdown("""
    ### Welcome! 👋

    This platform analyzes intraday market microstructure using ML-based signal prediction
    and backtesting. Here's what it does:

    1. **Ingests** real or synthetic trade data (Binance, Coinbase, Yahoo Finance, or synthetic)
    2. **Engineers 16+ features** including VPIN, Kyle's Lambda, Amihud Illiquidity, and more
    3. **Detects market regimes** using Hidden Markov Models
    4. **Trains 3 ML models** (Lasso, XGBoost, Random Forest) with walk-forward validation
    5. **Backtests** a directional strategy with realistic transaction costs
    6. **Evaluates** signal quality using Information Coefficient and calibration analysis

    ---

    **To get started:** Configure settings in the sidebar and click **🚀 Run Pipeline**.

    ---

    | Data Source | Description | API Key Needed? |
    |-------------|-------------|-----------------|
    | `sample` | Bundled synthetic tape (~10k ticks) | No |
    | `synthetic` | Generate fresh synthetic data | No |
    | `yahoo` | Real US equity intraday (AAPL, SPY...) | No |
    | `binance` | Crypto aggTrades via REST | No |
    | `coinbase` | Crypto trades via REST | No |
    """)
