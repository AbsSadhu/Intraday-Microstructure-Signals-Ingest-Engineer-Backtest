# Intraday Market Microstructure Research Platform

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange?style=for-the-badge&logo=git" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?style=for-the-badge&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge&logo=github" alt="PRs Welcome" />
</p>

> A research-grade quantitative finance platform exploring high-frequency intraday signal extraction, microstructure modeling, and machine learning-based directional backtesting on tick-level trade data.

---

> [!WARNING]  
> **⚠️ ACTIVE RESEARCH PROTOTYPE & WORK-IN-PROGRESS**  
> * This project is currently in **active development** and is intended strictly as an engineering and academic prototype.
> * **It does not currently yield profitable trading results.** Intraday high-frequency signal extraction is highly sensitive to noise, latency, and execution model assumptions.
> * A significant amount of work remains to be done, including integration of order book depth (L2/L3 data), latency modeling, fee optimizations, and feature selection.

---

## 📌 Table of Contents
1. [What This Project Does](#-what-this-project-does)
2. [Why Lasso and LSTM Show Identical Results (Known Fallback)](#-why-lasso-and-lstm-show-identical-results-known-fallback)
3. [System Architecture](#-system-architecture)
4. [Detailed Step-by-Step Usage Guide](#-detailed-step-by-step-usage-guide)
5. [Empirical Results & Analysis](#-empirical-results--analysis)
6. [Visual Gallery of Research Charts](#-visual-gallery-of-research-charts)
7. [Academic Foundation & Microstructure Signals](#-academic-foundation--microstructure-signals)
8. [Optional Enhancements](#-optional-enhancements)
9. [References](#-references)

---

## 🔍 What This Project Does

In simple terms: **this system watches how trades happen tick-by-tick and tries to predict which way prices will move next.**

Market microstructure is the study of *how* trading happens — not just what prices are, but who's buying and selling, how urgently, and what information their trades reveal. This project:

1. **Ingests** real trade-level data (Binance, Coinbase, Yahoo Finance) or generates synthetic trade tapes.
2. **Resamples** raw trades into 1-minute OHLCV bars.
3. **Engineers 16+ signals** including four academic microstructure measures.
4. **Detects market regimes** (bearish / normal / bullish) using Gaussian Hidden Markov Models (HMM).
5. **Trains ML models** (Lasso, XGBoost, Random Forest, LSTM) to predict 5-minute forward returns.
6. **Validates** them using walk-forward expanding-window CV with purge + embargo gaps.
7. **Backtests** a long/short strategy with realistic costs (commission + slippage).
8. **Evaluates** signal quality via Information Coefficient, ICIR, and calibration analysis.
9. **Outputs** a full research report: JSON + 10 publication-quality charts.

---

## 💡 Why Lasso and LSTM Show Identical Results (Known Fallback)

If you ran the pipeline and observed that the **LSTM Network** results, charts, and metrics are exactly identical to **Lasso**, this is a **known design fallback**:

> [!IMPORTANT]  
> The LSTM network requires **PyTorch** (`torch`). If PyTorch is not installed in your Python environment, the system prints a warning and **automatically falls back to using the Lasso model** to prevent execution crashes.
> 
> To enable true LSTM training and resolve this duplicate behavior, install the lightweight CPU version of PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> Once installed, re-run the research pipeline to train the neural network and generate actual LSTM predictions and graphs.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  Binance REST/WS  ·  Coinbase REST  ·  Yahoo Finance    │
│            ·  Synthetic (OU process)                     │
└────────────────────────┬─────────────────────────────────┘
                         │ raw trades [timestamp, price, qty, side]
                         ▼
┌──────────────────────────────────────────────────────────┐
│              PREPROCESSING  (data_fetch.py)              │
│    resample_trades() → 1-min OHLCV + signed_volume       │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
┌─────────────────────┐      ┌──────────────────────────┐
│  FEATURE ENG.       │      │  REGIME DETECTION        │
│  (features.py)      │      │  (regime.py)             │
│                     │      │                          │
│  Core (12):         │      │  Gaussian HMM (3 states) │
│  · realized_vol     │      │  0 = bearish             │
│  · vpin             │      │  1 = normal              │
│  · vol_imbalance    │      │  2 = bullish             │
│  · momentum, etc.   │      │                          │
│                     │      │  Fallback: vol-quantile  │
│  Microstructure (4):│      └──────────┬───────────────┘
│  · Kyle's Lambda    │                 │ regime label
│  · Amihud ILLIQ     │◄────────────────┘
│  · Roll Spread      │
│  · OFI              │
└──────────┬──────────┘
           │ 16-dim feature matrix
           ▼
┌──────────────────────────────────────────────────────────┐
│             MODEL TRAINING  (models.py)                  │
│                                                          │
│  Walk-Forward Expanding Window (5 folds)                 │
│  with purge=5 bars + embargo=3 bars gap                  │
│                                                          │
│  ① Lasso (LassoCV + StandardScaler)                      │
│  ② XGBoost (n=150, max_depth=3, lr=0.05)                │
│  ③ Random Forest (hyperparams selected by TS-CV)         │
│  ④ LSTM Network (2-layer Recurrent Neural Network)       │
│                                                          │
│  Metrics: MAE · RMSE · R² · DA · IC                     │
└──────────┬───────────────────────────────────────────────┘
           │ predictions
           ▼
┌──────────────────────────────────────────────────────────┐
│          BACKTESTING  (backtest.py)                      │
│                                                          │
│  Position: sign(pred) × size                            │
│  Sizing:  fixed / Kelly / vol-target                    │
│  Costs:   commission (1 bps) + slippage (0.5 bps)       │
│                                                          │
│  Outputs: Sharpe · Calmar · Hit Rate · Profit Factor    │
│           Max Drawdown · Avg Trade · Rolling Sharpe      │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│         EVALUATION  (evaluation.py)                      │
│                                                          │
│  · Information Coefficient (IC) — Spearman rank corr   │
│  · IC Information Ratio (ICIR) — signal consistency     │
│  · Calibration — monotonicity of pred→realized buckets  │
│  · Turnover-adjusted alpha                              │
│  · Walk-forward fold summary stats                      │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Detailed Step-by-Step Usage Guide

### 1. Installation

Create a virtual environment and install the required scientific and quantitative dependencies:

```bash
# Initialize Python virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS/Linux

# Install requirements
pip install -r requirements.txt
```

*(Optional: Install PyTorch CPU to run the LSTM model as described in the [Fallback section](#-why-lasso-and-lstm-show-identical-results-known-fallback))*

### 2. Run the Full Research Pipeline
Execute the full quantitative pipeline using synthetic sample trades. This will compute all signals, train the models using walk-forward validation, perform a directional backtest, evaluate the results, and generate 10 publication-quality plots:

```bash
python -m src.main --source sample --plots --save
```

This command runs in ~35 seconds and writes all charts to `artifacts/plots/`.

### 3. Launch the Interactive Dashboards

#### Option A: Streamlit Dashboard (Python-only)
A rapid research dashboard to select data sources, adjust configurations, and trigger runs:
```bash
streamlit run src/dashboard.py
```
*Opens at `http://localhost:8501`*

#### Option B: React + TypeScript Production Frontend
A premium modern UI displaying metrics and feature importance.
First, start the FastAPI backend server:
```bash
# Terminal 1: Run the backend API
python -m src.api
```
Next, launch the Vite-based React application:
```bash
# Terminal 2: Run the frontend web app
cd frontend
npm run dev
```
*Opens at `http://localhost:5173`*

### 4. Fetch Real US Equity or Crypto Data

```bash
# Run on Apple Inc. (AAPL) Yahoo Finance trade bars
python -m src.main --source yahoo --symbol AAPL --plots

# Run on public Binance spot market aggTrades
python -m src.main --source binance --symbol BTCUSDT --start 2024-06-01T00:00:00 --end 2024-06-01T02:00:00 --plots
```

---

## 📊 Empirical Results & Analysis

The following results were obtained from the baseline research pipeline run on the default synthetic trade tape (1-minute resampled OHLCV bars):

### Model Comparison Summary

| Model | MAE | R² | Directional Accuracy (Hit Rate) | Information Coefficient (IC) | Annualized Return | Annualized Vol | Sharpe Ratio | Max Drawdown | Net P&L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lasso** (Linear) | `0.003349` | `-0.00077` | 45.83% | `-0.0075` | 0.00% | 0.00% | `NaN` | 0.00% | 0.00% |
| **XGBoost** (Tree) | `0.003712` | `-0.28242` | **52.50%** | `0.1031` | 57.57% | 2.93% | 19.62 | **-3.12%** | **+1.31%** |
| **Random Forest** | `0.003628` | `-0.18817` | 51.67% | **0.1075** | **59.99%** | 2.79% | **21.52** | -4.52% | **+1.37%** |
| **LSTM Network** | `0.003349` | `-0.00077` | 45.83% | `-0.0075` | 0.00% | 0.00% | `NaN` | 0.00% | 0.00% |

> [!NOTE]  
> **Understanding HFT Volatility and Sharpe Annualization:**  
> The Sharpe ratios (e.g. 19.62 and 21.52) in this test are extremely high. This is a mathematical artifact of high-frequency annualization on short-sample datasets (120 test bars = 2 hours) using the standard multiplier:  
> $$N_{\text{annual}} = 365 \times 1440 = 525,600$$  
> Volatility is multiplied by $\sqrt{525,600} \approx 725$. In live trading, transaction costs, execution delay, and market impact will significantly degrade these numbers.

### Key Observations
* **Predictive Value in Microstructure:** Both tree-based models (XGBoost and Random Forest) extract meaningful predictive signals, achieving a **Directional Accuracy above 51.5%** and a **Spearman Information Coefficient (IC) above 0.10**. In quantitative finance, an IC > 0.05 is considered highly useful.
* **Lasso Flatline:** Lasso prints a flat forecast return, resulting in zero position changes and 0% return. This indicates that microstructure dynamics (like order flow imbalance and bid-ask bounce) have non-linear relationships with returns that linear models cannot capture.
* **LSTM Fallback Identity:** The LSTM network yields identical metrics to Lasso because PyTorch was not present during the run, triggering the automatic Lasso fallback.

---

## 🎨 Visual Gallery of Research Charts

The research pipeline automatically generates these charts under `artifacts/plots/`:

### 1. Cumulative P&L & Drawdowns
Plots the cumulative strategy returns (net of 1.0 bps fee and 0.5 bps slippage) and visualizes the drawdown cycles.
![Cumulative P&L](artifacts/plots/equity_drawdown.png)

### 2. Model Performance comparison
Compares metrics side-by-side (Sharpe Ratio, Hit Rate, Max Drawdown) to help select the best candidate.
![Model Comparison](artifacts/plots/model_comparison.png)

### 3. Feature Importance (Random Forest)
Ranks engineered features by their Gini importance. Academic signals (Roll Spread, VPIN, Amihud, Kyle's Lambda) consistently show strong predictive contribution.
![Feature Importance](artifacts/plots/feature_importance.png)

### 4. Correlation Heatmap
Shows the linear relationships between all 16 engineered features to identify collinearity.
![Feature Correlation Heatmap](artifacts/plots/feature_correlation.png)

### 5. HMM Regime Overlay
Displays the identified market states (bearish, normal, bullish) overlaid onto the asset price chart.
![HMM Regime Overlay](artifacts/plots/regime_overlay.png)

### 6. Quintile Signal Calibration
Tests the monotonicity of predictions. A well-calibrated model shows actual returns increasing as predicted quintiles increase.
![Signal Calibration](artifacts/plots/calibration.png)

### 7. Rolling Sharpe Ratio
Tracks the stability of the model's Sharpe Ratio across expanding windows.
![Rolling Sharpe](artifacts/plots/rolling_sharpe.png)

---

## 📚 Academic Foundation & Microstructure Signals

The four microstructure signals implemented in `src/signals/microstructure.py` are sourced from peer-reviewed finance literature:

| Signal | Academic Source | Intuition |
| :--- | :--- | :--- |
| **Kyle's $\lambda$** | Kyle (1985) *Continuous Auctions and Insider Trading* | Measures price impact per unit of order flow — indicates illiquidity. |
| **Amihud ILLIQ** | Amihud (2002) *Illiquidity and Stock Returns* | Absolute return divided by dollar volume — measures price impact per dollar traded. |
| **Roll Spread** | Roll (1984) *A Simple Implicit Measure of the Bid-Ask Spread* | Estimates the bid-ask spread using serial price covariance (autocovariance). |
| **OFI** | Cont et al. (2014) *The Price Impact of Order Book Events* | Normalised signed order flow imbalance pressure. |

See the [Methodology Document](docs/METHODOLOGY.md) for full mathematical derivations.

---

## 🚀 Optional Enhancements

To unlock additional features, uncomment these lines in `requirements.txt` and install:

```bash
pip install shap       # SHAP-based feature importance explanations
pip install hmmlearn   # Unlocks full Hidden Markov Model regime detection
```

---

## 📖 References

* Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315–1335.
* Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*, 5(1), 31–56.
* Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance*, 39(4), 1127–1139.
* Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
