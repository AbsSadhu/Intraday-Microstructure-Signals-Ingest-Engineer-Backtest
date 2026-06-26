# Methodology: Intraday Market Microstructure Research Platform

This document describes the mathematical foundations behind each component of the pipeline.

---

## 1. Data Generation & Preprocessing

### 1.1 Synthetic Trade Tape (Ornstein-Uhlenbeck Process)

When no real data is available, a synthetic trade tape is generated using a mean-reverting log-price process:

$$d\ln P_t = -\kappa (\ln P_t - \ln P_0)\,dt + \sigma\,dW_t$$

where:
- $\kappa = 0.05$ — mean-reversion speed (controls how quickly price reverts to $P_0$)
- $\sigma = 0.0008$ — volatility per tick
- $dW_t$ — Brownian motion increment

Trade prices are perturbed by a half-spread $s/2$ depending on buyer/seller initiation:

$$p_{\text{trade}} = P_t + \frac{s}{2} \cdot \mathbf{1}_{\text{buy}} - \frac{s}{2} \cdot \mathbf{1}_{\text{sell}}$$

Trade sizes follow a log-normal distribution: $Q \sim \text{LogNormal}(-2.7, 0.6)$, calibrated to typical crypto trade size distributions.

### 1.2 Bar Resampling

Raw trades $(t_i, p_i, q_i, \text{side}_i)$ are aggregated into time bars of width $\Delta t$ (default: 1 minute):

$$\text{SignedVolume}_t = \sum_{i \in [t, t+\Delta t]} q_i \cdot \text{sign}_i \qquad \text{where sign} = +1 \text{ (buy)}, -1 \text{ (sell)}$$

Each bar carries: $\{O, H, L, C, V, \text{SignedVolume}, \text{TradeCount}\}$.

---

## 2. Core Features

### 2.1 Log Returns

$$r_t = \ln\frac{C_t}{C_{t-1}}$$

Used instead of arithmetic returns for additive decomposition over time.

### 2.2 Realized Volatility

Approximation of integrated variance using rolling squared returns:

$$\hat{\sigma}_t = \sqrt{\sum_{i=t-N+1}^{t} r_i^2}$$

This is the sum-of-squares estimator — a discrete version of the quadratic variation. Window $N = 20$ bars.

### 2.3 Volume Imbalance

Per-bar imbalance between buy and sell volume:

$$\text{Imbalance}_t = \frac{\text{SignedVolume}_t}{V_t} \in [-1, 1]$$

A value near +1 indicates almost all volume was buyer-initiated.

### 2.4 VPIN Proxy (Volume-Synchronized Probability of Informed Trading)

Rolling estimate of the fraction of informed trading, adapted from Easley et al. (2012):

$$\text{VPIN}_t = \frac{\sum_{i=t-N+1}^{t} |\text{SignedVolume}_i|}{\sum_{i=t-N+1}^{t} V_i}$$

High VPIN suggests one-sided, potentially informed order flow. Window $N = 20$.

### 2.5 High-Low Range Z-Score

Normalised intrabar price range, measuring relative volatility:

$$z_t^{\text{range}} = \frac{(H_t - L_t)/C_t - \mu_{\text{range}}}{\sigma_{\text{range}}}$$

where $\mu$ and $\sigma$ are computed over a rolling window of 20 bars.

### 2.6 Momentum

Short-term price momentum over $k$ bars:

$$\text{Mom}_t = \frac{C_t - C_{t-k}}{C_{t-k}}$$

Also computed as log-momentum: $\text{LogMom}_t = \ln C_t - \ln C_{t-k}$ for $k=5$.

### 2.7 Tick Rate

Rolling mean of trade count per bar — a proxy for market activity and attention:

$$\text{TickRate}_t = \frac{1}{5}\sum_{i=t-4}^{t} \text{TradeCount}_i$$

### 2.8 Volume Z-Score

$$z_t^{\text{vol}} = \frac{V_t - \mu_V}{\sigma_V}$$

Identifies abnormally high or low volume relative to the recent baseline.

### 2.9 Return Skewness and Kurtosis

Rolling higher moments of the log-return distribution:

$$\text{Skew}_t = \frac{E[(r - \mu)^3]}{\sigma^3}, \qquad \text{Kurt}_t = \frac{E[(r - \mu)^4]}{\sigma^4} - 3$$

Excess kurtosis (leptokurtosis) is typical of financial returns; changes in skewness may precede trend reversals.

---

## 3. Advanced Microstructure Signals

### 3.1 Kyle's Lambda (Price Impact)

From Albert Kyle's seminal 1985 paper, the price impact coefficient $\lambda$ is estimated by OLS:

$$\Delta P_t = \lambda \cdot \text{SignedVolume}_t + \epsilon_t$$

Estimated as a rolling regression:

$$\hat{\lambda}_t = \frac{\text{Cov}_t(\Delta P, \text{SV})}{\text{Var}_t(\text{SV})}$$

**Interpretation:** Higher $\lambda$ means trades move prices more — the market is less liquid and informed traders have more price impact. Rolling window $N = 20$.

*Reference: Kyle, A. S. (1985). Continuous Auctions and Insider Trading. Econometrica, 53(6), 1315–1335.*

### 3.2 Amihud Illiquidity Ratio

Yakov Amihud's (2002) illiquidity measure relates absolute returns to dollar volume:

$$\text{ILLIQ}_t = \frac{1}{N}\sum_{i=t-N+1}^{t} \frac{|r_i|}{V_i \cdot C_i}$$

**Interpretation:** The ratio measures how much price moves per dollar of trading volume. Higher values indicate lower liquidity. It is a proxy for Kyle's $\lambda$ when order-flow direction is unavailable.

*Reference: Amihud, Y. (2002). Illiquidity and stock returns. Journal of Financial Markets, 5(1), 31–56.*

### 3.3 Roll's Implied Spread

Richard Roll (1984) showed that the bid-ask spread can be estimated purely from the serial covariance of price changes, without needing quote data:

$$\hat{s} = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}$$

**Intuition:** If a trade alternates between the bid and ask (a "bounce"), consecutive price changes will be negatively autocorrelated. The magnitude of this covariance is related to half the spread.

The spread is only defined when $\text{Cov} < 0$; otherwise it is set to zero.

*Reference: Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. Journal of Finance, 39(4), 1127–1139.*

### 3.4 Order Flow Imbalance (OFI)

Inspired by Cont, Kukanov & Stoikov (2014), the OFI captures the pressure of incoming orders relative to total volume:

$$\text{OFI}_t = \frac{\text{SignedVolume}_t}{V_t}$$

Smoothed with a rolling mean over $N = 10$ bars to reduce noise.

**Interpretation:** A strongly positive OFI indicates persistent buy-side pressure and is associated with upward price pressure in the near term.

*Reference: Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. Journal of Financial Econometrics, 12(1), 47–88.*

---

## 4. Regime Detection

### Hidden Markov Model

A Gaussian HMM with $K = 3$ hidden states is fit to the observed sequence of (log-return, rolling-vol, log-volume) vectors:

$$P(\mathbf{x}_1, \ldots, \mathbf{x}_T) = \sum_{\mathbf{z}} \prod_{t=1}^{T} p(\mathbf{x}_t \mid z_t)\, p(z_t \mid z_{t-1})$$

where each emission distribution is multivariate Gaussian: $p(\mathbf{x}_t \mid z_t = k) = \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

Parameters $\{\pi, A, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$ are estimated by the Baum-Welch (EM) algorithm. The Viterbi algorithm decodes the most likely state sequence.

States are relabelled post-fit by their mean return: regime 0 (bearish) → regime 1 (normal) → regime 2 (bullish).

**Fallback:** If `hmmlearn` is not installed, states are assigned by volatility quantile (low / medium / high vol).

---

## 5. Labeling

The prediction target is the $h$-bar forward percentage return:

$$y_t = \frac{C_{t+h} - C_t}{C_t}$$

where $h$ is the horizon (default: 5 bars = 5 minutes for 1-min bars). The sign of $y_t$ determines the trade direction in the backtest.

---

## 6. Model Training & Validation

### 6.1 Walk-Forward Expanding Window

To avoid look-ahead bias, models are evaluated using an expanding training window:

```
Fold 1:  [====TRAIN====] [--TEST--]
Fold 2:  [======TRAIN======] [--TEST--]
Fold 3:  [========TRAIN========] [--TEST--]
...
```

### 6.2 Purge and Embargo

Inspired by López de Prado (2018), a **purge gap** of 5 bars is removed from the end of each training set, and an **embargo gap** of 3 bars is removed from the start of each test set. This prevents information from overlapping labels leaking between train and test:

```
[=TRAIN=] [PURGE] [EMBG] [TEST]
              ↑       ↑
          5 bars   3 bars
```

Without purging, label overlap from the $h$-step lookahead creates spurious in-sample performance.

### 6.3 Models

**Lasso (L1-regularised linear regression):**

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\| \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1$$

Features are standardised ($z$-scored) before fitting. $\alpha$ is selected by 3-fold time-series CV.

**XGBoost:** Gradient boosted decision trees with `n_estimators=150`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`. Shallow trees reduce overfitting on financial time series.

**Random Forest:** Ensemble of trees with hyperparameters selected by time-series CV over three candidate configurations. Uses `n_jobs=-1` for parallel fitting.

---

## 7. Backtesting

### 7.1 Position Sizing

**Fixed:** $\text{pos}_t = \text{sign}(\hat{y}_t)$

**Kelly Criterion (Quarter-Kelly):**

$$f^* = \frac{\mu_r}{\sigma_r^2}, \qquad \text{pos}_t = \text{sign}(\hat{y}_t) \cdot \min(f^*/4,\, 1)$$

**Volatility Targeting:**

$$\text{pos}_t = \text{sign}(\hat{y}_t) \cdot \frac{\sigma_{\text{target}}}{\hat{\sigma}_t^{\text{ann}}}$$

### 7.2 Cost Model

Round-trip cost per position change:

$$\text{cost}_t = |\Delta \text{pos}_t| \cdot \left(\frac{c_{\text{fee}}}{10{,}000} + \frac{c_{\text{slip}}}{10{,}000}\right)$$

Net P&L per bar: $\pi_t = \text{pos}_t \cdot r_t^{\text{realized}} - \text{cost}_t$

### 7.3 Performance Metrics

| Metric | Formula |
|--------|---------|
| Annualised Return | $\bar{\pi} \cdot N_{\text{annual}}$ |
| Annualised Volatility | $\text{std}(\pi) \cdot \sqrt{N_{\text{annual}}}$ |
| Sharpe Ratio | $\frac{\bar{\pi}_{\text{ann}}}{\hat{\sigma}_{\text{ann}}}$ |
| Calmar Ratio | $\frac{\bar{\pi}_{\text{ann}}}{\text{MDD}}$ |
| Max Drawdown | $\min_t \left(\frac{E_t}{\max_{s\le t} E_s} - 1\right)$ |
| Profit Factor | $\frac{\sum \pi_t^+}{\sum |\pi_t^-|}$ |

where $N_{\text{annual}} = 365 \times 1440$ for 1-minute bars.

---

## 8. Signal Evaluation

### 8.1 Information Coefficient (IC)

Spearman rank correlation between predictions and realisations:

$$\text{IC} = \rho_S(\hat{y}_t, y_t)$$

IC > 0.05 is considered a useful signal in practice.

### 8.2 IC Information Ratio (ICIR)

Measures signal *consistency* across time:

$$\text{ICIR} = \frac{E[\text{IC}_t]}{\text{std}(\text{IC}_t)}$$

An ICIR > 0.5 indicates the signal is stable across market regimes.

### 8.3 Calibration

Predictions are sorted into $Q = 5$ quantile buckets; the average realised return per bucket is computed. A well-calibrated model exhibits **monotonically increasing** realised returns from Q1 (lowest predicted) to Q5 (highest predicted).

The monotonicity score is:

$$m = \frac{1}{Q-1}\sum_{k=1}^{Q-1} \mathbf{1}\left[\bar{r}_{k+1} > \bar{r}_k\right]$$

$m = 1.0$ means perfect rank ordering.

---

## References

1. Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315–1335.
2. Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
3. Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*, 39(4), 1127–1139.
4. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
5. Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high frequency world. *Review of Financial Studies*, 25(5), 1457–1493.
6. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Chapter 7: Cross-Validation in Finance)
7. Hasbrouck, J. (2007). *Empirical Market Microstructure*. Oxford University Press.
