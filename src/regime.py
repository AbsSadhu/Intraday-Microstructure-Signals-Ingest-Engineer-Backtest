"""
Market regime detection using Hidden Markov Models.

Detects latent market states (e.g., trending, mean-reverting, volatile)
from observable features (returns, volatility).

Uses hmmlearn for HMM fitting; falls back to a simple volatility-based
regime classifier if hmmlearn is unavailable.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def detect_regimes_hmm(
    bars: pd.DataFrame,
    n_regimes: int = 3,
    features: list = None,
    random_state: int = 42,
) -> Tuple[pd.Series, object]:
    """Fit a Gaussian HMM to market data and return regime labels.

    Parameters
    ----------
    bars : OHLCV DataFrame with at least 'close' and 'volume' columns
    n_regimes : number of hidden states (2 = bull/bear, 3 adds sideways)
    features : list of column names to use as observable features
    random_state : for reproducibility

    Returns
    -------
    regime_labels : Series of integer regime labels (0, 1, 2, ...)
    model : fitted HMM model (or None if fallback was used)
    """
    # Prepare observable features
    obs = pd.DataFrame(index=bars.index)
    obs["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))
    obs["vol_proxy"] = obs["log_ret"].rolling(20, min_periods=5).std()
    if "volume" in bars.columns:
        obs["log_volume"] = np.log1p(bars["volume"])
    obs = obs.dropna()

    if features:
        obs = obs[features]

    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        model.fit(obs.values)
        labels = model.predict(obs.values)
        regime_series = pd.Series(labels, index=obs.index, name="regime")

        # Sort regimes by mean return so regime 0 = lowest return (bearish)
        means = obs["log_ret"].groupby(regime_series).mean().sort_values()
        label_map = {old: new for new, old in enumerate(means.index)}
        regime_series = regime_series.map(label_map)

        return regime_series, model

    except ImportError:
        # Fallback: simple volatility-based regime detection
        return _fallback_regime_detection(obs, n_regimes), None


def _fallback_regime_detection(obs: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
    """Simple quantile-based regime detection using volatility.

    Regime 0: Low vol (trending / calm)
    Regime 1: Medium vol (normal)
    Regime 2: High vol (volatile / crisis)
    """
    vol = obs["vol_proxy"]
    labels = pd.qcut(vol, q=n_regimes, labels=False, duplicates="drop")
    return pd.Series(labels, index=obs.index, name="regime")


def regime_summary(bars: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute summary statistics per detected regime.

    Returns
    -------
    DataFrame with regime-level stats: mean return, vol, duration, frequency
    """
    log_ret = np.log(bars["close"] / bars["close"].shift(1))
    df = pd.DataFrame({"log_ret": log_ret, "regime": regimes}).dropna()

    summary = df.groupby("regime").agg(
        mean_return=("log_ret", "mean"),
        volatility=("log_ret", "std"),
        count=("log_ret", "count"),
    )
    summary["frequency"] = summary["count"] / summary["count"].sum()
    summary["ann_return"] = summary["mean_return"] * 365 * 1440  # 1-min bars
    summary["ann_vol"] = summary["volatility"] * np.sqrt(365 * 1440)
    summary["sharpe"] = summary["ann_return"] / summary["ann_vol"].replace(0, np.nan)

    regime_names = {0: "bearish", 1: "normal", 2: "bullish"}
    if len(summary) <= 3:
        summary.index = summary.index.map(lambda x: regime_names.get(x, f"regime_{x}"))

    return summary
