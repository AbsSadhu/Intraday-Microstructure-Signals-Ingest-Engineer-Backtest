"""
Feature engineering for microstructure analysis.

Computes:
- Core bar features (returns, volatility, imbalance, VPIN, range, momentum)
- Advanced microstructure signals (Kyle's Lambda, Amihud, Roll Spread, OFI)
- Regime labels (HMM-based or fallback)
- Higher-moment return statistics
"""

import numpy as np
import pandas as pd

from .signals.microstructure import compute_all_microstructure_signals
from .regime import detect_regimes_hmm


def compute_bar_features(
    bars: pd.DataFrame,
    vol_window: int = 20,
    vpin_window: int = 20,
    mom_window: int = 5,
    include_advanced: bool = True,
    include_regime: bool = True,
) -> pd.DataFrame:
    """Compute microstructure-style features from resampled bars.

    Parameters
    ----------
    bars : OHLCV bars from resample_trades()
    include_advanced : if True, add Kyle's Lambda, Amihud, Roll Spread, OFI
    include_regime : if True, add HMM-based regime labels
    """
    df = bars.copy()
    df["mid"] = (df["high"] + df["low"]) / 2
    df["log_price"] = np.log(df["close"])
    df["log_ret"] = df["log_price"].diff()

    # --- Core features ---

    # Realized volatility proxy: rolling sqrt of sum of squared returns
    df["realized_vol"] = (
        df["log_ret"].rolling(vol_window).apply(lambda x: np.sqrt((x**2).sum()), raw=True)
    )

    # Volume imbalance and VPIN-style metric
    df["vol_imbalance"] = df["signed_volume"].fillna(0) / df["volume"].replace(0, np.nan)
    df["vol_imbalance"] = df["vol_imbalance"].fillna(0)
    abs_signed = df["signed_volume"].abs().rolling(vpin_window).sum()
    vol_roll = df["volume"].rolling(vpin_window).sum().replace(0, np.nan)
    df["vpin"] = (abs_signed / vol_roll).fillna(0)

    # Range and momentum
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["range_z"] = (df["hl_range"] - df["hl_range"].rolling(vol_window).mean()) / df[
        "hl_range"
    ].rolling(vol_window).std(ddof=0)
    df["momentum"] = df["close"].pct_change(periods=mom_window)
    df["log_mom"] = df["log_price"].diff(periods=mom_window)

    # Liquidity proxies
    df["tick_rate"] = df["trade_count"].rolling(5).mean()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(vol_window).mean()) / df[
        "volume"
    ].rolling(vol_window).std(ddof=0)

    # Higher moment estimates for return distribution
    df["ret_skew"] = df["log_ret"].rolling(vol_window).skew()
    df["ret_kurt"] = df["log_ret"].rolling(vol_window).kurt()

    # --- Feature list ---
    feature_cols = [
        "log_ret",
        "realized_vol",
        "vol_imbalance",
        "vpin",
        "hl_range",
        "range_z",
        "momentum",
        "log_mom",
        "tick_rate",
        "volume_z",
        "ret_skew",
        "ret_kurt",
    ]

    # --- Advanced microstructure signals ---
    if include_advanced:
        micro_signals = compute_all_microstructure_signals(bars)
        for col in micro_signals.columns:
            df[col] = micro_signals[col]
            feature_cols.append(col)

    # --- Regime labels ---
    if include_regime:
        try:
            regimes, _ = detect_regimes_hmm(bars, n_regimes=3)
            df["regime"] = regimes
            feature_cols.append("regime")
        except Exception:
            pass  # Skip regime if it fails (e.g., too few data points)

    features = df[feature_cols]

    # Replace ±inf with NaN (can arise in microstructure signals on zero-volume bars)
    features = features.replace([float("inf"), float("-inf")], float("nan"))

    # Winsorize extreme values at 99.9th/0.1th percentile to avoid sklearn crashes
    lower = features.quantile(0.001)
    upper = features.quantile(0.999)
    features = features.clip(lower=lower, upper=upper, axis=1)

    return features.dropna()


def prepare_supervised(features: pd.DataFrame, target: pd.Series, horizon: int = 1) -> pd.DataFrame:
    """Align features with forward-looking target."""
    aligned = features.copy()
    y = target.shift(-horizon)
    df = aligned.join(y.rename("target"))
    return df.dropna()
