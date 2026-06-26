"""
Advanced market microstructure signals from academic literature.

Implements:
- Kyle's Lambda (1985) — price impact / illiquidity measure
- Amihud Illiquidity Ratio (2002) — |return| / volume
- Roll's Implied Spread (1984) — bid-ask spread from autocovariance
- Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) — signed volume pressure
"""

import numpy as np
import pandas as pd


def kyles_lambda(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Kyle's Lambda — regression slope of price change on signed order flow.

    λ = Cov(ΔP, SignedFlow) / Var(SignedFlow)

    Higher λ means trades have more price impact (illiquid market).
    From Albert Kyle, "Continuous Auctions and Insider Trading" (1985).

    Parameters
    ----------
    bars : DataFrame with 'close' and 'signed_volume' columns
    window : rolling window for estimation

    Returns
    -------
    Series of rolling Kyle's Lambda estimates
    """
    dp = bars["close"].diff()
    sv = bars["signed_volume"].fillna(0)

    cov = dp.rolling(window, min_periods=window // 2).cov(sv)
    var = sv.rolling(window, min_periods=window // 2).var()

    lam = (cov / var.replace(0, np.nan)).fillna(0)
    lam.name = "kyle_lambda"
    return lam


def amihud_illiquidity(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Amihud Illiquidity Ratio — |return| / dollar volume.

    ILLIQ = (1/N) * Σ |r_t| / Volume_t

    Higher values indicate lower liquidity.
    From Yakov Amihud, "Illiquidity and Stock Returns" (2002).

    Parameters
    ----------
    bars : DataFrame with 'close' and 'volume' columns
    window : rolling window for averaging

    Returns
    -------
    Series of rolling Amihud illiquidity estimates
    """
    ret = bars["close"].pct_change().abs()
    dollar_vol = bars["volume"] * bars["close"]
    ratio = (ret / dollar_vol.replace(0, np.nan)).fillna(0)

    amihud = ratio.rolling(window, min_periods=window // 2).mean()
    amihud.name = "amihud_illiq"
    return amihud


def roll_implied_spread(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Roll's Implied Spread — estimate bid-ask spread from price autocovariance.

    Spread = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))  if covariance is negative
           = 0                                  otherwise

    From Richard Roll, "A Simple Implicit Measure of the Effective Bid-Ask Spread" (1984).

    Parameters
    ----------
    bars : DataFrame with 'close' column
    window : rolling window for autocovariance estimation

    Returns
    -------
    Series of rolling Roll spread estimates (as fraction of price)
    """
    dp = bars["close"].diff()
    dp_lag = dp.shift(1)

    cov = dp.rolling(window, min_periods=window // 2).cov(dp_lag)

    # Spread is only defined when autocovariance is negative (bid-ask bounce)
    spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0)
    spread_pct = spread / bars["close"].replace(0, np.nan).values

    result = pd.Series(spread_pct, index=bars.index, name="roll_spread")
    return result.fillna(0)


def order_flow_imbalance(
    bars: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """Order Flow Imbalance (OFI) — normalized signed volume pressure.

    OFI_t = SignedVolume_t / TotalVolume_t (per bar)
    Smoothed OFI = rolling mean of OFI over window

    Inspired by Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events" (2014).

    Parameters
    ----------
    bars : DataFrame with 'signed_volume' and 'volume' columns
    window : smoothing window

    Returns
    -------
    Series of smoothed OFI values in [-1, 1]
    """
    raw_ofi = bars["signed_volume"].fillna(0) / bars["volume"].replace(0, np.nan)
    raw_ofi = raw_ofi.fillna(0)

    smoothed = raw_ofi.rolling(window, min_periods=window // 2).mean()
    smoothed.name = "ofi"
    return smoothed


def compute_all_microstructure_signals(
    bars: pd.DataFrame,
    kyle_window: int = 20,
    amihud_window: int = 20,
    roll_window: int = 20,
    ofi_window: int = 10,
) -> pd.DataFrame:
    """Compute all microstructure signals and return as a DataFrame.

    Parameters
    ----------
    bars : OHLCV bars with 'signed_volume' column

    Returns
    -------
    DataFrame with columns: kyle_lambda, amihud_illiq, roll_spread, ofi
    """
    signals = pd.DataFrame(index=bars.index)
    signals["kyle_lambda"] = kyles_lambda(bars, window=kyle_window)
    signals["amihud_illiq"] = amihud_illiquidity(bars, window=amihud_window)
    signals["roll_spread"] = roll_implied_spread(bars, window=roll_window)
    signals["ofi"] = order_flow_imbalance(bars, window=ofi_window)
    return signals
