"""
Backtesting engine with realistic cost model, position sizing, and trade analytics.

Supports:
- Configurable slippage + commission (not just flat bps)
- Kelly-criterion and fixed-fractional position sizing
- Trade-level analytics (avg win, avg loss, profit factor, holding period)
- Rolling Sharpe and rolling drawdown
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    fee_bps: float = 1.0           # round-trip commission in basis points
    slippage_bps: float = 0.5      # slippage per side in basis points
    position_sizing: str = "fixed"  # "fixed", "kelly", or "volatility"
    kelly_fraction: float = 0.25   # fraction of full Kelly to use (quarter-Kelly is safer)
    vol_target: float = 0.10       # annualized vol target for vol-sizing
    max_position: float = 1.0      # max absolute position size (1.0 = fully invested)
    freq_per_day: int = 1440       # bars per day (1440 for 1-min bars)
    confidence_threshold: float = 0.0002 # 2 bps predicted return minimum to trade


def _compute_position_size(
    preds: pd.Series,
    realized: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """Compute position sizes based on the chosen sizing method."""
    direction = np.sign(preds)

    # Confidence Filter: Only trade if prediction magnitude is > threshold
    if config.confidence_threshold > 0:
        direction = direction.where(preds.abs() >= config.confidence_threshold, 0)

    if config.position_sizing == "fixed":
        return direction * config.max_position

    if config.position_sizing == "kelly":
        # Simplified Kelly: f* = mean(r) / var(r), scaled by kelly_fraction
        rolling_mean = realized.rolling(20, min_periods=5).mean()
        rolling_var = realized.rolling(20, min_periods=5).var()
        kelly = (rolling_mean / rolling_var.replace(0, np.nan)).fillna(0)
        kelly = kelly.clip(-config.max_position, config.max_position)
        size = direction * kelly.abs() * config.kelly_fraction
        return size.clip(-config.max_position, config.max_position)

    if config.position_sizing == "volatility":
        # Target vol: scale position inversely to realized volatility
        ann_factor = np.sqrt(365 * config.freq_per_day)
        rolling_vol = realized.rolling(20, min_periods=5).std() * ann_factor
        target_size = (config.vol_target / rolling_vol.replace(0, np.nan)).fillna(1.0)
        target_size = target_size.clip(0, config.max_position)
        return direction * target_size

    return direction * config.max_position


def run_directional_strategy(
    preds: pd.Series,
    realized: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """Run a long/short backtest with realistic costs and position sizing.

    Parameters
    ----------
    preds : predicted returns (signal)
    realized : actual forward returns
    config : backtest configuration (costs, sizing, etc.)

    Returns
    -------
    DataFrame with columns: pnl, equity, position, gross_pnl, costs
    """
    if config is None:
        config = BacktestConfig()

    preds = preds.loc[realized.index]
    position = _compute_position_size(preds, realized, config)

    # Gross P&L before costs
    gross_pnl = position * realized

    # Cost model: commission on turnover + slippage on turnover
    turnover = position.diff().abs().fillna(0)
    commission = turnover * (config.fee_bps / 10_000)
    slippage = turnover * (config.slippage_bps / 10_000)
    total_cost = commission + slippage

    pnl = gross_pnl - total_cost
    equity = (1 + pnl).cumprod()

    return pd.DataFrame({
        "pnl": pnl,
        "equity": equity,
        "position": position,
        "gross_pnl": gross_pnl,
        "costs": total_cost,
        "turnover": turnover,
    })


def rolling_sharpe(pnl: pd.Series, window: int = 60, freq_per_day: int = 1440) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    ann_factor = 365 * freq_per_day
    rolling_mean = pnl.rolling(window, min_periods=window // 2).mean() * ann_factor
    rolling_std = pnl.rolling(window, min_periods=window // 2).std() * np.sqrt(ann_factor)
    return (rolling_mean / rolling_std.replace(0, np.nan)).fillna(0)


def rolling_drawdown(equity: pd.Series) -> pd.Series:
    """Compute rolling drawdown from peak."""
    peak = equity.cummax()
    return (equity / peak) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1
    return dd.min()


def _extract_trades(perf: pd.DataFrame) -> pd.DataFrame:
    """Extract individual trades from position changes."""
    pos = perf["position"]
    # Detect trade entries: position changes from 0 or sign flip
    trade_starts = pos.ne(pos.shift(1).fillna(0))
    trade_starts = trade_starts[trade_starts].index

    trades = []
    for i, start_idx in enumerate(trade_starts):
        start_loc = perf.index.get_loc(start_idx)
        # Find end: next position change or end of series
        remaining = pos.iloc[start_loc + 1:]
        end_mask = remaining.ne(pos.iloc[start_loc])
        if end_mask.any():
            end_idx = end_mask.idxmax()
            end_loc = perf.index.get_loc(end_idx)
        else:
            end_loc = len(perf) - 1
            end_idx = perf.index[end_loc]

        trade_pnl = perf["pnl"].iloc[start_loc:end_loc + 1].sum()
        holding_bars = end_loc - start_loc + 1
        direction = "long" if pos.iloc[start_loc] > 0 else "short"

        trades.append({
            "entry_time": start_idx,
            "exit_time": end_idx,
            "direction": direction,
            "holding_bars": holding_bars,
            "pnl": trade_pnl,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "holding_bars", "pnl"]
    )


def summary(perf: pd.DataFrame, freq_per_day: int = 1440) -> dict:
    """Compute comprehensive performance summary."""
    pnl = perf["pnl"]
    ann_factor = 365 * freq_per_day

    # Core metrics
    mean_ret = pnl.mean() * ann_factor
    vol = pnl.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = mean_ret / vol if vol > 0 else np.nan
    hit_rate = (pnl > 0).mean()
    max_dd = _max_drawdown(perf["equity"])
    calmar = mean_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Trade-level analytics
    trades_df = _extract_trades(perf)
    n_trades = len(trades_df)
    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if n_trades > 0 else 0
    avg_loss = trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean() if n_trades > 0 else 0
    total_wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    total_losses = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    avg_holding = trades_df["holding_bars"].mean() if n_trades > 0 else 0

    # Cost analysis
    total_cost = perf["costs"].sum() if "costs" in perf.columns else 0
    total_gross = perf["gross_pnl"].sum() if "gross_pnl" in perf.columns else pnl.sum()

    return {
        "ann_return": mean_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "final_equity": perf["equity"].iloc[-1],
        "n_trades": n_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_bars": avg_holding,
        "total_cost": total_cost,
        "gross_pnl": total_gross,
        "net_pnl": pnl.sum(),
    }
