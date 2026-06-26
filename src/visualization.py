"""
Rich visualization suite for quantitative analysis.

Auto-generates a full research-quality report with 8+ chart types,
saved as PNG files to artifacts/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from .config import OUTPUT_DIR

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Style config
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 10,
})

# Color palette
COLORS = {
    "primary": "#58a6ff",
    "secondary": "#f0883e",
    "success": "#3fb950",
    "danger": "#f85149",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "yellow": "#d29922",
    "grid": "#21262d",
}


def plot_equity_with_drawdown(
    perf: pd.DataFrame,
    title: str = "Strategy Equity Curve",
    save: bool = True,
) -> plt.Figure:
    """Equity curve with drawdown shading underneath."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.05})

    # Equity curve
    ax1.plot(perf.index, perf["equity"], color=COLORS["primary"], linewidth=1.5, label="Equity")
    ax1.fill_between(perf.index, 1, perf["equity"],
                     where=perf["equity"] >= 1, alpha=0.15, color=COLORS["success"])
    ax1.fill_between(perf.index, 1, perf["equity"],
                     where=perf["equity"] < 1, alpha=0.15, color=COLORS["danger"])
    ax1.axhline(y=1.0, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax1.set_ylabel("Equity")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    peak = perf["equity"].cummax()
    dd = (perf["equity"] / peak) - 1.0
    ax2.fill_between(perf.index, 0, dd, color=COLORS["danger"], alpha=0.4)
    ax2.plot(perf.index, dd, color=COLORS["danger"], linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    return fig


def plot_feature_correlation(
    features: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                ax=ax, square=True,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "feature_correlation.png", dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance",
    save: bool = True,
) -> Optional[plt.Figure]:
    """Horizontal bar chart of feature importance."""
    if importance is None or importance.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(5, len(importance) * 0.35)))

    sorted_imp = importance.sort_values(ascending=True)
    colors = [COLORS["primary"]] * len(sorted_imp)
    colors[-1] = COLORS["success"]  # highlight top feature
    colors[-2] = COLORS["cyan"] if len(colors) > 1 else COLORS["primary"]

    ax.barh(sorted_imp.index, sorted_imp.values, color=colors, height=0.6)
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_scatter(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Predicted vs. actual return scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color=COLORS["primary"], edgecolors="none")

    # Regression line
    if len(y_true) > 2:
        z = np.polyfit(y_true.values, y_pred.values, 1)
        p = np.poly1d(z)
        x_range = np.linspace(y_true.min(), y_true.max(), 100)
        ax.plot(x_range, p(x_range), color=COLORS["secondary"], linewidth=2,
                label=f"Fit: y = {z[0]:.3f}x + {z[1]:.6f}")

    # 45-degree line (perfect prediction)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color=COLORS["success"], alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(f"{model_name} — Predicted vs Actual", fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / f"pred_scatter_{model_name.lower().replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    rolling_sharpe_series: pd.Series,
    title: str = "Rolling Sharpe Ratio (60-bar window)",
    save: bool = True,
) -> plt.Figure:
    """Rolling Sharpe ratio over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(rolling_sharpe_series.index, rolling_sharpe_series, color=COLORS["primary"], linewidth=1)
    ax.fill_between(rolling_sharpe_series.index, 0, rolling_sharpe_series,
                    where=rolling_sharpe_series >= 0, alpha=0.2, color=COLORS["success"])
    ax.fill_between(rolling_sharpe_series.index, 0, rolling_sharpe_series,
                    where=rolling_sharpe_series < 0, alpha=0.2, color=COLORS["danger"])
    ax.axhline(y=0, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_ic(
    cum_ic: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Cumulative Information Coefficient plot."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(cum_ic.index, cum_ic, color=COLORS["purple"], linewidth=1.5)
    ax.fill_between(cum_ic.index, 0, cum_ic,
                    where=cum_ic >= 0, alpha=0.15, color=COLORS["purple"])
    ax.axhline(y=0, color=COLORS["grid"], linestyle="--", alpha=0.5)
    ax.set_ylabel("Cumulative IC")
    ax.set_title("Cumulative Information Coefficient", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "cumulative_ic.png", dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(
    cal_table: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Calibration plot: predicted vs realized return per bucket."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(len(cal_table))
    ax.bar(x, cal_table["avg_realized"], color=COLORS["primary"], alpha=0.7,
           label="Avg Realized Return", width=0.4, align="center")
    ax.bar([i + 0.4 for i in x], cal_table["avg_predicted"], color=COLORS["secondary"],
           alpha=0.7, label="Avg Predicted Return", width=0.4, align="center")
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_xlabel("Prediction Quantile (Q1 = lowest predicted)")
    ax.set_ylabel("Return")
    score = cal_table.attrs.get("monotonicity_score", "N/A")
    ax.set_title(f"Signal Calibration — Monotonicity: {score:.0%}" if isinstance(score, float)
                 else "Signal Calibration",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    return fig


def plot_return_distribution(
    returns: pd.Series,
    title: str = "Return Distribution",
    save: bool = True,
) -> plt.Figure:
    """Histogram of returns with normal distribution overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(returns.dropna(), bins=80, density=True, alpha=0.6,
            color=COLORS["primary"], edgecolor="none", label="Actual")

    # Normal overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    ax.plot(x, normal, color=COLORS["secondary"], linewidth=2, label=f"Normal(μ={mu:.5f}, σ={sigma:.5f})")

    # Stats annotation
    skew = returns.skew()
    kurt = returns.kurtosis()
    ax.text(0.98, 0.95, f"Skew: {skew:.3f}\nKurtosis: {kurt:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=COLORS["cyan"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=COLORS["cyan"], alpha=0.8))

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "return_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    report: dict,
    save: bool = True,
) -> plt.Figure:
    """Side-by-side model comparison card."""
    model_names = [k for k in report if k not in ("latest_signal", "best_model")]
    if not model_names:
        return None

    metrics_to_show = ["sharpe", "hit_rate", "max_drawdown", "final_equity"]
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(16, 5))

    for i, metric in enumerate(metrics_to_show):
        ax = axes[i]
        vals = []
        names = []
        for name in model_names:
            perf = report[name].get("perf", {})
            v = perf.get(metric, 0)
            vals.append(v)
            names.append(name.upper())

        colors = [COLORS["primary"], COLORS["secondary"], COLORS["purple"]][:len(vals)]
        ax.bar(names, vals, color=colors, alpha=0.8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    return fig


def plot_regime_overlay(
    bars: pd.DataFrame,
    regimes: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Price chart with regime-colored background."""
    fig, ax = plt.subplots(figsize=(14, 5))

    price = bars["close"]
    regime_colors = {0: COLORS["danger"], 1: COLORS["yellow"], 2: COLORS["success"]}

    # Color background by regime
    aligned = regimes.reindex(price.index).fillna(method="ffill")
    for regime_val, color in regime_colors.items():
        mask = aligned == regime_val
        if mask.any():
            ax.fill_between(price.index, price.min() * 0.999, price.max() * 1.001,
                           where=mask, alpha=0.1, color=color,
                           label=f"Regime {regime_val}")

    ax.plot(price.index, price, color=COLORS["primary"], linewidth=1)
    ax.set_ylabel("Price")
    ax.set_title("Price with Market Regime Overlay", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "regime_overlay.png", dpi=150, bbox_inches="tight")
    return fig


def generate_full_report(
    perf: pd.DataFrame,
    features: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    report: dict,
    importance: Optional[pd.Series] = None,
    rolling_sharpe_series: Optional[pd.Series] = None,
    cum_ic: Optional[pd.Series] = None,
    cal_table: Optional[pd.DataFrame] = None,
    bars: Optional[pd.DataFrame] = None,
    regimes: Optional[pd.Series] = None,
) -> List[Path]:
    """Generate all charts and save to artifacts/plots/. Returns list of saved paths."""
    saved = []

    try:
        plot_equity_with_drawdown(perf, title=f"{model_name} Equity Curve")
        saved.append(PLOT_DIR / "equity_drawdown.png")
    except Exception as e:
        print(f"  Warning: equity plot failed: {e}")

    try:
        plot_feature_correlation(features)
        saved.append(PLOT_DIR / "feature_correlation.png")
    except Exception as e:
        print(f"  Warning: correlation plot failed: {e}")

    try:
        if importance is not None:
            plot_feature_importance(importance, title=f"{model_name} Feature Importance")
            saved.append(PLOT_DIR / "feature_importance.png")
    except Exception as e:
        print(f"  Warning: importance plot failed: {e}")

    try:
        plot_prediction_scatter(y_true, y_pred, model_name=model_name)
        saved.append(PLOT_DIR / f"pred_scatter_{model_name.lower().replace(' ', '_')}.png")
    except Exception as e:
        print(f"  Warning: scatter plot failed: {e}")

    try:
        if rolling_sharpe_series is not None:
            plot_rolling_sharpe(rolling_sharpe_series)
            saved.append(PLOT_DIR / "rolling_sharpe.png")
    except Exception as e:
        print(f"  Warning: rolling sharpe plot failed: {e}")

    try:
        if cum_ic is not None:
            plot_cumulative_ic(cum_ic)
            saved.append(PLOT_DIR / "cumulative_ic.png")
    except Exception as e:
        print(f"  Warning: cumulative IC plot failed: {e}")

    try:
        if cal_table is not None:
            plot_calibration(cal_table)
            saved.append(PLOT_DIR / "calibration.png")
    except Exception as e:
        print(f"  Warning: calibration plot failed: {e}")

    try:
        plot_return_distribution(y_true, title="Realized Return Distribution")
        saved.append(PLOT_DIR / "return_distribution.png")
    except Exception as e:
        print(f"  Warning: return dist plot failed: {e}")

    try:
        plot_model_comparison(report)
        saved.append(PLOT_DIR / "model_comparison.png")
    except Exception as e:
        print(f"  Warning: model comparison plot failed: {e}")

    try:
        if bars is not None and regimes is not None:
            plot_regime_overlay(bars, regimes)
            saved.append(PLOT_DIR / "regime_overlay.png")
    except Exception as e:
        print(f"  Warning: regime overlay plot failed: {e}")

    plt.close("all")
    return saved
