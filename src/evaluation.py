"""
Evaluation framework for quantitative signal quality.

Computes:
- Information Coefficient (IC) and IC Information Ratio (ICIR)
- Cumulative IC over time
- Calibration analysis (predicted vs. actual return buckets)
- Walk-forward fold summary statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman rank correlation between predicted and realized returns."""
    return y_true.corr(y_pred, method="spearman")


def rolling_ic(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling Information Coefficient over a window."""
    df = pd.DataFrame({"actual": y_true, "pred": y_pred})
    return df["actual"].rolling(window, min_periods=window // 2).corr(df["pred"])


def ic_information_ratio(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC) — measures signal consistency."""
    std = ic_series.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return ic_series.mean() / std


def cumulative_ic(y_true: pd.Series, y_pred: pd.Series, window: int = 20) -> pd.Series:
    """Compute cumulative average IC over expanding windows."""
    ric = rolling_ic(y_true, y_pred, window=window)
    return ric.expanding().mean()


def calibration_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Bucket predictions into quantiles and compute average realized return per bucket.

    A well-calibrated model should show monotonically increasing realized
    returns from the lowest to highest predicted return bucket.
    """
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bucket"] = pd.qcut(df["pred"], q=n_buckets, labels=False, duplicates="drop")

    cal = df.groupby("bucket").agg(
        avg_predicted=("pred", "mean"),
        avg_realized=("actual", "mean"),
        std_realized=("actual", "std"),
        count=("actual", "count"),
    ).reset_index()

    # Monotonicity score: fraction of adjacent buckets with correct ordering
    if len(cal) > 1:
        diffs = cal["avg_realized"].diff().dropna()
        cal_score = (diffs > 0).mean()
    else:
        cal_score = np.nan

    cal.attrs["monotonicity_score"] = cal_score
    return cal


def walk_forward_summary(fold_results: List[dict]) -> Dict[str, float]:
    """Aggregate walk-forward fold results into a summary."""
    if not fold_results:
        return {}

    df = pd.DataFrame(fold_results)
    summary = {}
    for col in ["mae", "rmse", "r2", "directional_accuracy", "information_coefficient"]:
        if col in df.columns:
            summary[f"wf_{col}_mean"] = df[col].mean()
            summary[f"wf_{col}_std"] = df[col].std()
            summary[f"wf_{col}_min"] = df[col].min()
            summary[f"wf_{col}_max"] = df[col].max()

    summary["wf_n_folds"] = len(df)
    summary["wf_total_test_samples"] = int(df["test_size"].sum()) if "test_size" in df.columns else 0
    return summary


def turnover_adjusted_alpha(
    preds: pd.Series,
    realized: pd.Series,
    cost_bps: float = 1.5,
) -> float:
    """Compute alpha after subtracting estimated transaction costs from turnover.

    This gives a more realistic picture of signal value than raw IC.
    """
    position = np.sign(preds)
    gross_alpha = (position * realized).sum()
    turnover = position.diff().abs().sum()
    cost = turnover * (cost_bps / 10_000)
    return gross_alpha - cost


def full_evaluation(
    y_true: pd.Series,
    y_pred: pd.Series,
    fold_results: Optional[List[dict]] = None,
) -> Dict:
    """Run the full evaluation suite and return a structured report."""
    ic = information_coefficient(y_true, y_pred)
    ric = rolling_ic(y_true, y_pred)
    icir = ic_information_ratio(ric.dropna())
    cum_ic = cumulative_ic(y_true, y_pred)
    cal_table = calibration_table(y_true, y_pred)
    adj_alpha = turnover_adjusted_alpha(y_pred, y_true)

    report = {
        "ic": ic,
        "icir": icir,
        "turnover_adjusted_alpha": adj_alpha,
        "calibration_monotonicity": cal_table.attrs.get("monotonicity_score", np.nan),
        "rolling_ic": ric,
        "cumulative_ic": cum_ic,
        "calibration_table": cal_table,
    }

    if fold_results:
        report["walk_forward_summary"] = walk_forward_summary(fold_results)

    return report
