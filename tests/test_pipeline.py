"""
Test suite for the Microstructure Research Platform.

Covers: data loading, feature engineering, signal computation,
model training, backtesting, and evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_fetch import load_local_trades, resample_trades
from src.features import compute_bar_features, prepare_supervised
from src.labeling import forward_returns
from src.models import train_linear_model, train_tree_model, train_forest_model
from src.backtest import BacktestConfig, run_directional_strategy, summary, rolling_sharpe
from src.evaluation import (
    information_coefficient, rolling_ic, calibration_table,
    walk_forward_summary, full_evaluation,
)
from src.signals.microstructure import (
    kyles_lambda, amihud_illiquidity, roll_implied_spread,
    order_flow_imbalance, compute_all_microstructure_signals,
)
from src.regime import detect_regimes_hmm, regime_summary


# ── Fixtures ──

@pytest.fixture(scope="module")
def trades():
    return load_local_trades()


@pytest.fixture(scope="module")
def bars(trades):
    return resample_trades(trades)


@pytest.fixture(scope="module")
def features(bars):
    return compute_bar_features(bars, include_advanced=True, include_regime=False)


@pytest.fixture(scope="module")
def dataset(features, bars):
    target = forward_returns(bars, horizon=3)
    return prepare_supervised(features, target, horizon=3)


# ── Data Loading ──

def test_sample_pipeline(trades, bars, features, dataset):
    """End-to-end smoke test on sample data."""
    assert not trades.empty
    assert not bars.empty
    assert not features.empty
    assert not dataset.empty


def test_resample_rule_override(trades):
    bars = resample_trades(trades, rule="5min")
    feats = compute_bar_features(bars, include_advanced=False, include_regime=False)
    assert not feats.empty


def test_trades_have_required_columns(trades):
    for col in ["timestamp", "price", "qty", "side"]:
        assert col in trades.columns


def test_bars_have_required_columns(bars):
    for col in ["open", "high", "low", "close", "volume", "signed_volume", "trade_count"]:
        assert col in bars.columns


# ── Features ──

def test_feature_count(features):
    """Should produce at least 12 core features + 4 microstructure signals."""
    assert features.shape[1] >= 16


def test_no_nan_in_features(features):
    """Features should have no NaN after dropna in compute_bar_features."""
    assert features.isna().sum().sum() == 0


def test_feature_ranges(features):
    """VPIN should be in [0, 1], vol_imbalance in [-1, 1]."""
    assert features["vpin"].min() >= -0.01  # allow small float error
    assert features["vpin"].max() <= 1.01
    assert features["vol_imbalance"].min() >= -1.01
    assert features["vol_imbalance"].max() <= 1.01


# ── Microstructure Signals ──

def test_kyles_lambda(bars):
    lam = kyles_lambda(bars)
    assert not lam.empty
    assert lam.name == "kyle_lambda"


def test_amihud_illiquidity(bars):
    amihud = amihud_illiquidity(bars)
    assert not amihud.empty
    assert (amihud.dropna() >= 0).all()  # should be non-negative


def test_roll_implied_spread(bars):
    spread = roll_implied_spread(bars)
    assert not spread.empty
    assert (spread.dropna() >= 0).all()  # spread is non-negative


def test_order_flow_imbalance(bars):
    ofi = order_flow_imbalance(bars)
    assert not ofi.empty
    assert ofi.dropna().abs().max() <= 1.01  # bounded


def test_all_microstructure_signals(bars):
    signals = compute_all_microstructure_signals(bars)
    assert set(signals.columns) == {"kyle_lambda", "amihud_illiq", "roll_spread", "ofi"}


# ── Model Training ──

def test_linear_model_walk_forward(dataset):
    result = train_linear_model(dataset, use_walk_forward=True)
    assert result.name == "lasso"
    assert "directional_accuracy" in result.metrics
    assert "information_coefficient" in result.metrics
    assert len(result.preds) > 0
    assert result.walk_forward_results is not None


def test_tree_model_single_split(dataset):
    result = train_tree_model(dataset, use_walk_forward=False)
    assert result.name == "xgboost"
    assert len(result.preds) > 0
    assert result.walk_forward_results is None


def test_forest_model(dataset):
    result = train_forest_model(dataset, use_walk_forward=False)
    assert result.name == "random_forest"
    assert result.feature_importance is not None


# ── Backtesting ──

def test_backtest_fixed_sizing(dataset):
    result = train_linear_model(dataset, use_walk_forward=False)
    config = BacktestConfig(fee_bps=1.0, slippage_bps=0.5, position_sizing="fixed")
    perf = run_directional_strategy(result.preds, result.y_test, config=config)

    assert "pnl" in perf.columns
    assert "equity" in perf.columns
    assert "costs" in perf.columns
    assert "turnover" in perf.columns


def test_backtest_summary_has_trade_analytics(dataset):
    result = train_linear_model(dataset, use_walk_forward=False)
    perf = run_directional_strategy(result.preds, result.y_test)
    stats = summary(perf)

    assert "sharpe" in stats
    assert "calmar" in stats
    assert "n_trades" in stats
    assert "profit_factor" in stats
    assert "avg_holding_bars" in stats


def test_rolling_sharpe(dataset):
    result = train_linear_model(dataset, use_walk_forward=False)
    perf = run_directional_strategy(result.preds, result.y_test)
    rs = rolling_sharpe(perf["pnl"], window=20)
    assert not rs.empty


# ── Evaluation ──

def test_information_coefficient():
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1.1, 2.2, 2.9, 4.1, 5.0])
    ic = information_coefficient(y_true, y_pred)
    assert ic > 0.9  # highly correlated


def test_calibration_table():
    y_true = pd.Series(np.random.randn(200))
    y_pred = pd.Series(np.random.randn(200))
    cal = calibration_table(y_true, y_pred, n_buckets=5)
    assert "avg_predicted" in cal.columns
    assert "avg_realized" in cal.columns
    assert len(cal) == 5


def test_full_evaluation(dataset):
    result = train_linear_model(dataset, use_walk_forward=True)
    eval_result = full_evaluation(
        result.y_test, result.preds,
        fold_results=result.walk_forward_results,
    )
    assert "ic" in eval_result
    assert "icir" in eval_result
    assert "calibration_table" in eval_result
    assert "rolling_ic" in eval_result


# ── Regime Detection ──

def test_regime_detection(bars):
    regimes, _ = detect_regimes_hmm(bars, n_regimes=3)
    assert not regimes.empty
    assert set(regimes.dropna().unique()).issubset({0, 1, 2})


def test_regime_summary(bars):
    regimes, _ = detect_regimes_hmm(bars, n_regimes=3)
    summary_df = regime_summary(bars, regimes)
    assert "mean_return" in summary_df.columns
    assert "volatility" in summary_df.columns
    assert "sharpe" in summary_df.columns
