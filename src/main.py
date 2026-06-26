"""
End-to-end CLI entrypoint for the Microstructure Research Platform.

Supports: sample/synthetic/binance/coinbase/yahoo data sources.
Runs full pipeline: data → features → signals → regime → models → backtest → evaluation → charts.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from joblib import dump
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for chart generation

from .backtest import BacktestConfig, run_directional_strategy, summary, rolling_sharpe, rolling_drawdown
from .config import DEFAULT_HORIZON, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SPLIT, OUTPUT_DIR
from .data_fetch import load_or_fetch
from .evaluation import full_evaluation
from .features import compute_bar_features, prepare_supervised
from .labeling import forward_returns
from .models import predict_next, train_linear_model, train_tree_model, train_forest_model, train_lstm_model
from .regime import detect_regimes_hmm, regime_summary
from .visualization import generate_full_report, PLOT_DIR


def _parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)


def _load_yahoo(symbol, start, end, rule):
    """Load data from Yahoo Finance."""
    from .data_sources.yahoo import fetch_yahoo_trades, yahoo_to_bars
    trades = fetch_yahoo_trades(symbol=symbol, start=start, end=end)
    bars = yahoo_to_bars(trades)
    return trades, bars


def _load_klines(symbol, start, end, interval="1m", parquet_file=None):
    """Load data via Binance klines API (full OHLCV, any date range)."""
    from .data_sources.binance_klines import (
        fetch_klines, klines_to_trades_format, load_parquet, save_parquet
    )
    from pathlib import Path
    from .config import CACHE_DIR

    # Use cached parquet if it exists
    if parquet_file and Path(parquet_file).exists():
        print(f"  Loading from cached file: {parquet_file}")
        bars = load_parquet(parquet_file)
    else:
        bars = fetch_klines(symbol=symbol, interval=interval,
                            start=start, end=end, verbose=True)
        if parquet_file:
            save_parquet(bars, parquet_file)
        else:
            # Auto-cache to avoid re-downloading
            cache_path = CACHE_DIR / f"klines_{symbol}_{interval}.parquet"
            save_parquet(bars, cache_path)

    trades = klines_to_trades_format(bars)
    return trades, bars


def run_pipeline(
    source: str = "sample",
    symbol: str = "BTCUSDT",
    start: datetime | None = None,
    end: datetime | None = None,
    horizon: int = DEFAULT_HORIZON,
    test_split: float = DEFAULT_TEST_SPLIT,
    seed: int = DEFAULT_RANDOM_STATE,
    rule: str = "1min",
    synthetic_rows: int = 10_000,
    max_rows: int = 10_000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    position_sizing: str = "fixed",
    use_walk_forward: bool = True,
    **kwargs,
):
    """Run the full research pipeline.

    Returns
    -------
    report : dict with all metrics and evaluation results
    best_name : name of the best-performing model
    best_res : ModelResult of the best model
    extras : dict with intermediate data for dashboard/plotting
    """
    print(f"{'='*60}")
    print(f"  Microstructure Research Platform")
    print(f"  Source: {source} | Symbol: {symbol} | Horizon: {horizon}m")
    print(f"{'='*60}")

    # ── 1. Data ingestion ──
    print("\n[1/7] Loading data...")
    if source == "yahoo":
        trades, bars = _load_yahoo(symbol, start, end, rule)
    elif source == "klines":
        trades, bars = _load_klines(symbol, start, end,
                                    interval=rule,
                                    parquet_file=kwargs.get("klines_file"))
    else:
        trades, bars = load_or_fetch(
            source=source, symbol=symbol, start=start, end=end,
            rule=rule, synthetic_rows=synthetic_rows, max_rows=max_rows,
        )
    print(f"  [OK] {len(bars):,} bars loaded ({source})")

    # ── 2. Feature engineering ──
    print("[2/7] Computing features (core + microstructure signals)...")
    features = compute_bar_features(bars, include_advanced=True, include_regime=True)
    print(f"  [OK] {features.shape[1]} features x {len(features)} bars")

    # ── 3. Regime detection ──
    print("[3/7] Detecting market regimes...")
    try:
        regimes, hmm_model = detect_regimes_hmm(bars, n_regimes=3)
        reg_summary = regime_summary(bars, regimes)
        print(f"  [OK] {len(reg_summary)} regimes detected")
        print(reg_summary[["frequency", "ann_return", "ann_vol", "sharpe"]].to_string(
            float_format="{:.4f}".format))
    except Exception as e:
        print(f"  [WARN] Regime detection skipped: {e}")
        regimes = None
        reg_summary = None

    # ── 4. Target labeling & supervised dataset ──
    print("[4/7] Labeling forward returns...")
    target = forward_returns(bars, horizon=horizon)
    dataset = prepare_supervised(features, target, horizon=horizon)
    print(f"  [OK] {len(dataset)} labeled samples")

    # ── 5. Model training ──
    wf_label = "walk-forward" if use_walk_forward else "single-split"
    print(f"[5/7] Training models ({wf_label})...")

    linear_res = train_linear_model(dataset, test_size=test_split,
                                     random_state=seed, use_walk_forward=use_walk_forward)
    print(f"  [OK] Lasso  -- DA: {linear_res.metrics['directional_accuracy']:.1%}"
          f" | IC: {linear_res.metrics['information_coefficient']:.4f}")

    tree_res = train_tree_model(dataset, test_size=test_split,
                                 random_state=seed, use_walk_forward=use_walk_forward)
    print(f"  [OK] XGBoost -- DA: {tree_res.metrics['directional_accuracy']:.1%}"
          f" | IC: {tree_res.metrics['information_coefficient']:.4f}")

    forest_res = train_forest_model(dataset, test_size=test_split,
                                      random_state=seed, use_walk_forward=use_walk_forward)
    print(f"  [OK] Random Forest -- DA: {forest_res.metrics['directional_accuracy']:.1%}"
          f" | IC: {forest_res.metrics['information_coefficient']:.4f}")

    lstm_res = train_lstm_model(dataset, test_size=test_split,
                                random_state=seed, use_walk_forward=use_walk_forward)
    print(f"  [OK] LSTM Network -- DA: {lstm_res.metrics['directional_accuracy']:.1%}"
          f" | IC: {lstm_res.metrics['information_coefficient']:.4f}")

    # ── 6. Backtesting ──
    print("[6/7] Running backtest...")
    bt_config = BacktestConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_sizing=position_sizing,
    )

    perf_linear = run_directional_strategy(linear_res.preds, linear_res.y_test, config=bt_config)
    perf_tree = run_directional_strategy(tree_res.preds, tree_res.y_test, config=bt_config)
    perf_forest = run_directional_strategy(forest_res.preds, forest_res.y_test, config=bt_config)
    perf_lstm = run_directional_strategy(lstm_res.preds, lstm_res.y_test, config=bt_config)

    stats_linear = summary(perf_linear)
    stats_tree = summary(perf_tree)
    stats_forest = summary(perf_forest)
    stats_lstm = summary(perf_lstm)

    # ── 7. Evaluation ──
    print("[7/7] Running evaluation suite...")
    candidates = [
        ("lasso", linear_res, stats_linear, perf_linear),
        ("xgboost", tree_res, stats_tree, perf_tree),
        ("random_forest", forest_res, stats_forest, perf_forest),
        ("lstm", lstm_res, stats_lstm, perf_lstm),
    ]
    best_name, best_res, best_stats, best_perf = max(
        candidates, key=lambda x: x[2].get("sharpe", float("-inf"))
    )

    # Full evaluation for the best model
    eval_report = full_evaluation(
        best_res.y_test, best_res.preds,
        fold_results=best_res.walk_forward_results,
    )

    latest_signal = predict_next(best_res.model, features)

    # ── Build report ──
    report = {
        "lasso": {"metrics": linear_res.metrics, "perf": stats_linear},
        "xgboost": {"metrics": tree_res.metrics, "perf": stats_tree},
        "random_forest": {"metrics": forest_res.metrics, "perf": stats_forest},
        "lstm": {"metrics": lstm_res.metrics, "perf": stats_lstm},
        "latest_signal": latest_signal,
        "best_model": best_name,
        "evaluation": {
            "ic": eval_report["ic"],
            "icir": eval_report["icir"],
            "turnover_adjusted_alpha": eval_report["turnover_adjusted_alpha"],
            "calibration_monotonicity": eval_report["calibration_monotonicity"],
        },
    }

    if eval_report.get("walk_forward_summary"):
        report["walk_forward"] = eval_report["walk_forward_summary"]

    # Extras for visualization / dashboard
    extras = {
        "features": features,
        "bars": bars,
        "trades": trades,
        "best_perf": best_perf,
        "best_importance": best_res.feature_importance,
        "regimes": regimes,
        "regime_summary": reg_summary,
        "eval_report": eval_report,
        "all_results": {name: res for name, res, _, _ in candidates},
        "all_perfs": {name: perf for name, _, _, perf in candidates},
    }

    return report, best_name, best_res, extras


def _print_report(report: dict, best_name: str):
    """Pretty-print the research report."""
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    for model_name in ["lasso", "xgboost", "random_forest", "lstm"]:
        if model_name not in report:
            continue
        payload = report[model_name]
        is_best = " ** BEST **" if model_name == best_name else ""
        print(f"\n-- {model_name.upper()}{is_best} --")
        print(f"  Metrics:")
        for k, v in payload["metrics"].items():
            print(f"    {k:>25s}: {v:.6f}")
        print(f"  Backtest Performance:")
        for k, v in payload["perf"].items():
            if isinstance(v, float):
                print(f"    {k:>25s}: {v:.6f}")
            else:
                print(f"    {k:>25s}: {v}")

    print(f"\n-- EVALUATION ({best_name.upper()}) --")
    if "evaluation" in report:
        for k, v in report["evaluation"].items():
            print(f"  {k:>30s}: {v:.6f}" if isinstance(v, float) else f"  {k:>30s}: {v}")

    if "walk_forward" in report:
        print(f"\n-- WALK-FORWARD SUMMARY --")
        for k, v in report["walk_forward"].items():
            print(f"  {k:>35s}: {v:.6f}" if isinstance(v, float) else f"  {k:>35s}: {v}")

    print(f"\n  Latest predicted {report.get('horizon', 5)}m return: {report['latest_signal']:.6f}")
    signal = report["latest_signal"]
    direction = "[LONG]" if signal > 0 else "[SHORT]" if signal < 0 else "[FLAT]"
    print(f"  Signal direction: {direction}")


def main():
    parser = argparse.ArgumentParser(
        description="Microstructure Research Platform — Intraday Signal Analysis & Backtesting"
    )
    parser.add_argument("--source", type=str, default="sample",
                        choices=["sample", "binance", "synthetic", "coinbase", "yahoo", "klines"],
                        help="Data source (klines = Binance OHLCV bars, supports large date ranges)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--start", type=str, help="ISO start datetime (UTC)")
    parser.add_argument("--end", type=str, help="ISO end datetime (UTC)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--test_split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--rule", type=str, default="1min", help="Resample rule (e.g., 30s, 5min)")
    parser.add_argument("--rows", type=int, default=10_000, help="Synthetic rows to generate")
    parser.add_argument("--max_rows", type=int, default=10_000, help="Max trades from HTTP sources")
    parser.add_argument("--interval", type=str, default="1m",
                        choices=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Bar interval for klines source")
    parser.add_argument("--klines_file", type=str, default=None,
                        help="Path to pre-downloaded parquet file (klines source)")

    # Backtest config
    parser.add_argument("--fee_bps", type=float, default=1.0, help="Commission in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.5, help="Slippage in basis points")
    parser.add_argument("--position_sizing", type=str, default="fixed",
                        choices=["fixed", "kelly", "volatility"],
                        help="Position sizing method")

    # Outputs
    parser.add_argument("--save", action="store_true", help="Save report JSON")
    parser.add_argument("--save_model", action="store_true", help="Save best model")
    parser.add_argument("--plots", action="store_true", help="Generate all research charts")
    parser.add_argument("--no_walk_forward", action="store_true", help="Use single split instead of walk-forward")

    args = parser.parse_args()

    start_dt = _parse_dt(args.start) if args.start else datetime.now(tz=timezone.utc) - timedelta(hours=1)
    end_dt = _parse_dt(args.end) if args.end else datetime.now(tz=timezone.utc)

    report, best_name, best_res, extras = run_pipeline(
        source=args.source,
        symbol=args.symbol,
        start=start_dt,
        end=end_dt,
        horizon=args.horizon,
        test_split=args.test_split,
        seed=args.seed,
        rule=args.interval if args.source == "klines" else args.rule,
        synthetic_rows=args.rows,
        max_rows=args.max_rows,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        klines_file=getattr(args, 'klines_file', None),
        position_sizing=args.position_sizing,
        use_walk_forward=not args.no_walk_forward,
    )

    _print_report(report, best_name)

    # ── Generate plots ──
    if args.plots:
        print(f"\nGenerating research charts to {PLOT_DIR}...")
        best_perf = extras["best_perf"]
        eval_report = extras["eval_report"]

        rs = rolling_sharpe(best_perf["pnl"])

        saved = generate_full_report(
            perf=best_perf,
            features=extras["features"],
            y_true=best_res.y_test,
            y_pred=best_res.preds,
            model_name=best_name,
            report=report,
            importance=extras.get("best_importance"),
            rolling_sharpe_series=rs,
            cum_ic=eval_report.get("cumulative_ic"),
            cal_table=eval_report.get("calibration_table"),
            bars=extras.get("bars"),
            regimes=extras.get("regimes"),
        )
        print(f"  [OK] {len(saved)} charts saved to {PLOT_DIR}")

    # ── Save artifacts ──
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        target = OUTPUT_DIR / "report.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  [OK] Report saved to {target}")

    if args.save_model:
        OUTPUT_DIR.mkdir(exist_ok=True)
        model_path = OUTPUT_DIR / "best_model.pkl"
        dump(best_res.model, model_path)
        print(f"  [OK] Best model ({best_name}) saved to {model_path}")


if __name__ == "__main__":
    main()
