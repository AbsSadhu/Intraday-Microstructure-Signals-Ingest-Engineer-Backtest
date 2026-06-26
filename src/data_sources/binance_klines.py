"""
Binance Klines (OHLCV bars) bulk downloader.

Uses the public /api/v3/klines endpoint — no API key needed.
Supports any date range, any interval, unlimited history back to 2017.

Usage:
    python -m src.data_sources.binance_klines --symbol BTCUSDT --interval 1m \
        --start 2024-01-01 --end 2024-02-01 --out data/btcusdt_jan2024.parquet
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINES_LIMIT = 1000  # max per request
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


def _to_ms(dt: datetime) -> int:
    """Convert datetime to Binance millisecond timestamp."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetch_klines_chunk(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch up to 1000 klines from Binance REST."""
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": KLINES_LIMIT,
    }
    resp = requests.get(KLINES_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=KLINE_COLS)
    df = df[["open_time", "open", "high", "low", "close", "volume", "n_trades",
             "taker_buy_base_vol"]].copy()
    for col in ["open", "high", "low", "close", "volume", "taker_buy_base_vol"]:
        df[col] = df[col].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df.index.name = "timestamp"
    return df


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
    max_bars: int | None = None,
    sleep_ms: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download any number of 1-min (or any interval) klines from Binance.

    Parameters
    ----------
    symbol     : e.g. 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
    interval   : '1m', '5m', '15m', '1h', '1d' etc.
    start      : start datetime (UTC). Defaults to 30 days ago.
    end        : end datetime (UTC). Defaults to now.
    max_bars   : cap on total bars (None = unlimited)
    sleep_ms   : delay between requests (be polite to free API)
    verbose    : print progress

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume, n_trades,
                             taker_buy_base_vol, signed_volume, trade_count
    """
    now = datetime.now(tz=timezone.utc)
    if start is None:
        from datetime import timedelta
        start = now - timedelta(days=30)
    if end is None:
        end = now

    start_ms = _to_ms(start)
    end_ms = _to_ms(end)

    # Map interval string to milliseconds
    _interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
        "4h": 14_400_000, "1d": 86_400_000,
    }
    bar_ms = _interval_ms.get(interval, 60_000)

    chunks = []
    total = 0
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + KLINES_LIMIT * bar_ms, end_ms)
        chunk = fetch_klines_chunk(symbol, interval, cursor, chunk_end)
        if chunk.empty:
            break
        chunks.append(chunk)
        total += len(chunk)
        last_ts = chunk.index[-1]
        cursor = int(last_ts.timestamp() * 1000) + bar_ms

        if verbose:
            print(f"  Fetched {total:,} bars up to {last_ts.strftime('%Y-%m-%d %H:%M')} UTC", end="\r")

        if max_bars and total >= max_bars:
            break

        if cursor < end_ms:
            time.sleep(sleep_ms / 1000)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Add derived columns to match the pipeline's bar format
    df["signed_volume"] = (
        2 * df["taker_buy_base_vol"] - df["volume"]
    )  # buy_vol - sell_vol approximation
    df["trade_count"] = df["n_trades"]

    if verbose:
        print(f"\n  [OK] Downloaded {len(df):,} {interval} bars for {symbol}")
        print(f"       Range: {df.index[0]} -> {df.index[-1]}")
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"       Memory: {mem_mb:.1f} MB")

    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save bars to compressed Parquet (much faster than CSV for large data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", engine="pyarrow")
    size_mb = path.stat().st_size / 1e6
    print(f"  [OK] Saved {len(df):,} rows to {path} ({size_mb:.1f} MB)")
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load bars from Parquet, return DataFrame with DatetimeTZDtype index."""
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeTZAware if hasattr(pd, 'DatetimeTZAware') else type(df.index)):
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def klines_to_trades_format(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Convert kline bars into a synthetic trades DataFrame matching the
    pipeline's expected format: [timestamp, price, qty, side].

    Each bar is expanded into 2 synthetic trades: one buy and one sell.
    This lets existing code (compute_bar_features, etc.) work unchanged.
    """
    rows = []
    for ts, row in bars.iterrows():
        buy_vol = max(row["taker_buy_base_vol"], 0)
        sell_vol = max(row["volume"] - buy_vol, 0)
        rows.append({"timestamp": ts, "price": row["close"], "qty": buy_vol, "side": 1})
        rows.append({"timestamp": ts, "price": row["close"], "qty": sell_vol, "side": -1})
    return pd.DataFrame(rows)


# ── CLI for bulk downloading ──

def main():
    parser = argparse.ArgumentParser(
        description="Download Binance OHLCV klines to Parquet"
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m",
                        choices=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"])
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-02-01",
                        help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file (.parquet or .csv). Auto-named if omitted.")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"Downloading {args.symbol} {args.interval} bars: {args.start} -> {args.end}")
    bars = fetch_klines(
        symbol=args.symbol,
        interval=args.interval,
        start=start_dt,
        end=end_dt,
        verbose=True,
    )

    if bars.empty:
        print("No data returned.")
        return

    if args.out is None:
        out_name = (
            f"data/{args.symbol.lower()}_{args.interval}_"
            f"{args.start[:10]}_{args.end[:10]}.parquet"
        )
    else:
        out_name = args.out

    if out_name.endswith(".csv"):
        bars.to_csv(out_name)
        print(f"  [OK] Saved to {out_name}")
    else:
        save_parquet(bars, out_name)


if __name__ == "__main__":
    main()
