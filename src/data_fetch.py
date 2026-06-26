import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
import requests

from .config import (
    BINANCE_AGG_TRADES_URL,
    COINBASE_TRADES_URL,
    DATA_DIR,
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_TRADE_RULE,
)
from .synthetic import write_sample


def load_local_trades(path: Optional[str] = None, min_rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    """Load sample trade data; regenerate synthetic sample if missing or too small."""
    target = DATA_DIR / "sample_trades.csv" if path is None else Path(path)
    if not target.exists():
        write_sample(target, n=min_rows)
    df = pd.read_csv(target, parse_dates=["timestamp"])
    # legacy compatibility
    if "size" in df.columns and "qty" not in df.columns:
        df = df.rename(columns={"size": "qty"})
    if len(df) < min_rows:
        target.unlink(missing_ok=True)
        write_sample(target, n=min_rows)
        df = pd.read_csv(target, parse_dates=["timestamp"])
    df["side"] = df["side"].str.lower()
    return df


def _binance_params(symbol: str, start_ms: int, end_ms: int, limit: int) -> dict:
    return {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit}


def fetch_binance_trades(
    symbol: str,
    start: datetime,
    end: datetime,
    limit: int = 1000,
    pause: float = 0.25,
) -> pd.DataFrame:
    """Fetch aggregated trades from Binance public REST (no API key required).

    Notes
    -----
    - Binance caps `limit` at 1000; we step through time using the last trade id.
    - Returns a DataFrame with columns [timestamp, price, qty, side].
    """

    start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000)
    params = _binance_params(symbol, start_ms, end_ms, limit)

    trades = []
    last_id: Optional[int] = None
    while True:
        if last_id is not None:
            params["fromId"] = last_id + 1
        resp = requests.get(BINANCE_AGG_TRADES_URL, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for row in batch:
            # Row keys: a=aggId, p=price, q=qty, f=firstId, l=lastId, T=ms, m=buyerMaker
            trades.append(
                {
                    "timestamp": datetime.fromtimestamp(row["T"] / 1000, tz=timezone.utc),
                    "price": float(row["p"]),
                    "qty": float(row["q"]),
                    "side": "sell" if row["m"] else "buy",
                }
            )
        last_id = batch[-1]["a"]
        # Stop if we reached end of window
        last_ts = batch[-1]["T"]
        if last_ts >= end_ms:
            break
        time.sleep(pause)

    df = pd.DataFrame(trades)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_trades(trades: pd.DataFrame, rule: str = DEFAULT_TRADE_RULE) -> pd.DataFrame:
    """Aggregate trade-level data to bars while preserving imbalance metrics."""
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["signed_qty"] = df["qty"] * df["side"].map({"buy": 1, "sell": -1})
    df = df.set_index("timestamp").sort_index()

    grouped = df.resample(rule)
    ohlcv = grouped["price"].ohlc()
    ohlcv["volume"] = grouped["qty"].sum()
    ohlcv["signed_volume"] = grouped["signed_qty"].sum()
    ohlcv["trade_count"] = grouped["price"].count()
    ohlcv = ohlcv.dropna(subset=["open", "close"])  # drop empty buckets
    return ohlcv


def _to_coinbase_product(symbol: str) -> str:
    """Map symbols like BTCUSDT/BTCUSD/BTC-USD to Coinbase product_id."""
    if "-" in symbol:
        return symbol.upper()
    sym = symbol.upper()
    if sym.endswith("USDT"):
        return f"{sym[:-4]}-USD"
    if sym.endswith("USD"):
        return f"{sym[:-3]}-USD"
    return f"{sym}-USD"


def fetch_coinbase_trades(
    product_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    max_rows: int = 5000,
    page_size: int = 100,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Fetch recent trades from Coinbase public REST (Exchange) without auth.

    Notes
    -----
    - Coinbase returns most recent first; we paginate backwards using `before` cursor.
    - Filtering by time is handled client-side using the returned timestamps.
    """
    url = COINBASE_TRADES_URL.format(product_id=product_id)
    rows = []
    before = None
    stop = False
    while len(rows) < max_rows and not stop:
        params = {"limit": page_size}
        if before is not None:
            params["before"] = before
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for trade in batch:
            ts = datetime.fromisoformat(trade["time"].replace("Z", "+00:00"))
            if end and ts > end:
                continue
            if start and ts < start:
                stop = True
                break
            rows.append(
                {
                    "timestamp": ts,
                    "price": float(trade["price"]),
                    "qty": float(trade["size"]),
                    "side": trade.get("side", "buy").lower(),
                }
            )
        before = batch[-1]["trade_id"]
        if stop:
            break
        time.sleep(pause)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_tick_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """Enrich trade-level data with returns and signed volume for VPIN."""
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["price"].pct_change().fillna(0.0)
    df["signed_qty"] = df["qty"] * df["side"].map({"buy": 1, "sell": -1})
    return df


def load_or_fetch(
    source: Literal["sample", "binance", "synthetic", "coinbase"] = "sample",
    symbol: str = "BTCUSDT",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    rule: str = DEFAULT_TRADE_RULE,
    synthetic_rows: int = DEFAULT_SAMPLE_ROWS,
    max_rows: int = 10_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Utility to load bundled sample, generate synthetic, or fetch live trades."""
    if source == "binance":
        if start is None or end is None:
            raise ValueError("start and end datetimes are required for live fetch")
        trades = fetch_binance_trades(symbol=symbol, start=start, end=end)
    elif source == "coinbase":
        product = _to_coinbase_product(symbol)
        trades = fetch_coinbase_trades(product_id=product, start=start, end=end, max_rows=max_rows)
    elif source == "synthetic":
        target = DATA_DIR / "sample_trades.csv"
        write_sample(target, n=synthetic_rows)
        trades = load_local_trades(target, min_rows=synthetic_rows)
    else:
        trades = load_local_trades(min_rows=synthetic_rows)

    if trades.empty:
        raise ValueError("No trades loaded; check data source")

    bars = resample_trades(trades, rule=rule)
    return trades, bars
