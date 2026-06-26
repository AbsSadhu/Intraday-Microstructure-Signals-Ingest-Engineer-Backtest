import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import websockets

from .config import DATA_DIR

BINANCE_STREAM = "wss://stream.binance.com:9443/ws/{symbol}@aggTrade"


async def stream_agg_trades(
    symbol: str,
    duration_minutes: int = 5,
    max_trades: Optional[int] = 20_000,
    out_path: Path | None = None,
) -> Path:
    """Stream Binance aggTrades over WebSocket and persist to CSV."""
    sym = symbol.lower()
    uri = BINANCE_STREAM.format(symbol=sym)
    end_time = datetime.now(tz=timezone.utc) + timedelta(minutes=duration_minutes)
    out = out_path or DATA_DIR / f"live_{symbol}.csv"
    out.parent.mkdir(exist_ok=True, parents=True)

    rows = []
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
                    "price": float(data["p"]),
                    "qty": float(data["q"]),
                    "side": "sell" if data["m"] else "buy",
                }
            )
            if max_trades and len(rows) >= max_trades:
                break
            if datetime.now(tz=timezone.utc) >= end_time:
                break

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stream live aggTrades from Binance")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument("--max_trades", type=int, default=20000)
    parser.add_argument("--out", type=str, help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    asyncio.run(stream_agg_trades(args.symbol, args.minutes, args.max_trades, out_path))
    print("Done streaming.")


if __name__ == "__main__":
    main()
