from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_SAMPLE_ROWS


def generate_synthetic_trades(
    n: int = DEFAULT_SAMPLE_ROWS,
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc),
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.0008,
    spread_bps: float = 2.0,
    buy_prob: float = 0.5,
    base_price: float = 42_000.0,
) -> pd.DataFrame:
    """Create a pseudo realistic trade tape using a simple OU-like process."""
    rng = np.random.default_rng(seed)
    dt = 1.0  # seconds between ticks on average

    # OU process for log returns
    kappa = 0.05
    log_price = np.log(base_price)
    prices = []
    times = []
    sides = []
    qtys = []

    t = start
    for _ in range(n):
        shock = rng.normal(loc=drift * dt, scale=vol * (dt**0.5))
        log_price = log_price + -kappa * (log_price - np.log(base_price)) * dt + shock
        price = float(np.exp(log_price))

        # Add tiny microstructure noise for bid/ask bounce
        half_spread = price * spread_bps / 20000
        is_buy = rng.random() < buy_prob
        trade_price = price + (half_spread if is_buy else -half_spread)
        size = rng.lognormal(mean=-2.7, sigma=0.6)

        prices.append(trade_price)
        sides.append("buy" if is_buy else "sell")
        qtys.append(size)
        times.append(t)
        t += timedelta(seconds=max(1, int(rng.poisson(lam=1))))

    df = pd.DataFrame({
        "timestamp": times,
        "price": prices,
        "qty": qtys,
        "side": sides,
    })
    return df


def write_sample(path: Optional[Path] = None, n: int = DEFAULT_SAMPLE_ROWS) -> Path:
    target = path or DATA_DIR / "sample_trades.csv"
    df = generate_synthetic_trades(n=n)
    target.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(target, index=False)
    return target
