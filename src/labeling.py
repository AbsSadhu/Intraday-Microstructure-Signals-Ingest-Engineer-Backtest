import pandas as pd


def forward_returns(bars: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Compute forward percentage returns over the given horizon."""
    close = bars["close"]
    fwd = close.shift(-horizon) / close - 1.0
    return fwd
