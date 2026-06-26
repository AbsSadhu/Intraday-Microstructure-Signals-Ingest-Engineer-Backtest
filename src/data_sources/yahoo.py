import yfinance as yf
import pandas as pd
import numpy as np

def fetch_yahoo_trades(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch intraday data from Yahoo Finance.
    Yahoo Finance only provides 1-minute OHLCV bars for the last 7 days.
    This function pulls the bars and structures them so the pipeline can use them.
    """
    print(f"Fetching data from Yahoo Finance for {symbol}...")
    
    # Download 1-minute data
    if start and end:
        df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
        # If the requested time was outside market hours (empty df), fallback to 7 days
        if df.empty:
            print(f"Warning: No data found between {start} and {end} (market likely closed). Falling back to the last 7 days...")
            df = yf.download(symbol, period="7d", interval="1m", progress=False)
    else:
        df = yf.download(symbol, period="7d", interval="1m", progress=False)
        
    if df.empty:
        raise ValueError(f"No data found for {symbol}. If it's an Indian stock, ensure you add '.NS' (e.g., RELIANCE.NS) or '.BO'. Note: 1-minute data is only available for the last 7 days.")

    # Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "timestamp"
    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    
    # We need SignedVolume and TradeCount for our features, but yfinance doesn't provide them.
    # We will simulate them so the pipeline doesn't break.
    df["trade_count"] = df["volume"] / 100  # Rough proxy for trade count
    
    # Simulate SignedVolume based on price change
    price_change = df["close"] - df["open"]
    direction = np.sign(price_change)
    # If open == close, use previous close to determine direction
    diff_sign = np.sign(df["close"].diff()).fillna(1)
    direction = np.where(direction == 0, diff_sign, direction)
    df["signed_volume"] = df["volume"] * direction
    
    return df

def yahoo_to_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the DataFrame exactly as is, since it's already in OHLCV bar format.
    """
    return df
