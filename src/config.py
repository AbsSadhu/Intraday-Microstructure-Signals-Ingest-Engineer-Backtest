from pathlib import Path
import os

# Base project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Default settings
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"
DEFAULT_HORIZON = 5  # minutes ahead for labeling
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_ROWS = 10_000
DEFAULT_TRADE_RULE = "1min"

# Binance public API endpoint (no key needed for aggTrades)
BINANCE_AGG_TRADES_URL = "https://api.binance.com/api/v3/aggTrades"
# Coinbase public trades endpoint (no auth required)
COINBASE_TRADES_URL = "https://api.exchange.coinbase.com/products/{product_id}/trades"

# Optional API keys (from environment variables)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
