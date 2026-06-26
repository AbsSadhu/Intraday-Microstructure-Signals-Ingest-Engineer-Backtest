"""
merge_and_retrain.py
--------------------
Merges downloaded parquet files and retrains on the full 2023-2026 dataset.

Usage:
    python merge_and_retrain.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

# ── 1. Merge all parquet files ────────────────────────────────────────────────

parquet_files = sorted(DATA_DIR.glob("btcusdt_1m_*.parquet"))
# Skip the merged file itself if it already exists
parquet_files = [p for p in parquet_files if "ALL" not in p.name]

if not parquet_files:
    print("[ERROR] No parquet files found in data/")
    sys.exit(1)

print(f"\nFound {len(parquet_files)} file(s) to merge:")
frames = []
for f in parquet_files:
    df = pd.read_parquet(f, engine="pyarrow")
    print(f"  {f.name:45s}  {len(df):>8,} bars")
    frames.append(df)

merged = pd.concat(frames)
merged = merged[~merged.index.duplicated(keep="first")]
merged = merged.sort_index()

merged_path = DATA_DIR / "btcusdt_1m_ALL.parquet"
merged.to_parquet(merged_path, compression="snappy", engine="pyarrow")

total_bars = len(merged)
size_mb    = merged_path.stat().st_size / 1e6
print(f"\n[OK] Merged: {total_bars:,} bars  ({merged.index[0].date()} -> {merged.index[-1].date()})")
print(f"     Saved to {merged_path} ({size_mb:.1f} MB)")

# ── 2. Run pipeline via CLI entrypoint (cleanest approach) ────────────────────

print("\n" + "="*60)
print("  Launching research pipeline on merged dataset")
print(f"  {total_bars:,} 1-min bars | Jan 2023 -> May 2026")
print("="*60 + "\n")

import subprocess, sys

cmd = [
    sys.executable, "-m", "src.main",
    "--source", "klines",
    "--symbol", "BTCUSDT",
    "--start", str(merged.index[0].date()),
    "--end",   str(merged.index[-1].date()),
    "--interval", "1m",
    "--klines_file", str(merged_path),
    "--plots",
    "--save",
    "--save_model",
    "--horizon", "5",
    "--fee_bps", "1.0",
    "--slippage_bps", "0.5",
    "--position_sizing", "fixed",
]

result = subprocess.run(cmd, cwd=str(ROOT))
sys.exit(result.returncode)
