"""
Pytest configuration and shared fixtures for the Microstructure Research Platform.

Adds the project root to sys.path so `from src.xxx import ...` works cleanly
whether tests are run from the project root or from the tests/ directory.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
