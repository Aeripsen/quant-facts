import sys
from pathlib import Path

# Flat modules, no package install: put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
