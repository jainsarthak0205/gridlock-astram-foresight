"""Central paths & constants for ASTraM Foresight."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

RAW_CSV = DATA_RAW / "astram_events.csv"

# Bengaluru is UTC+5:30; source timestamps are tagged +00 (treated as UTC).
IST_OFFSET = ("5h", "30min")

for _d in (DATA_PROC, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)
