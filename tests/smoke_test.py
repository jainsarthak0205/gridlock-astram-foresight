"""Smoke test: run every app page through Streamlit's AppTest and assert no exceptions.

    uv run python tests/smoke_test.py
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")
PAGES = ["Overview", "🔮 Live Impact Predictor", "📝 Text Intelligence",
         "🗺️ Hotspot Maps", "🛣️ Corridor & Venue Risk", "🚦 Live Triage Board"]


def main():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, f"initial run failed: {at.exception}"
    radio = at.sidebar.radio[0]
    failures = []
    for p in PAGES:
        radio.set_value(p).run()
        ok = not at.exception
        print(f"{'OK ' if ok else 'ERR'}  {p}")
        if not ok:
            failures.append((p, at.exception))
    if failures:
        for p, e in failures:
            print(f"FAIL {p}: {e}")
        sys.exit(1)
    print("RESULT: ALL PAGES OK")


if __name__ == "__main__":
    main()
