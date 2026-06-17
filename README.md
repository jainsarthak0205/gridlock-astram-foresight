# ASTraM Foresight

A predictive intelligence layer for **event-driven traffic congestion** in Bengaluru —
built on the Bengaluru Traffic Police **ASTraM** event dataset (Gridlock Hackathon 2.0, PS2).

When an incident/event is reported, ASTraM Foresight instantly forecasts its traffic impact
and turns it into an actionable response recommendation:

- **Road-closure prediction** — will this event need a closure?
- **Priority prediction** — High vs Low.
- **Clearance-time estimate** — how long until the road is cleared?
- **Cause & severity from free-text** (NLP) — *planned*
- **Corridor & venue risk profiles + recurring-event playbook**
- **Breakdown hotspot map** (bus / heavy-vehicle breakdowns)
- **Diversion suggester + live triage board** — *planned*

## Project layout

```
data/raw/          raw ASTraM CSV (git-ignored, local only)
data/processed/    small parquet artifacts the app reads (committed)
models/            trained model files (committed)
src/               offline pipeline (preprocess + training)
app/               Streamlit app (the demo)
reports/           metrics & summaries
```

## Run it

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync                              # create .venv + install deps

# 1. drop the provided CSV at data/raw/astram_events.csv, then:
uv run python src/preprocess.py      # raw -> processed parquet + aggregates  (seconds)
uv run python src/train_impact.py    # road-closure model + diagnostics       (seconds)
uv run python src/train_nlp.py       # NLP cause classifier + severity signal (seconds)

# 2. launch the demo
uv run streamlit run app/app.py
```

> `requirements.txt` is kept in sync for Streamlit Community Cloud deployment.

## Deployment

Hosted free on **Streamlit Community Cloud**, connected directly to this GitHub repo —
every push redeploys. The app reads only the small committed artifacts, never the raw CSV.
