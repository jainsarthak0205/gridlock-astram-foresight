# 🚦 ASTraM Foresight

**A predictive intelligence layer for event-driven traffic congestion in Bengaluru.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://gridlock-astram-foresight-demo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
![License](https://img.shields.io/badge/data-anonymized-blue)

> **🔗 Live app: https://gridlock-astram-foresight-demo.streamlit.app/**
> Built for the **Gridlock Hackathon 2.0 — Prototype Round 2 (Problem Statement 2: Event-Driven Congestion).**

---

## The problem

Bengaluru loses an estimated **₹20,000 crore a year** to traffic congestion and is consistently ranked among the most congested cities in the world. A large share of that gridlock is **event-driven** — vehicle breakdowns, water-logging, tree-falls, accidents, construction, processions and public events that suddenly choke a corridor.

The Bengaluru Traffic Police already run **ASTraM** (Actionable Intelligence for Sustainable Traffic Management) to log these events. But today:

- Event impact is **not quantified in advance** — officers can't tell which reports will need a road closure.
- Resource deployment is **experience-driven**, not data-driven.
- There is **no post-event learning system** that turns thousands of historical incidents into foresight.

**ASTraM Foresight** is the missing predictive layer: the moment an event is reported, it forecasts the traffic impact and turns it into an actionable response recommendation — and it surfaces the historical patterns (hotspots, corridor risk, recurring venues) that make enforcement proactive instead of reactive.

It is built on **8,173 real (anonymized) ASTraM events** from Nov 2023 – Apr 2024.

---

## What it does

The app ([live here](https://gridlock-astram-foresight-demo.streamlit.app/)) has six sections:

| Section | What it gives an operator |
|---|---|
| **🔮 Live Impact Predictor** | Enter a reported event → **road-closure probability**, priority, an expected clearance time from history, and a recommended response (officers / barricades / diversion). |
| **📝 Text Intelligence** | Paste an operator's free-text note (English *or* Kannada) → auto-classified **cause** + a **congestion-severity** signal extracted from the text. |
| **🗺️ Hotspot Maps** | Interactive map of Bengaluru — breakdown hotspots and events coloured by congestion severity. |
| **🛣️ Corridor & Venue Risk** | Per-corridor risk profiles, a recurring-event "playbook" for known venues, and historical clearance times. |
| **🚦 Live Triage Board** | All currently-active events, ranked by a transparent impact score (closure risk + severity + priority). |
| **📊 Overview** | KPIs and an honest model scorecard. |

---

## The models — stated honestly

A core design principle: **predict what is genuinely predictable, and surface reliable history for what isn't.** Every component below was validated on a chronological (time-based) train/test split, using only information available *at the moment an event is reported* (no leakage).

| Component | Method | Result | Verdict |
|---|---|---|---|
| **Road-closure predictor** | LightGBM (binary) | **ROC-AUC 0.81**, PR-AUC 0.36 (vs 0.12 base rate), recall 0.55 | ✅ Real, useful signal |
| **Cause classifier (NLP)** | TF-IDF (word + char) + Logistic Regression | **Acc 0.66 vs 0.52 baseline**; F1 up to **0.82** (breakdowns), 0.75 (water-logging), 0.71 (tree-fall) | ✅ Real, useful signal |
| **Congestion severity** | Keyword lexicon (multilingual) | Surfaces congestion cues in ~23% of events that are absent from structured fields | ➕ Additive, transparent (not a trained model) |
| **Clearance time** | — | Not predictable from report-time data (regression and 3-band classifier both fail to beat the baseline) | ❌ Replaced with honest historical medians |
| **Priority (High/Low)** | — | 99.8% determined by "is the event on a named corridor" | 🚩 A deterministic rule, not a model |

This transparency is deliberate: the manpower/barricading recommendation is presented as **tunable policy logic** on top of the validated model outputs, never as a black box.

---

## How it works (architecture)

```
                 OFFLINE (run once, locally)                     ONLINE (Streamlit Cloud)
  ┌─────────────────────────────────────────────┐      ┌──────────────────────────────────┐
  raw ASTraM CSV ──► preprocess.py ──► clean parquet + aggregates ─┐
  (local only,                                                     ├──► app/app.py serves
   git-ignored)    train_impact.py ─► road-closure model  ─────────┤    predictions live
                   train_nlp.py    ─► NLP model + severity ────────┘    (reads only the small
                                       (committed to git)                committed artifacts)
  └─────────────────────────────────────────────┘      └──────────────────────────────────┘
```

The raw dataset is **never read at runtime**. Training happens offline; only small precomputed parquet files (a few MB) and the trained model files are committed. This keeps the deployed app fast and well within free-tier memory.

---

## Project structure

```
data/raw/          raw ASTraM CSV (git-ignored, local only)
data/processed/    small parquet artifacts the app reads (committed)
models/            trained model files (committed)
src/               offline pipeline: config, preprocess, train_impact, train_nlp,
                   nlp_severity, inference (shared by the app)
app/app.py         the Streamlit app (the demo)
tests/smoke_test.py  runs every page through Streamlit's AppTest
reports/           model metrics & data summary (JSON)
```

---

## Run it locally

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync                              # create .venv + install pinned dependencies

# (1) place the provided dataset at data/raw/astram_events.csv, then build everything:
uv run python src/preprocess.py      # raw events -> clean parquet + aggregates   (seconds)
uv run python src/train_impact.py    # road-closure model + diagnostics           (seconds)
uv run python src/train_nlp.py       # NLP cause classifier + severity signal      (seconds)

# (2) run the tests and launch the app
uv run python tests/smoke_test.py    # verify all pages render
uv run streamlit run app/app.py      # open the demo at http://localhost:8501
```

Total training time is only a few seconds — the dataset is small (≈8K rows).

---

## Tech stack

- **Python 3.13**, **uv** for dependency management
- **pandas / NumPy / PyArrow** for data
- **scikit-learn** + **LightGBM** for the models
- **Streamlit** + **pydeck** for the app and maps (Carto basemap — no API key required)

## Deployment

Hosted free on **Streamlit Community Cloud**, deployed directly from this GitHub repo (`main` branch, `app/app.py`). Every push auto-redeploys. The app reads only the committed parquet/model artifacts, so it boots in seconds and needs no secrets.

## Data & privacy

The dataset is the **anonymized** ASTraM event log provided by the hackathon organizers (IDs, vehicle numbers and personal references are masked). The raw file is kept out of version control.
