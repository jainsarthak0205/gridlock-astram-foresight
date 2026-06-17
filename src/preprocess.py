"""Offline preprocessing: raw ASTraM CSV -> clean parquet + small aggregates.

Run once (or whenever features change):
    python src/preprocess.py

Outputs (all tiny, committed to git):
    data/processed/events_clean.parquet     one row per event, engineered features + targets
    data/processed/corridor_risk.parquet     per-corridor risk profile
    data/processed/venue_recurrence.parquet  recurring event locations (>=2 events)
    reports/preprocess_summary.json
"""
import json
import numpy as np
import pandas as pd
import config as C

IST = pd.Timedelta(hours=5, minutes=30)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV, low_memory=False).replace("NULL", pd.NA)


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False}).astype("boolean")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]

    # --- timestamps ---
    start = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    closed = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
    out["start_datetime"] = start
    start_ist = start + IST
    out["start_hour"] = start_ist.dt.hour
    out["start_dow"] = start_ist.dt.dayofweek          # 0 = Monday
    out["is_weekend"] = out["start_dow"].isin([5, 6])
    out["is_night"] = out["start_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])

    # --- clearance-time target (cleaned: 0 < d < 24h) ---
    dur = (closed - start).dt.total_seconds() / 60.0
    out["duration_min"] = dur.where((dur > 0) & (dur < 1440))

    # --- creation-time categoricals ---
    out["event_type"] = df["event_type"].astype("category")
    out["is_planned"] = df["event_type"].eq("planned")
    out["event_cause"] = (
        df["event_cause"].astype(str).str.strip().str.lower().replace("nan", pd.NA).astype("category")
    )
    out["corridor"] = df["corridor"].astype("category")
    out["veh_type"] = df["veh_type"].astype("category")
    out["zone"] = df["zone"].astype("category")
    out["police_station"] = df["police_station"].astype("category")

    # --- spatial ---
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # --- targets + creation-time closure flag ---
    out["requires_closure"] = _to_bool(df["requires_road_closure"])
    out["priority_high"] = df["priority"].map({"High": True, "Low": False}).astype("boolean")

    # --- ops / NLP ---
    out["status"] = df["status"].astype("category")
    out["description"] = df["description"]
    return out


def make_aggregates(out: pd.DataFrame):
    rows = []
    for cor, d in out.groupby("corridor", observed=True):
        dd = d["duration_min"].dropna()
        rows.append({
            "corridor": cor,
            "n_events": int(len(d)),
            "closure_rate": round(float(d["requires_closure"].mean()) * 100, 1),
            "high_prio_rate": round(float(d["priority_high"].mean()) * 100, 1),
            "median_duration_min": round(float(dd.median()), 1) if len(dd) else np.nan,
        })
    corridor_risk = pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)

    g = out.assign(glat=out["latitude"].round(4), glon=out["longitude"].round(4))
    ven = (
        g.groupby(["glat", "glon"]).agg(
            n_events=("id", "size"),
            n_planned=("is_planned", "sum"),
            n_closure=("requires_closure", "sum"),
            med_duration_min=("duration_min", "median"),
        ).reset_index().sort_values("n_events", ascending=False)
    )
    venue = ven[ven["n_events"] >= 2].reset_index(drop=True)
    return corridor_risk, venue


def breakdown_hotspots(out: pd.DataFrame):
    bd = out[out["event_cause"] == "vehicle_breakdown"].copy()
    bd = bd.assign(glat=bd["latitude"].round(3), glon=bd["longitude"].round(3))
    hot = (
        bd.groupby(["glat", "glon"]).agg(
            n_breakdowns=("id", "size"),
            corridor=("corridor", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            bus_share=("veh_type", lambda s: round(float(s.astype(str).str.contains("bus").mean()), 2)),
        ).reset_index().sort_values("n_breakdowns", ascending=False)
    )
    hot = hot[hot["n_breakdowns"] >= 2].reset_index(drop=True)
    corr = (
        bd.groupby("corridor", observed=True).size()
        .reset_index(name="n_breakdowns").sort_values("n_breakdowns", ascending=False)
        .reset_index(drop=True)
    )
    return hot, corr


def clearance_medians(out: pd.DataFrame):
    """Honest descriptive replacement for the (un-learnable) clearance-time regressor."""
    d = out.dropna(subset=["duration_min"])
    by_cause = d.groupby("event_cause", observed=True)["duration_min"].agg(
        median_min="median", n="count").reset_index().sort_values("n", ascending=False)
    cc = d.groupby(["event_cause", "corridor"], observed=True)["duration_min"].agg(
        median_min="median", n="count").reset_index()
    cc = cc[cc["n"] >= 5].sort_values("n", ascending=False).reset_index(drop=True)
    by_cause["median_min"] = by_cause["median_min"].round(1)
    cc["median_min"] = cc["median_min"].round(1)
    return by_cause, cc


def main():
    raw = load_raw()
    out = build_features(raw)
    out.to_parquet(C.DATA_PROC / "events_clean.parquet", index=False)

    corridor_risk, venue = make_aggregates(out)
    corridor_risk.to_parquet(C.DATA_PROC / "corridor_risk.parquet", index=False)
    venue.to_parquet(C.DATA_PROC / "venue_recurrence.parquet", index=False)

    bd_hot, bd_corr = breakdown_hotspots(out)
    bd_hot.to_parquet(C.DATA_PROC / "breakdown_hotspots.parquet", index=False)
    bd_corr.to_parquet(C.DATA_PROC / "breakdown_by_corridor.parquet", index=False)

    clr_cause, clr_cause_corr = clearance_medians(out)
    clr_cause.to_parquet(C.DATA_PROC / "clearance_median_by_cause.parquet", index=False)
    clr_cause_corr.to_parquet(C.DATA_PROC / "clearance_median_by_cause_corridor.parquet", index=False)

    summary = {
        "n_events": int(len(out)),
        "n_with_duration": int(out["duration_min"].notna().sum()),
        "closure_rate_pct": round(float(out["requires_closure"].mean()) * 100, 1),
        "high_priority_pct": round(float(out["priority_high"].mean(skipna=True)) * 100, 1),
        "n_corridors": int(len(corridor_risk)),
        "n_recurring_venues": int(len(venue)),
        "n_breakdown_hotspots": int(len(bd_hot)),
        "date_min": str(out["start_datetime"].min()),
        "date_max": str(out["start_datetime"].max()),
    }
    (C.REPORTS / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nwrote events_clean / corridor_risk / venue_recurrence parquet")


if __name__ == "__main__":
    main()
