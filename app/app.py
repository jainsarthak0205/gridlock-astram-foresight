"""ASTraM Foresight — Streamlit demo.

    uv run streamlit run app/app.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

import inference as inf
from nlp_severity import severity_score, severity_label

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BLR_VIEW = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.4, pitch=0)
SEV_COLOR = {"none": [120, 144, 156], "low": [255, 213, 79],
             "medium": [255, 138, 101], "high": [229, 57, 53]}

st.set_page_config(page_title="ASTraM Foresight", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 3.5rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
.small {color:#9aa0a6; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def load_data():
    names = ["events_clean", "corridor_risk", "venue_recurrence", "breakdown_hotspots",
             "breakdown_by_corridor", "clearance_median_by_cause",
             "clearance_median_by_cause_corridor", "text_signals"]
    return {n: pd.read_parquet(PROC / f"{n}.parquet") for n in names}


@st.cache_resource(show_spinner=False)
def load_models():
    return {"closure": inf.load_bundle("model_closure.joblib"), "nlp": inf.load_nlp()}


@st.cache_data(show_spinner=False)
def load_reports():
    out = {}
    for f in ["impact_metrics.json", "nlp_metrics.json", "preprocess_summary.json"]:
        p = REPORTS / f
        if p.exists():
            out[f] = json.loads(p.read_text())
    return out


@st.cache_data(show_spinner=False)
def options(_ev):
    def opts(col):
        return sorted([x for x in _ev[col].dropna().unique().tolist()])
    return {
        "causes": _ev["event_cause"].value_counts().index.tolist(),
        "corridors": opts("corridor"),
        "veh_types": opts("veh_type"),
        "stations": opts("police_station"),
    }


D = load_data()
M = load_models()
R = load_reports()
EV = D["events_clean"]
OPT = options(EV)


# ------------------------------------------------------------- small helpers
def clearance_estimate(cause, corridor):
    cc = D["clearance_median_by_cause_corridor"]
    hit = cc[(cc.event_cause == cause) & (cc.corridor == corridor)]
    if len(hit):
        return float(hit.iloc[0]["median_min"]), f"{cause} on {corridor} (n={int(hit.iloc[0]['n'])})"
    bc = D["clearance_median_by_cause"]
    hit = bc[bc.event_cause == cause]
    if len(hit):
        return float(hit.iloc[0]["median_min"]), f"{cause}, all corridors (n={int(hit.iloc[0]['n'])})"
    return float(EV["duration_min"].median()), "city-wide median"


def risk_band(p, priority_high, severity):
    if p >= 0.5 or (priority_high and p >= 0.35):
        return "HIGH", "#e53935", "Pre-stage diversion & barricades · deploy 3–4 officers · alert adjacent junctions."
    if p >= 0.2 or severity >= 3:
        return "MEDIUM", "#fb8c00", "Deploy 2 officers · stage cones · monitor for escalation."
    return "LOW", "#43a047", "Routine response · 1 officer · no diversion expected."


def scatter(df, lon="longitude", lat="latitude", color=[33, 150, 243], radius=60, tooltip=None):
    layer = pdk.Layer("ScatterplotLayer", df, get_position=[lon, lat],
                      get_fill_color=color if isinstance(color, list) else color,
                      get_radius=radius, radius_min_pixels=2, radius_max_pixels=40,
                      opacity=0.6, pickable=True)
    return pdk.Deck(layers=[layer], initial_view_state=BLR_VIEW,
                    map_provider="carto", map_style="light",
                    tooltip=tooltip or {"text": "{tip}"})


def legend(items, prefix=""):
    chips = " &nbsp;&nbsp; ".join(
        f"<span style='display:inline-block;width:11px;height:11px;background:rgb({r},{g},{b});"
        f"border-radius:2px;margin-right:5px;vertical-align:middle'></span>{lbl}"
        for lbl, (r, g, b) in items)
    st.markdown(f"<div class='small'>{prefix}{chips}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------- pages
def page_overview():
    st.title("🚦 ASTraM Foresight")
    st.markdown("**Predictive intelligence for event-driven congestion — Bengaluru.** "
                "Built on Bengaluru Traffic Police ASTraM event data. "
                "<span class='small'>Gridlock Hackathon 2.0 · PS2</span>", unsafe_allow_html=True)
    s = R.get("preprocess_summary.json", {})
    c = st.columns(4)
    c[0].metric("Events analysed", f"{s.get('n_events', len(EV)):,}")
    c[1].metric("Currently active", f"{int((EV.status == 'active').sum()):,}")
    c[2].metric("Need road closure", f"{s.get('closure_rate_pct', '—')}%")
    c[3].metric("Breakdown hotspots", f"{s.get('n_breakdown_hotspots', '—'):,}")

    st.divider()
    st.subheader("Model scorecard — what's real, stated honestly")
    im, nm = R.get("impact_metrics.json", {}), R.get("nlp_metrics.json", {})
    rows = []
    if im.get("closure"):
        cl = im["closure"]
        rows.append(["Road-closure predictor", "LightGBM (binary)", f"ROC-AUC {cl['roc_auc']} · PR-AUC {cl['pr_auc']} · recall {cl['recall']}", "✅ real signal"])
    if nm:
        rows.append(["Cause classifier (NLP)", "TF-IDF + LogReg", f"acc {nm['accuracy']} vs {nm['baseline_majority_acc']} baseline · weighted-F1 {nm['weighted_f1']}", "✅ real signal"])
    sd = nm.get("severity_distribution", {})
    sd_txt = " · ".join(f"{k} {int(v):,}" for k, v in sd.items() if k != "none") if sd else "—"
    rows.append(["Congestion severity", "Lexicon (derived)", f"{sd_txt} (of {len(EV):,} events)", "➕ additive, heuristic"])
    if im.get("clearance_band"):
        rows.append(["Clearance-time", "—", f"acc {im['clearance_band']['accuracy']} vs {im['clearance_band']['baseline_majority_acc']} baseline", "❌ not learnable → use history"])
    rows.append(["Priority", "Corridor rule", f"{im.get('priority_rule', {}).get('rule_agreement_pct', '—')}% = named-corridor rule", "🚩 deterministic, not modelled"])
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Method", "Validation", "Verdict"]),
                 hide_index=True, width="stretch")
    st.caption("We predict what's predictable (closure risk, cause, congestion text) and surface "
               "reliable history for what isn't (clearance time). Transparency is the point.")


def page_predictor():
    st.header("🔮 Live Impact Predictor")
    st.caption("When an event is reported, forecast its traffic impact and the recommended response.")
    col = st.columns(2)
    with col[0]:
        cause = st.selectbox("Event cause", OPT["causes"], index=OPT["causes"].index("vehicle_breakdown") if "vehicle_breakdown" in OPT["causes"] else 0)
        corridor = st.selectbox("Corridor", OPT["corridors"], index=OPT["corridors"].index("Mysore Road") if "Mysore Road" in OPT["corridors"] else 0)
        veh = st.selectbox("Vehicle type (if any)", ["(none)"] + OPT["veh_types"])
        etype = st.radio("Event type", ["unplanned", "planned"], horizontal=True)
    with col[1]:
        station = st.selectbox("Police station", OPT["stations"])
        hour = st.slider("Hour of day (IST)", 0, 23, 9)
        dow = st.selectbox("Day of week", DOW, index=4)
        lat = st.number_input("Latitude", value=12.9716, format="%.5f")
        lon = st.number_input("Longitude", value=77.5946, format="%.5f")

    dow_i = DOW.index(dow)
    p = inf.closure_from_inputs(
        M["closure"], event_type=etype, event_cause=cause, corridor=corridor,
        veh_type=None if veh == "(none)" else veh, zone=None, police_station=station,
        latitude=lat, longitude=lon, start_hour=hour, start_dow=dow_i,
        is_weekend=dow_i in (5, 6), is_night=hour in (22, 23, 0, 1, 2, 3, 4, 5))

    priority_high = corridor != "Non-corridor"
    clr_min, clr_src = clearance_estimate(cause, corridor)
    band, color, action = risk_band(p, priority_high, 0)

    base = float(EV["requires_closure"].mean())
    st.divider()
    m = st.columns(4)
    m[0].metric("Road-closure probability", f"{p:.0%}", delta=f"{p / base:.1f}× city avg", delta_color="off")
    m[1].metric("Priority (rule)", "High" if priority_high else "Low")
    m[2].metric("Typical clearance", f"{clr_min:.0f} min")
    m[3].markdown(f"<div class='small'>Impact level</div><h2 style='color:{color};margin-top:-6px'>{band}</h2>", unsafe_allow_html=True)
    st.progress(min(p, 1.0))
    st.markdown(f"**Recommended response** &nbsp; <span style='color:{color}'>●</span> {action}", unsafe_allow_html=True)
    st.caption(f"City-average closure rate is {base:.0%}; this event is {p / base:.1f}× that. "
               f"Closure model ROC-AUC 0.81 · clearance from history ({clr_src}) · "
               "priority = corridor rule · response = transparent policy logic.")


def page_text():
    st.header("📝 Text Intelligence (NLP)")
    st.caption("Auto-classify the cause and extract a congestion-severity signal from an operator's "
               "free-text note — works on English and Kannada.")
    examples = {
        "BMTC bus breakdown": "Bmtc bus offload due to gear box problem vehicle not moving",
        "Water-logging": "water logging traffic slow moving near underpass",
        "Tree fall": "tree fall blocking the road one side",
        "Stadium event": "Cricket match at M Chinnaswamy Stadium heavy crowd",
        "Kannada (breakdown)": "ವೆಹಿಕಲ್ ಬ್ರೇಕ್ ಡೌನ್ ಆಗಿರುತ್ತದೆ",
    }
    pick = st.selectbox("Load an example", ["(type your own)"] + list(examples))
    default = examples.get(pick, "")
    text = st.text_area("Event description", value=default, height=110,
                        placeholder="e.g. lorry breakdown near junction, traffic slow…")

    if text.strip():
        top = inf.predict_cause(M["nlp"], text, k=3)
        sev = severity_score(text)
        c = st.columns([2, 1])
        with c[0]:
            st.markdown("**Predicted cause**")
            for name, prob in top:
                st.write(f"`{name}`")
                st.progress(min(prob, 1.0))
        with c[1]:
            lab = severity_label(sev)
            col = {"none": "#90a4ae", "low": "#ffd54f", "medium": "#ff8a65", "high": "#e53935"}[lab]
            st.markdown("**Congestion severity**")
            st.markdown(f"<h2 style='color:{col}'>{lab.upper()}</h2><div class='small'>score {sev}/9 · lexicon-derived</div>", unsafe_allow_html=True)
        st.caption("Cause = trained classifier (acc 0.66 vs 0.52 baseline; F1 0.82 on breakdowns). "
                   "Severity = transparent keyword signal capturing congestion cues missing from structured fields.")


def page_maps():
    st.header("🗺️ Hotspot Maps")
    view = st.radio("Layer", ["Breakdown hotspots", "All events by severity"], horizontal=True)
    if view == "Breakdown hotspots":
        h = D["breakdown_hotspots"].copy()
        h["tip"] = ("Breakdowns: " + h["n_breakdowns"].astype(str)
                    + " · " + h["corridor"].fillna("—").astype(str)
                    + " · bus share " + (h["bus_share"].fillna(0) * 100).round(0).astype(int).astype(str) + "%")
        h["radius"] = 30 + h["n_breakdowns"] * 14
        st.pydeck_chart(scatter(h.rename(columns={"glon": "longitude", "glat": "latitude"}),
                                color=[229, 57, 53], radius="radius"))
        st.caption("● Dot size = number of breakdowns at that location.")
        st.markdown("**Breakdowns by corridor** — buses & heavy vehicles are the #1 unplanned driver.")
        st.dataframe(D["breakdown_by_corridor"].head(12), hide_index=True, width="stretch")
    else:
        c1, c2 = st.columns([3, 1])
        sel = c1.multiselect("Causes", OPT["causes"],
                             default=["vehicle_breakdown", "water_logging", "accident", "tree_fall"])
        only_cue = c2.checkbox("Only events with a congestion cue", value=False)
        df = EV.merge(D["text_signals"][["id", "severity_label"]], on="id", how="left")
        df = df[df.event_cause.isin(sel)].dropna(subset=["latitude", "longitude"]).copy()
        if only_cue:
            df = df[df.severity_label != "none"]
        df["color"] = df["severity_label"].map(SEV_COLOR).apply(lambda v: v if isinstance(v, list) else [120, 144, 156])
        df["tip"] = df["event_cause"].astype(str) + " · severity " + df["severity_label"].astype(str)
        layer = pdk.Layer("ScatterplotLayer", df, get_position=["longitude", "latitude"],
                          get_fill_color="color", get_radius=70, opacity=0.55,
                          radius_min_pixels=2, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=BLR_VIEW,
                                 map_provider="carto", map_style="light", tooltip={"text": "{tip}"}))
        legend([("none", tuple(SEV_COLOR["none"])), ("low", tuple(SEV_COLOR["low"])),
                ("medium", tuple(SEV_COLOR["medium"])), ("high", tuple(SEV_COLOR["high"]))],
               prefix="congestion severity: &nbsp; ")
        st.caption(f"{len(df):,} events shown · colour = text-derived congestion severity.")


def page_corridor():
    st.header("🛣️ Corridor & Venue Risk")
    cr = D["corridor_risk"].sort_values("n_events", ascending=False)
    st.subheader("Corridor risk profile")
    st.dataframe(cr, hide_index=True, width="stretch")
    st.bar_chart(cr.set_index("corridor")["median_duration_min"].dropna().head(15), height=240)

    st.subheader("Recurring-event playbook")
    v = D["venue_recurrence"].copy()
    v["venue"] = "📍 " + v["glat"].round(4).astype(str) + ", " + v["glon"].round(4).astype(str)
    top = v.sort_values("n_events", ascending=False).head(25)
    choice = st.selectbox("Recurring location (lat, lon)", top["venue"].tolist())
    row = top[top["venue"] == choice].iloc[0]
    c = st.columns(4)
    c[0].metric("Events here", int(row["n_events"]))
    c[1].metric("Planned", int(row["n_planned"]))
    c[2].metric("Needed closure", int(row["n_closure"]))
    c[3].metric("Median duration", f"{row['med_duration_min']:.0f} min" if pd.notna(row['med_duration_min']) else "—")
    st.caption("For recurring venues, history tells you what to expect before the next event.")

    st.subheader("Typical clearance time by cause (honest history, not a forecast)")
    st.dataframe(D["clearance_median_by_cause"], hide_index=True, width="stretch")


def page_triage():
    st.header("🚦 Live Triage Board")
    st.caption("Currently-active events, ranked by a transparent impact score "
               "(closure risk + congestion severity + priority).")
    act = EV[EV.status == "active"].copy()
    if act.empty:
        st.info("No active events in the dataset.")
        return
    act["closure_p"] = inf.predict_closure(M["closure"], act)
    act = act.merge(D["text_signals"][["id", "severity_score", "severity_label"]], on="id", how="left")
    act["severity_score"] = act["severity_score"].fillna(0)
    act["severity_label"] = act["severity_label"].fillna("none")
    act["priority_high"] = act["priority_high"].fillna(False).astype(bool)
    act["impact"] = (0.5 * act["closure_p"]
                     + 0.3 * (act["severity_score"] / 9.0)
                     + 0.2 * act["priority_high"].astype(float)).round(3)
    act = act.sort_values("impact", ascending=False)

    k = st.columns(3)
    k[0].metric("Active events", f"{len(act):,}")
    k[1].metric("High closure-risk (>50%)", int((act["closure_p"] > 0.5).sum()))
    k[2].metric("With a congestion cue", int((act["severity_label"] != "none").sum()))

    top = act.head(200).copy()
    top["desc_short"] = top["description"].astype(str).str.slice(0, 70)
    show = top[["impact", "closure_p", "event_cause", "corridor", "police_station",
                "priority_high", "severity_label", "desc_short"]].copy()
    show["closure_p"] = (show["closure_p"] * 100).round(0).astype(int).astype(str) + "%"
    show.columns = ["Impact", "Closure", "Cause", "Corridor", "Station", "High-prio", "Severity", "Description"]
    st.dataframe(show, hide_index=True, width="stretch", height=380)

    mp = top.dropna(subset=["latitude", "longitude"]).copy()
    mp["tip"] = "impact " + mp["impact"].astype(str) + " · " + mp["event_cause"].astype(str)
    mp["radius"] = 60 + mp["impact"] * 300
    st.pydeck_chart(scatter(mp, color=[229, 57, 53], radius="radius"))


# ------------------------------------------------------------------- router
PAGES = {
    "Overview": page_overview,
    "🔮 Live Impact Predictor": page_predictor,
    "📝 Text Intelligence": page_text,
    "🗺️ Hotspot Maps": page_maps,
    "🛣️ Corridor & Venue Risk": page_corridor,
    "🚦 Live Triage Board": page_triage,
}

st.sidebar.title("🚦 ASTraM Foresight")
st.sidebar.caption("Event-driven congestion intelligence · Bengaluru")
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.divider()
st.sidebar.caption("Data: BTP ASTraM events (Nov 2023 – Apr 2024).")
st.sidebar.caption("Models trained offline; the app serves predictions.")
PAGES[choice]()
