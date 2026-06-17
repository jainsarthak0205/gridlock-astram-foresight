"""Shared inference helpers for the Streamlit app (and a runnable self-test).

    uv run python src/inference.py    # smoke-test predictions
"""
import joblib
import numpy as np
import pandas as pd
import config as C
from nlp_severity import severity_score, severity_label  # re-exported for the app

CAT = ["event_type", "event_cause", "corridor", "veh_type", "zone", "police_station"]
NUM = ["latitude", "longitude", "start_hour", "start_dow", "is_weekend", "is_night"]


def load_bundle(name):
    return joblib.load(C.MODELS / name)


def load_nlp():
    return joblib.load(C.MODELS / "model_nlp_cause.joblib")


def build_X(df: pd.DataFrame, features) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in CAT:
            X[c] = df[c].astype("category")
        elif c in NUM:
            col = df[c]
            X[c] = col.astype("float64") if col.dtype == bool else pd.to_numeric(col, errors="coerce").astype("float64")
        else:  # extra flags (requires_closure / priority_high)
            X[c] = df[c].map({True: 1.0, False: 0.0}).astype("float64") if df[c].dtype == object \
                else pd.to_numeric(df[c], errors="coerce").astype("float64")
    return X[features]


def predict_closure(bundle, df: pd.DataFrame) -> np.ndarray:
    X = build_X(df, bundle["features"])
    return bundle["model"].predict_proba(X)[:, 1]


def closure_from_inputs(bundle, **kwargs) -> float:
    row = pd.DataFrame([kwargs])
    return float(predict_closure(bundle, row)[0])


def predict_cause(pipe, text, k=3):
    proba = pipe.predict_proba([str(text)])[0]
    classes = pipe.classes_
    order = np.argsort(proba)[::-1][:k]
    return [(str(classes[i]), float(proba[i])) for i in order]


if __name__ == "__main__":
    ev = pd.read_parquet(C.DATA_PROC / "events_clean.parquet")
    b = load_bundle("model_closure.joblib")

    # batch on a few real events
    sample = ev.head(5)
    print("batch closure proba:", np.round(predict_closure(b, sample), 3).tolist())

    # single constructed event
    p = closure_from_inputs(
        b, event_type="planned", event_cause="construction", corridor="ORR East 2",
        veh_type=None, zone=None, police_station="Whitefield",
        latitude=12.9695, longitude=77.7006, start_hour=18, start_dow=4,
        is_weekend=False, is_night=False,
    )
    print(f"single construction-event closure proba: {p:.3f}")

    nlp = load_nlp()
    for t in ["BMTC bus offload due to gear box problem", "tree fall blocking road",
              "water logging traffic slow moving", "Cricket match at stadium heavy crowd"]:
        top = predict_cause(nlp, t)
        s = severity_score(t)
        print(f"  '{t[:40]:<40}' -> {top[0][0]} ({top[0][1]:.2f}) | severity={severity_label(s)}({s})")
    print("OK")
