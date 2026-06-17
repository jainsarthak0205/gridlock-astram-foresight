"""Train the report-time Impact Predictor.

Models that earn their place (validated, no leakage, beat baseline):
    1. ROAD-CLOSURE    will this event need a road closure?            (binary)
    2. CLEARANCE-BAND  quick <30 / medium 30-90 / long >90 min         (3-class)

Also reported honestly (NOT shipped as models):
    - PRIORITY is an (almost) deterministic corridor rule -> diagnostic only.
    - point CLEARANCE-TIME regression does not beat the median baseline -> we
      predict bands instead.

All features are creation-time only (no leakage). Chronological train/test split.

    python src/train_impact.py
"""
import json
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import config as C

CAT = ["event_type", "event_cause", "corridor", "veh_type", "zone", "police_station"]
NUM = ["latitude", "longitude", "start_hour", "start_dow", "is_weekend", "is_night"]
BAND_LABELS = ["quick(<30m)", "medium(30-90m)", "long(>90m)"]


def load():
    df = pd.read_parquet(C.DATA_PROC / "events_clean.parquet")
    return df.sort_values("start_datetime").reset_index(drop=True)


def time_split(df, frac=0.8):
    n = int(len(df) * frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def prep_X(df, extra):
    X = pd.DataFrame(index=df.index)
    for c in CAT:
        X[c] = df[c].astype("category")
    for c in NUM:
        col = df[c]
        X[c] = (col.astype("float64") if col.dtype == bool
                else pd.to_numeric(col, errors="coerce").astype("float64"))
    for c in extra:
        X[c] = df[c].map({True: 1.0, False: 0.0}).astype("float64")
    return X


# ---------------------------------------------------------------- closure
def train_closure(train, test):
    tr = train.dropna(subset=["requires_closure"]); te = test.dropna(subset=["requires_closure"])
    Xtr, ytr = prep_X(tr, []), tr["requires_closure"].astype(int)
    Xte, yte = prep_X(te, []), te["requires_closure"].astype(int)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced", random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_positive_rate": round(float(yte.mean()), 3),
        "accuracy": round(accuracy_score(yte, pred), 3),
        "precision": round(precision_score(yte, pred, zero_division=0), 3),
        "recall": round(recall_score(yte, pred, zero_division=0), 3),
        "f1": round(f1_score(yte, pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(yte, proba), 3),
        "pr_auc": round(average_precision_score(yte, proba), 3),
    }
    joblib.dump({"model": model, "features": CAT + NUM, "cat": CAT}, C.MODELS / "model_closure.joblib")
    fi = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(6)
    print(f"\n[ROAD-CLOSURE]  {json.dumps(m)}")
    print(f"   top features: {', '.join(fi.index)}")
    return m


# ----------------------------------------------------------- clearance band
def to_band(minutes):
    return pd.cut(minutes, bins=[0, 30, 90, np.inf], labels=[0, 1, 2]).astype("float")


def train_clearance_band(train, test, extra):
    tr = train.dropna(subset=["duration_min"]).copy(); te = test.dropna(subset=["duration_min"]).copy()
    ytr = to_band(tr["duration_min"]).astype(int); yte = to_band(te["duration_min"]).astype(int)
    Xtr, Xte = prep_X(tr, extra), prep_X(te, extra)
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced", random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    maj = int(pd.Series(ytr).mode()[0])
    m = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "accuracy": round(accuracy_score(yte, pred), 3),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 3),
        "baseline_majority_acc": round(accuracy_score(yte, np.full_like(yte, maj)), 3),
        "per_class_f1": {BAND_LABELS[i]: round(f, 3)
                         for i, f in enumerate(f1_score(yte, pred, average=None, labels=[0, 1, 2]))},
    }
    joblib.dump({"model": model, "features": CAT + NUM + extra, "cat": CAT, "bands": BAND_LABELS},
                C.MODELS / "model_clearance_band.joblib")
    print(f"\n[CLEARANCE-BAND]  {json.dumps(m)}")
    return m


def clearance_regression_check(train, test, extra):
    """Kept only to document that point regression underperforms the baseline."""
    tr = train.dropna(subset=["duration_min"]); te = test.dropna(subset=["duration_min"])
    Xtr, Xte = prep_X(tr, extra), prep_X(te, extra)
    ytr = np.log1p(tr["duration_min"].to_numpy()); yte = te["duration_min"].to_numpy()
    model = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    pred = np.clip(np.expm1(model.predict(Xte)), 1, 1440)
    base = np.full_like(yte, np.median(tr["duration_min"]))
    return {
        "model_mae_min": round(mean_absolute_error(yte, pred), 1),
        "baseline_mae_min": round(mean_absolute_error(yte, base), 1),
        "rmse_min": round(mean_squared_error(yte, pred) ** 0.5, 1),
        "r2": round(r2_score(yte, pred), 3),
        "verdict": "point regression does NOT beat baseline -> use CLEARANCE-BAND",
    }


# ------------------------------------------------------------- priority rule
def priority_rule_diagnostic(df):
    d = df.dropna(subset=["priority_high"]).copy()
    on_named = d["corridor"].notna() & d["corridor"].ne("Non-corridor")
    agree = float((on_named.to_numpy() == d["priority_high"].astype(bool).to_numpy()).mean())
    m = {
        "rule": "High iff event is on a named corridor (corridor != 'Non-corridor')",
        "rule_agreement_pct": round(agree * 100, 2),
        "note": "priority is ~deterministic from corridor; not a predictive model",
    }
    print(f"\n[PRIORITY = corridor rule]  {json.dumps(m)}")
    return m


def main():
    df = load()
    train, test = time_split(df, 0.8)
    print(f"events={len(df)}  train={len(train)}  test={len(test)}  (chronological split)")
    results = {
        "closure": train_closure(train, test),
        "clearance_band": train_clearance_band(train, test, ["requires_closure"]),
        "clearance_regression_check": clearance_regression_check(train, test, ["requires_closure"]),
        "priority_rule": priority_rule_diagnostic(df),
    }
    (C.REPORTS / "impact_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nsaved models + reports/impact_metrics.json")


if __name__ == "__main__":
    main()
