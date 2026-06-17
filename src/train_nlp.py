"""NLP module:
  (a) train a cause classifier from free-text descriptions (text -> event_cause),
  (b) enrich every event with a text-derived congestion-severity signal.

    uv run python src/train_nlp.py

Outputs:
    models/model_nlp_cause.joblib        TF-IDF (word+char) + LogisticRegression
    data/processed/text_signals.parquet  per-event severity score/label + has_text
    reports/nlp_metrics.json
"""
import json
import re
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import config as C
from nlp_severity import severity_score, severity_label

TRIVIAL = {"", "a", "no", "yes", ".", "-", "na", "nil", "none", "n/a", "ok"}


def is_meaningful(s):
    if pd.isna(s):
        return False
    t = re.sub(r"[^\w\s]", "", str(s).lower()).strip()
    return len(t) >= 3 and t not in TRIVIAL


def load():
    return pd.read_parquet(C.DATA_PROC / "events_clean.parquet").sort_values(
        "start_datetime").reset_index(drop=True)


def collapse_rare(y, min_count=25):
    vc = y.value_counts()
    keep = set(vc[vc >= min_count].index)
    return y.where(y.isin(keep), other="others")


def build_pipeline():
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    feats = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def main():
    df = load()

    # (b) congestion-severity signal for every event ------------------------
    sev = df["description"].apply(severity_score)
    sig = pd.DataFrame({
        "id": df["id"],
        "severity_score": sev,
        "severity_label": sev.apply(severity_label),
        "has_text": df["description"].apply(is_meaningful),
    })
    sig.to_parquet(C.DATA_PROC / "text_signals.parquet", index=False)
    sev_dist = sig["severity_label"].value_counts().to_dict()

    # (a) cause classifier on meaningful descriptions -----------------------
    d = df[df["description"].apply(is_meaningful)].copy()
    d["label"] = collapse_rare(d["event_cause"].astype(str))
    n = int(len(d) * 0.8)
    tr, te = d.iloc[:n], d.iloc[n:]

    pipe = build_pipeline()
    pipe.fit(tr["description"].astype(str), tr["label"])
    pred = pipe.predict(te["description"].astype(str))
    maj = tr["label"].mode()[0]

    rep = classification_report(te["label"], pred, output_dict=True, zero_division=0)
    top_classes = set(te["label"].value_counts().head(8).index)
    metrics = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_classes": int(d["label"].nunique()),
        "accuracy": round(accuracy_score(te["label"], pred), 3),
        "macro_f1": round(f1_score(te["label"], pred, average="macro"), 3),
        "weighted_f1": round(f1_score(te["label"], pred, average="weighted"), 3),
        "baseline_majority_acc": round(float((te["label"] == maj).mean()), 3),
        "per_class_f1_top": {k: round(rep[k]["f1-score"], 3) for k in top_classes if k in rep},
        "severity_distribution": {str(k): int(v) for k, v in sev_dist.items()},
    }

    joblib.dump(pipe, C.MODELS / "model_nlp_cause.joblib")
    (C.REPORTS / "nlp_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nsaved model_nlp_cause.joblib, text_signals.parquet, nlp_metrics.json")


if __name__ == "__main__":
    main()
