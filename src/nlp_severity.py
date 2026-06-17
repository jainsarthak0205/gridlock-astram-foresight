"""Lexicon-based congestion-severity signal from free-text event descriptions.

This is a transparent, rule-based DERIVED signal (NOT a trained model): it surfaces the
congestion cues operators type in free text but which are absent from structured fields.
Handles English plus common Romanized / transliterated spellings seen in the data.
"""
import re

HIGH = ["standstill", "gridlock", "not moving", "no movement", "blocked",
        "jam", "stuck", "choke", "snarl", "halt"]          # weight 3
MED = ["slow", "crawl", "congest", "swol", "heavy traffic",
       "pileup", "pile up", "bumper"]                       # weight 2
LOW = ["traffic", "busy", "delay", "diversion", "divert"]   # weight 1


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def severity_score(text):
    if text is None:
        return 0
    t = _norm(text)
    if not t or t == "nan":
        return 0
    score = 0
    for kw in HIGH:
        if kw in t:
            score += 3
    for kw in MED:
        if kw in t:
            score += 2
    for kw in LOW:
        if kw in t:
            score += 1
    return min(score, 9)


def severity_label(score):
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    if score >= 1:
        return "low"
    return "none"
