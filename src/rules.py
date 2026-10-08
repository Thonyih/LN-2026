"""Explicit corpus-specific metadata and duplicate-label hypotheses, fitted without evaluation gold."""

import numpy as np, pandas as pd
from config import LABELS
from data import KW_TO_LABEL


def first_keywords(df):
    return df.keywords.str.split(",").str[0].str.strip().str.lower()


def learn_keywords(df):
    tab = pd.crosstab(first_keywords(df), df.medical_specialty)
    return {
        k: r.idxmax()
        for k, r in tab.iterrows()
        if k and r.sum() >= 2 and r.max() / r.sum() >= 0.95
    }


def apply(frame, scores, train, policy="none"):
    s = np.array(scores, dtype=float, copy=True)
    routes = np.full(len(frame), "model", dtype=object)
    if policy in ["memory", "both", "learned_both"]:
        seen = {}
        for _, r in train.iterrows():
            if r.transcription.strip():
                seen.setdefault(r.group, set()).add(r.medical_specialty)
        for i, (_, r) in enumerate(frame.iterrows()):
            labels = seen.get(r.group, set()) if r.transcription.strip() else set()
            if labels and len(labels) < 12:
                for l in labels:
                    s[i, LABELS.index(l)] = -np.inf
                routes[i] = "memory"
    pred = np.array(LABELS, dtype=object)[s.argmax(1)]
    if policy in ["keyword", "learned_keyword", "both", "learned_both"]:
        mapping = learn_keywords(train) if policy.startswith("learned") else KW_TO_LABEL
        for i, k in enumerate(first_keywords(frame)):
            if k in mapping:
                pred[i] = mapping[k]
                routes[i] = "keyword"
    return pred, routes


POLICIES = ["none", "keyword", "learned_keyword", "memory", "both", "learned_both"]
