"""Read labeled inputs; official data has a separate final-prediction-only entry point."""

import csv, re, hashlib
import pandas as pd
from config import PACK, ART, FIELDS, LABELS

csv.field_size_limit(10**9)
KW_TO_LABEL = {
    "surgery": "Surgery",
    "cardiovascular / pulmonary": "Cardiovascular-Pulmonary",
    "orthopedic": "Orthopedic",
    "radiology": "Radiology",
    "general medicine": "General Medicine",
    "gastroenterology": "Gastroenterology",
    "neurology": "Neurology",
    "obstetrics / gynecology": "Obstetrics-Gynecology",
    "neurosurgery": "Neurosurgery",
    "ophthalmology": "Ophthalmology",
    "psychiatry / psychology": "Psychiatry-Psychology",
    "dermatology": "Dermatology",
}
EMBEDDED = re.compile(r"\s(?:" + "|".join(map(re.escape, LABELS)) + r");")


def _read(path, header):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    return rows[1:] if header else rows


def clean(frame, condition="stripped"):
    """Clinical=3 fields; stripped=remove first specialty clue; all=retain supplied clues."""
    out = frame.copy()
    for f in FIELDS:

        def trim(s):
            m = EMBEDDED.search(str(s))
            return str(s)[: m.start()] if m else str(s)

        out[f] = out[f].fillna("").map(trim)
    out["transcription"] = out.transcription.str.replace(r",(?=[A-Z])", " ", regex=True)
    if condition == "clinical":
        out["keywords"] = ""
    elif condition == "stripped":

        def strip(k):
            parts = k.split(",")
            return (
                ",".join(parts[1:]).strip()
                if parts[0].strip().lower() in KW_TO_LABEL
                else k
            )

        out["keywords"] = out.keywords.map(strip)
    elif condition != "all":
        raise ValueError(condition)
    return out


def note_keys(frame):
    c = clean(frame, "clinical")
    keys = []
    for _, r in c.iterrows():
        text = (
            r.transcription
            if r.transcription.strip()
            else "|".join(r[f] for f in FIELDS[:3])
        )
        norm = re.sub(r"\W+", "", text.lower())
        keys.append(hashlib.sha256(norm.encode()).hexdigest())
    return keys


def load_train():
    rows = _read(PACK / "train.csv", True)
    assert all(len(r) == 5 for r in rows)
    df = pd.DataFrame(rows, columns=["medical_specialty"] + FIELDS)
    df.insert(0, "source_record", range(len(df)))
    bad = df[~df.medical_specialty.isin(LABELS)]
    bad.to_csv(ART / "invalid_training_records.csv", index=False)
    df = df[df.medical_specialty.isin(LABELS)].reset_index(drop=True)
    df["group"] = note_keys(df)
    return df


def load_official_for_prediction():
    rows = _read(PACK / "test_no_labels.csv", False)
    assert all(len(r) == 4 for r in rows)
    df = pd.DataFrame(rows, columns=FIELDS)
    df.insert(0, "source_record", range(len(df)))
    df["group"] = note_keys(df)
    return df


def write_labels(path, pred):
    assert all(p in LABELS for p in pred)
    from pathlib import Path

    # Same conventions as the official CSV files: CRLF line endings, no trailing newline, no header,
    # labels written exactly as in train.csv (no added/removed whitespace).
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(pred))
