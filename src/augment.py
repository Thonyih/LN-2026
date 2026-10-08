"""Fold-local augmentation with complete source and donor provenance."""

import re, json
import numpy as np, pandas as pd
from config import FIELDS

MINORITY = ["Dermatology", "Psychiatry-Psychology", "Ophthalmology", "Neurosurgery"]
SEC = re.compile(r"(?:(?<=^)|(?<=\s))([A-Z][A-Z /&()]{2,40}):")


def sections(text):
    out = []
    last = 0
    head = ""
    for m in SEC.finditer(text):
        out.append((head, text[last : m.start()].strip()))
        head = m.group(1).strip()
        last = m.end()
    out.append((head, text[last:].strip()))
    return [(h, b) for h, b in out if b or h]


def generate(training, kind, seed=42):
    if kind == "none":
        return training.copy(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    records = []
    for label in MINORITY:
        sub = training[training.medical_specialty == label]
        k = min(4, max(1, int(np.ceil(150 / max(len(sub), 1))) - 1))
        pool = {}
        if kind == "sections":
            for _, r in sub.iterrows():
                for h, b in sections(r.transcription):
                    if h and b:
                        pool.setdefault(h, []).append((int(r.source_record), b))
        for _, r in sub.iterrows():
            for _ in range(k):
                variant = r.to_dict()
                donors = set()
                base = int(r.source_record)
                if kind == "dropout":
                    # Same sampling counts as resampling. Clinical assertions are retained in remaining fields.
                    variant["keywords"] = ""
                    if rng.random() < 0.5:
                        variant["sample_name"] = ""
                    if variant["description"].strip() and rng.random() < 0.35:
                        variant["transcription"] = ""
                elif kind == "sections":
                    parts = []
                    for h, b in sections(r.transcription):
                        alts = [(i, t) for i, t in pool.get(h, []) if i != base]
                        if h and alts and rng.random() < 0.5:
                            donor, b = alts[int(rng.integers(len(alts)))]
                            donors.add(donor)
                        parts.append(f"{h}: {b}" if h else b)
                    variant["transcription"] = " ".join(parts)
                elif kind != "resample":
                    raise ValueError(kind)
                variant["base_source_record"] = base
                variant["donor_source_records"] = json.dumps(sorted(donors))
                variant["augmentation"] = kind
                # Synthetic id never impersonates a real source row. Source-note grouping remains the base group.
                variant["source_record"] = -(len(records) + 1)
                records.append(variant)
    synthetic = pd.DataFrame(records)
    allowed = set(training.source_record)
    for _, r in synthetic.iterrows():
        assert (
            r.base_source_record in allowed
            and set(json.loads(r.donor_source_records)) <= allowed
        )
    return pd.concat([training, synthetic], ignore_index=True), synthetic
