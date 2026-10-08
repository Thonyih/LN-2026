"""Fixed protocol, source lineage, and labeled-data analysis. Never reads official inputs."""

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from config import (
    ART,
    ROOT,
    PACK,
    SEED,
    FIELDS,
    LABELS,
    save,
    sha,
    BERT_REV,
    MINILM_REV,
)
from data import load_train, clean, KW_TO_LABEL
from rules import first_keywords


def main():
    df = load_train()
    assignments = []
    for regime in ["random", "grouped"]:
        cls = StratifiedKFold if regime == "random" else StratifiedGroupKFold
        outer = cls(5, shuffle=True, random_state=SEED)
        fitpool, test = next(
            outer.split(df, df.medical_specialty, df.group)
            if regime == "grouped"
            else outer.split(df, df.medical_specialty)
        )
        sub = df.iloc[fitpool]
        inner = cls(5, shuffle=True, random_state=SEED + 1)
        tr, va = next(
            inner.split(sub, sub.medical_specialty, sub.group)
            if regime == "grouped"
            else inner.split(sub, sub.medical_specialty)
        )
        split = np.full(len(df), "test", dtype=object)
        split[fitpool[tr]] = "train"
        split[fitpool[va]] = "validation"
        f = df.assign(split=split)
        f.to_csv(ART / f"{regime}_splits.csv", index=False)
        assert f.groupby("split").medical_specialty.nunique().eq(12).all()
        if regime == "grouped":
            assert f.groupby("group").split.nunique().eq(1).all()
        assignments.append(
            {
                "regime": regime,
                "counts": f.split.value_counts().to_dict(),
                "class_counts": pd.crosstab(f.split, f.medical_specialty).to_dict(),
            }
        )
    c = clean(df, "clinical")
    tuple_groups = c.groupby(FIELDS[:3], dropna=False).medical_specialty.value_counts()
    bound = float(tuple_groups.groupby(level=[0, 1, 2]).max().sum() / len(df))
    kw = first_keywords(df).map(KW_TO_LABEL)
    mask = kw.notna()
    altered = {f: int((df[f] != clean(df, "all")[f]).sum()) for f in FIELDS}
    invalid = pd.read_csv(ART / "invalid_training_records.csv", keep_default_na=False)
    stats = {
        "raw_records": len(df) + len(invalid),
        "valid_records": len(df),
        "excluded_source_records": invalid.source_record.tolist(),
        "class_counts": df.medical_specialty.value_counts().to_dict(),
        "missing_fields": {f: int((df[f].str.strip() == "").sum()) for f in FIELDS},
        "duplicate_note_extra_records": len(df) - df.group.nunique(),
        "mixed_label_note_groups": int(
            df.groupby("group").medical_specialty.nunique().gt(1).sum()
        ),
        "recognized_first_keywords": int(mask.sum()),
        "keyword_agreement": int((kw[mask] == df.loc[mask, "medical_specialty"]).sum()),
        "clinical_tuple_deterministic_empirical_accuracy_bound": bound,
        "cleaning_altered_records": altered,
        "split_statistics": assignments,
        "train_sha256": sha(PACK / "train.csv"),
    }
    save(ART / "data_audit.json", stats)
    protocol = {
        "seed": SEED,
        "outer_folds": 5,
        "inner_folds": 5,
        "evaluated_outer_fold": 0,
        "regimes": ["random", "grouped"],
        "selection": "validation macro-F1, minimum F1, then accuracy",
        "final_submission_selection": "random validation full-system macro-F1; random is a development hypothesis about repeated-note conditions, not a conclusion from inspecting official inputs",
        "conditions": ["clinical", "stripped", "all"],
        "core_models": [
            "baseline",
            "word_lr",
            "wordchar_svm",
            "lsa",
            "minilm_lr",
            "clinicalbert_head",
            "clinicalbert_headtail",
        ],
        "C_svm": [0.3, 1.0, 3.0],
        "C_logistic": [1.0, 10.0],
        "balanced": [False, True],
        "augmentation": ["none", "resample", "dropout", "sections"],
        "augmentation_cv_folds": 3,
        "bert": {
            "epochs": 3,
            "max_length": 512,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "warmup_fraction": 0.1,
            "seed": 42,
            "revision": BERT_REV,
        },
        "minilm_revision": MINILM_REV,
        "ensemble": "optional; not part of this fixed primary protocol",
        "context": "Prior runs explored all supplied labeled cases. This fixed rerun is retrospective, not previously-unseen external confirmation. Outer results are not used to change candidate choices.",
        "outer_evaluation_fitting": "All compared models use the same train partition; validation selects configuration/checkpoint. Final winner is refitted on all valid labeled data.",
        "official_inputs": "Only predict.py may load test_no_labels.csv; final output only.",
    }
    save(ROOT / "protocol.json", protocol)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
