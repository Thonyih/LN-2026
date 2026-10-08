"""Matched augmentation controls on outer training data; all donor pools are fold-local."""

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from config import ART, ROOT, SEED, save, LABELS
from data import clean
from augment import generate, MINORITY
from models import model
from classical import rank
from evaluation import metrics


def main():
    selected = __import__("json").loads((ART / "classical_selected.json").read_text())
    results = []
    for regime in ["random", "grouped"]:
        frame = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        pool = frame[frame.split != "test"].reset_index(drop=True)
        # The feature condition is kept fixed at stripped; hyperparameters were chosen on development.
        spec = next(
            x["spec"]
            for x in selected[regime]
            if x["spec"]["family"] == "wordchar_svm"
            and x["policy"] == "none"
            and x["spec"]["condition"] == "stripped"
        )
        cls = StratifiedKFold if regime == "random" else StratifiedGroupKFold
        splitter = cls(3, shuffle=True, random_state=SEED + 2)
        folds = list(
            splitter.split(pool, pool.medical_specialty, pool.group)
            if regime == "grouped"
            else splitter.split(pool, pool.medical_specialty)
        )
        for kind in ["none", "resample", "dropout", "sections"]:
            predictions = np.empty(len(pool), dtype=object)
            lineage = []
            for fold, (tr, va) in enumerate(folds):
                training = clean(pool.iloc[tr], "stripped")
                target = clean(pool.iloc[va], "stripped")
                augmented, synth = generate(training, kind, seed=42 + fold)
                if len(synth):
                    lineage.append(synth.assign(fold=fold))
                assert set(training.source_record).isdisjoint(set(target.source_record))
                if regime == "grouped":
                    assert set(training.group).isdisjoint(set(target.group))
                est = model(spec)
                est.fit(augmented, augmented.medical_specialty)
                predictions[va] = est.predict(target)
            m = metrics(pool.medical_specialty, predictions)
            m["minority_mean_f1"] = float(
                np.mean([m["per_class"][l]["f1-score"] for l in MINORITY])
            )
            result = {
                "regime": regime,
                "augmentation": kind,
                "spec": spec,
                "metrics": m,
                "evaluation": "3-fold pooled out-of-fold diagnostics on outer training cases; parameters selected previously on development, not an independent estimate",
            }
            results.append(result)
            save(ART / "augmentation_results.json", results)
            pool[["source_record", "group", "medical_specialty"]].assign(
                predicted=predictions
            ).to_csv(ART / f"{regime}_augmentation_{kind}_oof.csv", index=False)
            if lineage:
                pd.concat(lineage, ignore_index=True).to_csv(
                    ROOT / f"data/augmented/{regime}_{kind}_lineage.csv", index=False
                )
            print(
                regime,
                kind,
                m["accuracy"],
                m["macro_f1"],
                m["minority_mean_f1"],
                flush=True,
            )
    save(ART / "augmentation_results.json", results)


if __name__ == "__main__":
    main()
