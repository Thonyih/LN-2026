"""Development-selected augmentation alternatives; no outer-target gold is read for selection."""

import pandas as pd
from config import ART, ROOT, save, load
from data import clean
from models import model, score
from augment import generate
from evaluation import metrics, store_scores
from rules import apply, POLICIES


def main():
    selections = load(ART / "classical_selected.json")
    rows = []
    for regime in ["random", "grouped"]:
        f = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        tr = f[f.split == "train"]
        va = f[f.split == "validation"]
        te = f[f.split == "test"]
        base = next(
            r["spec"]
            for r in selections[regime]
            if r["spec"]["family"] == "wordchar_svm"
            and r["policy"] == "none"
            and r["spec"]["condition"] == "stripped"
        )
        for kind in ["resample", "dropout", "sections"]:
            spec = {**base, "augmentation": kind}
            title = "wordchar_svm_" + kind
            augmented, synth = generate(clean(tr, "stripped"), kind)
            synth.to_csv(
                ROOT / f"data/augmented/{regime}_{kind}_development.csv", index=False
            )
            est = model(spec)
            est.fit(augmented, augmented.medical_specialty)
            vs = score(est, clean(va, "stripped"))
            ts = score(est, clean(te, "stripped"))
            store_scores(ART / f"{regime}_{title}_validation.npz", va, vs, spec)
            store_scores(ART / f"{regime}_{title}_test.npz", te, ts, spec)
            for policy in POLICIES:
                pred, _ = apply(va, vs, tr, policy)
                rows.append(
                    {
                        "regime": regime,
                        "name": title,
                        "spec": spec,
                        "policy": policy,
                        "metrics": metrics(va.medical_specialty, pred),
                    }
                )
            save(ART / "augmentation_validation.json", rows)
            print(regime, kind, rows[-6]["metrics"]["macro_f1"], flush=True)


if __name__ == "__main__":
    main()
