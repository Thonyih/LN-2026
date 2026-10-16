"""Memorisation analysis from saved predictions (no retraining, labelled data only).

On the RANDOM outer test, some notes also occur (same cleaned-note group) in the fitting partition.
Memorisation rate of a bare model = share of those repeated test notes for which it predicts a specialty the
note already carries in the fitting partition. Because this corpus never repeats a (note, specialty) pair,
such a prediction is almost always wrong, so a high rate means the model reproduces training copies.
Writes artifacts/memorisation.json.
"""
import glob
import re

import pandas as pd

from config import ART, save

splits = pd.read_csv(ART / "random_splits.csv", keep_default_na=False)
fit = splits[splits.split == "train"]
seen = fit.groupby("group").medical_specialty.apply(set).to_dict()

out = {}
for path in sorted(glob.glob(str(ART / "random_bare_*_predictions.csv"))):
    p = pd.read_csv(path, keep_default_na=False)
    rep = p[p.group.isin(seen)]
    name = re.sub(r"^random_bare_|_predictions\.csv$", "", path.split("/")[-1])
    gold_unseen = sum(g not in seen[k] for g, k in zip(rep.medical_specialty, rep.group))
    out[name] = {
        "test_rows": len(p),
        "repeated_test_notes": len(rep),
        "gold_label_new_for_repeated_notes": f"{gold_unseen}/{len(rep)}",
        "memorisation_rate": round(sum(q in seen[k] for q, k in zip(rep.predicted, rep.group)) / max(len(rep), 1), 3),
        "accuracy_on_repeated": round(float((rep.predicted == rep.medical_specialty).mean()), 3),
        "accuracy_on_unseen": round(float((p[~p.group.isin(seen)].predicted ==
                                           p[~p.group.isin(seen)].medical_specialty).mean()), 3),
    }
    print(f"{name:40s} mem={out[name]['memorisation_rate']:.3f}  acc repeated={out[name]['accuracy_on_repeated']:.3f}"
          f"  acc unseen={out[name]['accuracy_on_unseen']:.3f}  (repeated {len(rep)}/{len(p)}, gold new {gold_unseen}/{len(rep)})")
save(ART / "memorisation.json", out)
