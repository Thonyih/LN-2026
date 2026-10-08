"""Require equal fitting data for all outer comparisons; regenerate only mismatched score artifacts."""

import json, numpy as np, pandas as pd
from config import ART, load
from data import clean
from models import model, score
from evaluation import store_scores


def main():
    selected = load(ART / "classical_selected.json")
    for regime, choices in selected.items():
        frame = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        tr = frame[frame.split == "train"]
        te = frame[frame.split == "test"]
        for name, spec in {r["name"]: r["spec"] for r in choices}.items():
            path = ART / f"{regime}_{name}_test.npz"
            meta = json.loads(str(np.load(path)["metadata"]))
            if set(meta.get("fitted_source_ids", [])) == set(tr.source_record):
                continue
            if spec["family"] == "minilm_lr":
                from embeddings import embed

                x = embed(clean(frame, spec["condition"]), spec["condition"])
                xt = x[tr.index]
                xe = x[te.index]
            else:
                xt = clean(tr, spec["condition"])
                xe = clean(te, spec["condition"])
            est = model(spec)
            est.fit(xt, tr.medical_specialty)
            store_scores(
                path,
                te,
                score(est, xe),
                {"spec": spec, "fitted_source_ids": tr.source_record.tolist()},
            )
            print("Aligned fitting partition", regime, name, flush=True)


if __name__ == "__main__":
    main()
