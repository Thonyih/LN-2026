"""Shared development tuning; outer scores saved without using outer gold for selection."""

import time, itertools
import pandas as pd, numpy as np
from config import ART, save, load, LABELS
from data import clean
from evaluation import metrics, store_scores
from rules import apply, POLICIES
from models import model, score


def rank(m):
    return (m["macro_f1"], m["minimum_f1"], m["accuracy"])


def name(spec):
    return "_".join(
        str(spec[k]) for k in ["family", "condition", "C", "balanced"] if k in spec
    )


def specs():
    yield {"family": "baseline", "condition": "clinical"}
    for cond in ["clinical", "stripped", "all"]:
        for family, cs in [
            ("wordchar_svm", [0.3, 1.0, 3.0]),
            ("word_lr", [1.0, 10.0]),
            ("minilm_lr", [1.0, 10.0]),
        ]:
            for c, balanced in itertools.product(cs, [False, True]):
                yield {
                    "family": family,
                    "condition": cond,
                    "C": c,
                    "balanced": balanced,
                }
    for c in [1.0, 10.0]:
        yield {"family": "lsa", "condition": "stripped", "C": c, "balanced": True}


def main():
    results = []
    selected = {}
    for regime in ["random", "grouped"]:
        frame = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        tr = frame[frame.split == "train"].copy()
        va = frame[frame.split == "validation"].copy()
        te = frame[frame.split == "test"].copy()
        pool = frame[frame.split != "test"].copy()
        cached = {}
        fitted_specs = {}
        rows = []
        for spec in specs():
            title = name(spec)
            start = time.time()
            cond = spec["condition"]
            trc, vac = clean(tr, cond), clean(va, cond)
            if spec["family"] == "minilm_lr":
                from embeddings import embed

                if cond not in cached:
                    cached[cond] = embed(clean(frame, cond), cond)
                # frame indices are stable source-array positions; no source ids assumed contiguous.
                x = cached[cond]
                xt = x[tr.index]
                xv = x[va.index]
            else:
                xt, xv = trc, vac
            est = model(spec)
            est.fit(xt, tr.medical_specialty)
            s = score(est, xv)
            store_scores(ART / f"{regime}_{title}_validation.npz", va, s, spec)
            for policy in POLICIES:
                pred, _ = apply(va, s, tr, policy)
                m = metrics(va.medical_specialty, pred)
                row = {
                    "regime": regime,
                    "name": title,
                    "spec": spec,
                    "policy": policy,
                    "metrics": m,
                    "seconds": time.time() - start,
                }
                rows.append(row)
                results.append(row)
            fitted_specs[title] = spec
            save(ART / "classical_validation.json", results)
            print(
                regime,
                title,
                "bare",
                round(rows[-6]["metrics"]["macro_f1"], 4),
                "full",
                round(rows[-1]["metrics"]["macro_f1"], 4),
                "secs",
                round(time.time() - start),
                flush=True,
            )
        # Fair base-model comparison chooses stripped inputs; baseline has no keywords by definition.
        choices = []
        for family in ["baseline", "word_lr", "wordchar_svm", "lsa", "minilm_lr"]:
            candidates = [
                r
                for r in rows
                if r["spec"]["family"] == family
                and r["policy"] == "none"
                and (r["spec"]["condition"] == "stripped" or family == "baseline")
            ]
            choices.append(max(candidates, key=lambda r: rank(r["metrics"])))
        # Full-system candidates selected on validation; keep best in each family and the clinical-only SVM.
        for family in ["baseline", "word_lr", "wordchar_svm", "lsa", "minilm_lr"]:
            choices.append(
                max(
                    [r for r in rows if r["spec"]["family"] == family],
                    key=lambda r: rank(r["metrics"]),
                )
            )
        choices.append(
            max(
                [
                    r
                    for r in rows
                    if r["spec"]["family"] == "wordchar_svm"
                    and r["spec"]["condition"] == "clinical"
                    and r["policy"] == "none"
                ],
                key=lambda r: rank(r["metrics"]),
            )
        )
        unique = {r["name"]: r["spec"] for r in choices}
        for title, spec in unique.items():
            if spec["family"] == "minilm_lr":
                x = cached[spec["condition"]]
                xp = x[tr.index]
                xe = x[te.index]
            else:
                xp, xe = clean(tr, spec["condition"]), clean(te, spec["condition"])
            est = model(spec)
            est.fit(xp, tr.medical_specialty)
            s = score(est, xe)
            store_scores(
                ART / f"{regime}_{title}_test.npz",
                te,
                s,
                {"spec": spec, "fitted_source_ids": tr.source_record.tolist()},
            )
        selected[regime] = choices
    save(ART / "classical_selected.json", selected)
    print("Classical comparisons complete.", flush=True)


if __name__ == "__main__":
    main()
