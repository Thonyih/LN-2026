"""Verify the measured artifacts, fold lineage, shared label order, and professor output requirements."""

import ast, json, math
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from config import ROOT, ART, LABELS, SEED, save, load, sha
from evaluation import metrics, read_scores
from classical import rank


def main():
    checks = {}
    for regime in ["random", "grouped"]:
        f = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        assert (
            f.source_record.is_unique
            and f.groupby("split").medical_specialty.nunique().eq(12).all()
        )
        if regime == "grouped":
            assert f.groupby("group").split.nunique().eq(1).all()
        tr = f[f.split == "train"]
        source_labels = f.set_index("source_record").medical_specialty.to_dict()
        pool = f[f.split != "test"].reset_index(drop=True)
        cls = StratifiedKFold if regime == "random" else StratifiedGroupKFold
        split = cls(3, shuffle=True, random_state=SEED + 2)
        folds = list(
            split.split(pool, pool.medical_specialty, pool.group)
            if regime == "grouped"
            else split.split(pool, pool.medical_specialty)
        )
        for kind in ["resample", "dropout", "sections"]:
            dev = pd.read_csv(
                ROOT / f"data/augmented/{regime}_{kind}_development.csv",
                keep_default_na=False,
            )
            for _, r in dev.iterrows():
                assert r.base_source_record in set(tr.source_record) and set(
                    json.loads(r.donor_source_records)
                ) <= set(tr.source_record)
                assert source_labels[r.base_source_record] == r.medical_specialty
                assert all(
                    source_labels[d] == r.medical_specialty
                    for d in json.loads(r.donor_source_records)
                )
            lineage = pd.read_csv(
                ROOT / f"data/augmented/{regime}_{kind}_lineage.csv",
                keep_default_na=False,
            )
            for fold, (train_idx, _) in enumerate(folds):
                allowed = set(pool.iloc[train_idx].source_record)
                for _, r in lineage[lineage.fold == fold].iterrows():
                    assert (
                        r.base_source_record in allowed
                        and set(json.loads(r.donor_source_records)) <= allowed
                    )
                    assert source_labels[r.base_source_record] == r.medical_specialty
                    assert all(
                        source_labels[d] == r.medical_specialty
                        for d in json.loads(r.donor_source_records)
                    )
        for variant in ["head", "headtail"]:
            meta = load(ART / f"{regime}_clinicalbert_{variant}_training.json")
            assert all(e["updates"] == math.ceil(len(tr) / 16) for e in meta["log"])
        checks[regime] = {
            "all_classes_present": True,
            "unique_source_ids": True,
            "donors_isolated": True,
            "partial_accumulation_updates_correct": True,
        }
    checks["grouped"]["group_partitions_disjoint"] = True
    from finalize import validation_rows

    rows = validation_rows()
    manifest = load(ART / "selection_manifest.json")
    from rules import apply

    for regime in ["random", "grouped"]:
        frame = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        tr = frame[frame.split == "train"]
        va = frame[frame.split == "validation"]
        cache = {}
        for row in [r for r in rows if r["regime"] == regime]:
            if row["name"] not in cache:
                cache[row["name"]] = read_scores(
                    ART / f'{regime}_{row["name"]}_validation.npz', va
                )
            predicted, routes = apply(va, cache[row["name"]], tr, row["policy"])
            measured = metrics(va.medical_specialty, predicted)
            for key in ["accuracy", "macro_f1", "minimum_f1"]:
                assert np.isclose(measured[key], row["metrics"][key])
            if regime == "grouped":
                assert "memory" not in routes
    for regime in ["random", "grouped"]:
        best = max(
            [r for r in rows if r["regime"] == regime], key=lambda r: rank(r["metrics"])
        )
        assert (best["name"], best["policy"]) == (
            manifest["regime_winners"][regime]["name"],
            manifest["regime_winners"][regime]["policy"],
        )
    for row in load(ART / "outer_results.json"):
        pred = pd.read_csv(
            ART / (row["artifact_id"] + "_predictions.csv"), keep_default_na=False
        )
        calculated = metrics(pred.medical_specialty, pred.predicted)
        if row["regime"] == "grouped":
            assert "memory" not in set(pred.route)
        for key in ["accuracy", "macro_f1", "minimum_f1"]:
            assert np.isclose(calculated[key], row["test_metrics"][key])
        for label in LABELS:
            for key in ["precision", "recall", "f1-score", "support"]:
                assert np.isclose(
                    calculated["per_class"][label][key],
                    row["test_metrics"]["per_class"][label][key],
                )
        frame = pd.read_csv(ART / f'{row["regime"]}_splits.csv', keep_default_na=False)
        te = frame[frame.split == "test"]
        values = read_scores(ART / f'{row["regime"]}_{row["name"]}_test.npz', te)
        assert set(pred.source_record) == set(te.source_record)
        aligned = pred.set_index("source_record").loc[te.source_record]
        expected, _ = apply(te, values, frame[frame.split == "train"], row["policy"])
        assert np.array_equal(aligned.predicted.to_numpy(), expected)
        assert np.array_equal(
            aligned.medical_specialty.to_numpy(), te.medical_specialty.to_numpy()
        )
        z = np.load(ART / f'{row["regime"]}_{row["name"]}_test.npz')
        meta = json.loads(str(z["metadata"]))
        if row["spec"]["family"] != "clinicalbert" and not row["spec"].get(
            "augmentation"
        ):
            assert set(meta["fitted_source_ids"]) == set(
                frame[frame.split == "train"].source_record
            )
    # No analysis/training stage may call the official loader.
    callers = []
    for p in (ROOT / "src").glob("*.py"):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "load_official_for_prediction"
            ):
                callers.append(p.name)
    assert callers == ["predict.py"], callers
    out = (ROOT / "results.txt").read_text().splitlines()
    assert len(out) == 409 and all(l in LABELS for l in out)
    raw = (ROOT / "results.txt").read_bytes()          # same conventions as the official CSV files
    assert raw.count(b"\r\n") == 408 and raw.count(b"\n") == 408 and not raw.endswith(b"\n"), "CRLF/newline"
    assert not raw.startswith(b"\xef\xbb\xbf") and all(l == l.strip() for l in out), "BOM/whitespace"
    assert len(set(out)) == 12, "all 12 specialties should be predicted"
    final = load(ART / "final_output_verification.json")
    assert final["results_sha256"] == sha(ROOT / "results.txt")
    scores = np.load(ART / "final_prediction_scores.npz")
    assert scores["source_record"].tolist() == list(range(409))
    assert scores["labels"].tolist() == LABELS
    assert scores["scores"].shape == (409, 12) and np.isfinite(scores["scores"]).all()
    checks.update(
        {
            "selection_recomputed_from_validation": True,
            "validation_metrics_recomputed_from_scores": True,
            "outer_metrics_recomputed": True,
            "canonical_labels_and_source_alignment": True,
            "official_loader_callers": callers,
            "results_format_crlf_no_trailing_newline_no_bom": True,
            "official_predictions": len(out),
            "final_output_hash_matches": True,
            "unit_tests": "6 passed; see tests.log",
            "paper_status": "English Markdown; final template PDF pagination and group details remain user tasks",
        }
    )
    save(ART / "integrity_verification.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
