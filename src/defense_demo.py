"""Fit the frozen selected SVM live and reproduce its internal test results, offline."""

import time

import numpy as np
import pandas as pd

from config import ART, LABELS, PACK, ROOT, load, sha
from data import clean, load_train
from evaluation import metrics
from models import model, score
from rules import apply


def main():
    started = time.perf_counter()
    winner = load(ART / "selection_manifest.json")["winner"]
    spec = winner["spec"]
    if spec["family"] != "wordchar_svm" or spec.get("augmentation"):
        raise ValueError("This short demo expects the frozen, unaugmented SVM winner.")
    assert sha(PACK / "train.csv") == load(ART / "data_audit.json")["train_sha256"]
    splits = pd.read_csv(
        ART / f"{winner['regime']}_splits.csv",
        usecols=["source_record", "split"],
    )
    frame = load_train().merge(splits, on="source_record", validate="one_to_one")
    assert len(frame) == len(splits)
    fitting = frame[frame.split == "train"]
    target = frame[frame.split == "test"]
    assert set(fitting.source_record).isdisjoint(target.source_record)

    print("LIVE REPRODUCTION: frozen internal test, no official inputs", flush=True)
    print(f"Model: {winner['name']}; rule policy: {winner['policy']}", flush=True)
    print(
        f"Fit on {len(fitting)} labeled cases; evaluate on {len(target)} heldout cases.",
        flush=True,
    )
    estimator = model(spec)
    estimator.fit(clean(fitting, spec["condition"]), fitting.medical_specialty)
    values = score(estimator, clean(target, spec["condition"]))
    predicted, routes = apply(target, values, fitting, winner["policy"])
    measured = metrics(target.medical_specialty, predicted)
    expected = next(
        r["test_metrics"]
        for r in load(ART / "outer_results.json")
        if r["stage"] == "selected" and r["regime"] == winner["regime"]
    )
    for key in ["accuracy", "macro_f1", "minimum_f1"]:
        assert np.isclose(measured[key], expected[key]), (
            key,
            measured[key],
            expected[key],
        )
    for label in LABELS:
        assert np.isclose(
            measured["per_class"][label]["f1-score"],
            expected["per_class"][label]["f1-score"],
        )
    archive = ART / f"{winner['regime']}_selected_predictions.csv"
    if archive.exists():
        recorded = (
            pd.read_csv(archive).set_index("source_record").loc[target.source_record]
        )
        assert np.array_equal(predicted, recorded.predicted.to_numpy())

    print(f"\nAccuracy: {measured['accuracy']:.2%}")
    print(f"Macro-F1: {measured['macro_f1']:.4f}")
    print(f"Minimum class F1: {measured['minimum_f1']:.4f}")
    print("\nPer-class F1:")
    for label in LABELS:
        row = measured["per_class"][label]
        print(f"  {label:26s} {row['f1-score']:.4f} (n={int(row['support'])})")
    output = target.assign(
        predicted=predicted,
        route=routes,
        correct=predicted == target.medical_specialty.to_numpy(),
    )
    print("\nDecision routes (internal test):")
    for route, group in output.groupby("route"):
        print(
            f"  {route:8s} {len(group):3d} cases; accuracy {group.correct.mean():.2%}"
        )
    print("\nPaper error examples:")
    for note in load(ROOT / "paper/error_notes.json"):
        if note["regime"] != winner["regime"]:
            continue
        row = output[output.source_record == note["source_record"]].iloc[0]
        assert not row.correct
        print(f"  Source {note['source_record']}: {note['excerpt']}")
        print(
            f"    Predicted: {row.predicted}; gold: {row.medical_specialty}; route: {row.route}"
        )
    print("\nPASS: live results match the archived selected-system results.")
    print(f"Fit/evaluation/demo time: {time.perf_counter() - started:.1f} seconds.")


if __name__ == "__main__":
    main()
