"""Freeze development-selected choices, then evaluate fixed systems and group-bootstrap uncertainty."""

import json, numpy as np, pandas as pd
from config import ART, ROOT, LABELS, save, load
from classical import rank
from evaluation import metrics, read_scores, plot_confusion
from rules import apply, POLICIES


def validation_rows():
    rows = load(ART / "classical_validation.json") + load(
        ART / "augmentation_validation.json"
    )
    for regime in ["random", "grouped"]:
        for variant in ["head", "headtail"]:
            rows += load(ART / f"{regime}_clinicalbert_{variant}_validation.json")
    return rows


def bootstrap(frame, pred, other=None, repeats=500):
    from sklearn.metrics import accuracy_score, f1_score

    groups = list(frame.groupby("group").indices.values())
    rng = np.random.default_rng(123)
    y = frame.medical_specialty.to_numpy()
    values = []
    for _ in range(repeats):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        a = np.array(
            [
                accuracy_score(y[idx], pred[idx]),
                f1_score(
                    y[idx], pred[idx], labels=LABELS, average="macro", zero_division=0
                ),
            ]
        )
        if other is not None:
            a -= np.array(
                [
                    accuracy_score(y[idx], other[idx]),
                    f1_score(
                        y[idx],
                        other[idx],
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    ),
                ]
            )
        values.append(a)
    ci = np.percentile(values, [2.5, 97.5], axis=0)
    return {
        "accuracy_95_interval": ci[:, 0].tolist(),
        "macro_f1_95_interval": ci[:, 1].tolist(),
        "replicates": repeats,
        "unit": "normalized note group",
        "interpretation": "Conditional descriptive uncertainty; does not account for prior exploration or model-selection variability.",
    }


def main():
    rows = validation_rows()
    choices = {}
    core = {}
    full = {}
    metadata_choices = {}
    rule_choices = {}
    families = [
        "baseline",
        "word_lr",
        "wordchar_svm",
        "lsa",
        "minilm_lr",
        "clinicalbert",
    ]
    for regime in ["random", "grouped"]:
        candidates = [r for r in rows if r["regime"] == regime]
        choices[regime] = max(candidates, key=lambda r: rank(r["metrics"]))
        core[regime] = []
        full[regime] = []
        metadata_choices[regime] = []
        rule_choices[regime] = []
        for family in families:
            base = [
                r
                for r in candidates
                if r["spec"]["family"] == family
                and r["policy"] == "none"
                and not r["spec"].get("augmentation")
                and (r["spec"]["condition"] == "stripped" or family == "baseline")
            ]
            core[regime].append(max(base, key=lambda r: rank(r["metrics"])))
            full[regime].append(
                max(
                    [r for r in candidates if r["spec"]["family"] == family],
                    key=lambda r: rank(r["metrics"]),
                )
            )
        # Prespecified input conditions, selected using development gold only.
        for condition in ["clinical", "stripped", "all"]:
            metadata_choices[regime].append(
                max(
                    [
                        r
                        for r in candidates
                        if r["spec"]["family"] == "wordchar_svm"
                        and r["spec"]["condition"] == condition
                        and r["policy"] == "none"
                        and not r["spec"].get("augmentation")
                    ],
                    key=lambda r: rank(r["metrics"]),
                )
            )
        clinical = metadata_choices[regime][0]
        for policy in ["keyword", "memory", "both"]:
            metadata_choices[regime].append(
                next(
                    r
                    for r in candidates
                    if r["name"] == clinical["name"] and r["policy"] == policy
                )
            )
        for chosen in full[regime]:
            for policy in ["none", "keyword", "memory", "both"]:
                rule_choices[regime].append(
                    next(
                        r
                        for r in candidates
                        if r["name"] == chosen["name"] and r["policy"] == policy
                    )
                )
    # This manifest is written before computing or reading outer metric reports.
    save(
        ART / "selection_manifest.json",
        {
            "primary_submission_regime": "random",
            "winner": choices["random"],
            "regime_winners": choices,
            "core_choices": core,
            "full_choices": full,
            "metadata_choices": metadata_choices,
            "rule_choices": rule_choices,
            "selection_scope": "validation only; test gold absent from ranking inputs",
        },
    )
    evaluated = []
    selected_predictions = {}
    comparison = {}
    for regime in ["random", "grouped"]:
        f = pd.read_csv(ART / f"{regime}_splits.csv", keep_default_na=False)
        tr = f[f.split == "train"]
        te = f[f.split == "test"].reset_index(drop=True)
        scored = {}
        to_evaluate = {}
        for tag, selected in [("bare", core[regime]), ("full", full[regime])]:
            for row in selected:
                to_evaluate[(tag, row["name"], row["policy"])] = row
        winning = choices[regime]
        to_evaluate[("selected", winning["name"], winning["policy"])] = winning
        for row in metadata_choices[regime]:
            to_evaluate[("metadata", row["name"], row["policy"])] = row
        for row in rule_choices[regime]:
            to_evaluate[("rules", row["name"], row["policy"])] = row
        for (tag, title, policy), row in to_evaluate.items():
            if title not in scored:
                score_path = ART / f"{regime}_{title}_test.npz"
                if not score_path.exists():
                    from models import model, score
                    from data import clean
                    from evaluation import store_scores

                    est = model(row["spec"])
                    est.fit(clean(tr, row["spec"]["condition"]), tr.medical_specialty)
                    store_scores(
                        score_path,
                        te,
                        score(est, clean(te, row["spec"]["condition"])),
                        {
                            "spec": row["spec"],
                            "fitted_source_ids": tr.source_record.tolist(),
                        },
                    )
                scored[title] = read_scores(ART / f"{regime}_{title}_test.npz", te)
            pred, route = apply(te, scored[title], tr, policy)
            m = metrics(te.medical_specialty, pred)
            identifier = f"{regime}_{tag}_{title}_{policy}"
            output = te.assign(
                predicted=pred,
                route=route,
                correct=pred == te.medical_specialty.to_numpy(),
            )
            stored = (
                output[
                    [
                        "source_record",
                        "group",
                        "medical_specialty",
                        "predicted",
                        "route",
                        "correct",
                    ]
                ]
                if tag == "rules"
                else output
            )
            stored.to_csv(ART / f"{identifier}_predictions.csv", index=False)
            save(ART / f"{identifier}_report.json", m)
            summary = {
                **row,
                "stage": tag,
                "test_metrics": m,
                "artifact_id": identifier,
            }
            evaluated.append(summary)
            if tag == "selected":
                output.to_csv(ART / f"{regime}_selected_predictions.csv", index=False)
                output[~output.correct].to_csv(
                    ART / f"{regime}_errors.csv", index=False
                )
                output.groupby("route").agg(
                    records=("correct", "size"), accuracy=("correct", "mean")
                ).to_csv(ART / f"{regime}_route_performance.csv")
                plot_confusion(te, pred, ART / f"{regime}_confusion.png")
                selected_predictions[regime] = pred
                save(ART / f"{regime}_selected_bootstrap.json", bootstrap(te, pred))
        # Two predeclared comparisons: BERT versus SVM, stripped bare models, on each regime.
        br = next(r for r in core[regime] if r["spec"]["family"] == "clinicalbert")
        sr = next(r for r in core[regime] if r["spec"]["family"] == "wordchar_svm")
        bp, _ = apply(te, scored[br["name"]], tr, "none")
        sp, _ = apply(te, scored[sr["name"]], tr, "none")
        comparison[regime] = bootstrap(te, bp, sp)
        comparison[regime]["point_difference"] = {
            "accuracy": metrics(te.medical_specialty, bp)["accuracy"]
            - metrics(te.medical_specialty, sp)["accuracy"],
            "macro_f1": metrics(te.medical_specialty, bp)["macro_f1"]
            - metrics(te.medical_specialty, sp)["macro_f1"],
        }
    save(ART / "outer_results.json", evaluated)
    save(ART / "bert_minus_svm_bootstrap.json", comparison)
    print(
        "Frozen winner:",
        choices["random"]["name"],
        choices["random"]["policy"],
        flush=True,
    )
    for r in evaluated:
        if r["stage"] == "selected":
            print(
                r["regime"],
                r["test_metrics"]["accuracy"],
                r["test_metrics"]["macro_f1"],
                r["test_metrics"]["minimum_f1"],
                flush=True,
            )


if __name__ == "__main__":
    main()
