"""Render professor-ordered English Markdown using measured artifacts and reviewed errors."""

import json, pandas as pd
from config import ROOT, ART, LABELS, load, save


def tab(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, r)) + " |" for r in rows],
        ]
    )


def pct(x):
    return f"{100*x:.1f}%"


def main():
    audit = load(ART / "data_audit.json")
    manifest = load(ART / "selection_manifest.json")
    results = load(ART / "outer_results.json")
    aug = load(ART / "augmentation_results.json")
    notes = load(ROOT / "paper/error_notes.json")
    names = {
        "baseline": "Naive Bayes baseline",
        "word_lr": "Word TF-IDF + LR",
        "wordchar_svm": "Field word/character SVM",
        "lsa": "LSA + LR",
        "minilm_lr": "Frozen MiniLM + LR",
        "clinicalbert": "Fine-tuned ClinicalBERT",
    }
    core = {
        (r["regime"], r["spec"]["family"]): r for r in results if r["stage"] == "bare"
    }
    json_mem = {f: next(v["memorisation_rate"] for k, v in load(ART / "memorisation.json").items()
                        if k.startswith(f + "_")) for f in names}
    main_table = tab(
        ["Model", "Random acc.", "Random macro-F1", "Grouped acc.", "Grouped macro-F1", "Mem. rate"],
        [
            [
                names[f],
                pct(core["random", f]["test_metrics"]["accuracy"]),
                f"{core['random',f]['test_metrics']['macro_f1']:.3f}",
                pct(core["grouped", f]["test_metrics"]["accuracy"]),
                f"{core['grouped',f]['test_metrics']['macro_f1']:.3f}",
                f"{json_mem[f]:.2f}",
            ]
            for f in names
        ],
    )
    selected = {r["regime"]: r for r in results if r["stage"] == "selected"}
    winner = manifest["winner"]
    random = selected["random"]
    grouped = selected["grouped"]
    m = random["test_metrics"]
    perclass = tab(
        ["Specialty", "Train n", "Test n", "P", "R", "F1"],
        [
            [
                l,
                int(audit["class_counts"][l]),
                int(m["per_class"][l]["support"]),
                f"{m['per_class'][l]['precision']:.2f}",
                f"{m['per_class'][l]['recall']:.2f}",
                f"{m['per_class'][l]['f1-score']:.2f}",
            ]
            for l in LABELS
        ],
    )
    examples = []
    for n in notes:
        pred = pd.read_csv(
            ART / f"{n['regime']}_selected_predictions.csv", keep_default_na=False
        )
        r = pred[pred.source_record == n["source_record"]].iloc[0]
        assert not r.correct
        excerpt = n.get("excerpt") or " ".join(r.description.split())[:240]
        examples.append(
            f"- **{n['regime'].capitalize()} test, source record {n['source_record']}**: “{excerpt}” Predicted **{r.predicted}**, supplied label **{r.medical_specialty}**. {n['interpretation']}"
        )
    routes = pd.read_csv(ART / "random_route_performance.csv")
    route_text = "; ".join(
        f"{r['route']}: {int(r['records'])} cases, {pct(r['accuracy'])} accuracy"
        for _, r in routes.iterrows()
    )
    aug_table = tab(
        ["Regime", "Augmentation", "Acc.", "Macro-F1", "Minority mean F1"],
        [
            [
                r["regime"],
                r["augmentation"],
                pct(r["metrics"]["accuracy"]),
                f"{r['metrics']['macro_f1']:.3f}",
                f"{r['metrics']['minority_mean_f1']:.3f}",
            ]
            for r in aug
        ],
    )
    aug_details = []
    minority = ["Dermatology", "Psychiatry-Psychology", "Ophthalmology", "Neurosurgery"]
    aug_labels = tab(
        [
            "Regime",
            "Augmentation",
            "Derm F1",
            "Psych F1",
            "Ophth F1",
            "Neurosurg F1",
            "Minimum F1",
        ],
        [
            [
                r["regime"],
                r["augmentation"],
                *[f"{r['metrics']['per_class'][l]['f1-score']:.3f}" for l in minority],
                f"{r['metrics']['minimum_f1']:.3f}",
            ]
            for r in aug
        ],
    )
    for regime in ["random", "grouped"]:
        baseline = next(
            r for r in aug if r["regime"] == regime and r["augmentation"] == "none"
        )
        for kind in ["resample", "dropout", "sections"]:
            row = next(
                r for r in aug if r["regime"] == regime and r["augmentation"] == kind
            )
            aug_details.append(
                f"{regime} {kind}: Δ macro-F1 {row['metrics']['macro_f1']-baseline['metrics']['macro_f1']:+.3f}"
            )
    configuration = tab(
        [
            "Regime",
            "Comparison",
            "Family",
            "Condition",
            "Parameters / variant",
            "Rules",
        ],
        [
            [
                r["regime"],
                r["stage"],
                names[r["spec"]["family"]],
                r["spec"]["condition"],
                json.dumps(
                    {
                        k: v
                        for k, v in r["spec"].items()
                        if k not in ["family", "condition"]
                    }
                ),
                r["policy"],
            ]
            for r in results
            if r["stage"] in ["bare", "full", "selected"]
        ],
    )
    bootstrap = load(ART / "random_selected_bootstrap.json")
    difference = load(ART / "bert_minus_svm_bootstrap.json")
    random_counts = next(
        x["counts"] for x in audit["split_statistics"] if x["regime"] == "random"
    )
    grouped_counts = next(
        x["counts"] for x in audit["split_statistics"] if x["regime"] == "grouped"
    )
    metadata = {
        (r["regime"], r["spec"]["condition"], r["policy"]): r
        for r in results
        if r["stage"] == "metadata"
    }
    metadata_table = tab(
        ["SVM input / rules", "Random F1", "Grouped F1"],
        [
            [
                title,
                f"{metadata['random',condition,policy]['test_metrics']['macro_f1']:.3f}",
                f"{metadata['grouped',condition,policy]['test_metrics']['macro_f1']:.3f}",
            ]
            for condition, policy, title in [
                ("clinical", "none", "No keywords"),
                ("stripped", "none", "First specialty keyword removed"),
                ("all", "none", "All keywords"),
                ("clinical", "keyword", "No keywords + keyword rule"),
                ("clinical", "memory", "No keywords + duplicate rule"),
                ("clinical", "both", "No keywords + both rules"),
            ]
        ],
    )
    bare_best = max(names, key=lambda f: core["random", f]["test_metrics"]["macro_f1"])
    leading_score = core["random", bare_best]["test_metrics"]["macro_f1"]
    leading_models = " and ".join(
        names[f]
        for f in names
        if leading_score - core["random", f]["test_metrics"]["macro_f1"] < 0.001
    )
    random_errors = pd.read_csv(ART / "random_errors.csv", keep_default_na=False)
    common_pairs = (
        random_errors.groupby(["medical_specialty", "predicted"])
        .size()
        .sort_values(ascending=False)
    )
    common_gold, common_pred = common_pairs.index[0]
    common_text = f"The largest residual confusion is {common_gold} → {common_pred} ({int(common_pairs.iloc[0])} of {len(random_errors)} errors)."
    bert_random = core["random", "clinicalbert"]["test_metrics"]["macro_f1"]
    svm_random = core["random", "wordchar_svm"]["test_metrics"]["macro_f1"]
    bert_grouped = core["grouped", "clinicalbert"]["test_metrics"]["macro_f1"]
    svm_grouped = core["grouped", "wordchar_svm"]["test_metrics"]["macro_f1"]
    chosen_settings = ", ".join(
        f"{k}={v}"
        for k, v in winner["spec"].items()
        if k not in ["family", "condition"]
    )
    mem = load(ART / "memorisation.json")
    mem_of = {f: next(v for k, v in mem.items() if k.startswith(f + "_")) for f in names}
    rep = mem_of["wordchar_svm"]
    mem_rates = sorted(v["memorisation_rate"] for v in mem.values())
    splits_all = pd.read_csv(ART / "random_splits.csv", keep_default_na=False)
    grp = splits_all.groupby("group").medical_specialty.agg(list)
    multi = grp[grp.str.len() > 1]
    repeat_label = int((multi.str.len() != multi.apply(lambda v: len(set(v)))).sum())
    organs = {"Orthopedic", "Cardiovascular-Pulmonary", "Gastroenterology", "Obstetrics-Gynecology",
              "Neurosurgery", "Ophthalmology", "Dermatology", "Neurology"}
    surg_share = {}
    for f, key in (("clinicalbert", "clinicalbert"), ("wordchar_svm", "wordchar_svm")):
        path = sorted(ART.glob(f"grouped_bare_{key}_*_predictions.csv"))[0]
        g = pd.read_csv(path, keep_default_na=False)
        err = g[~g.correct.astype(str).str.lower().isin(["true", "1"])]
        surg_share[f] = (int(((err.medical_specialty == "Surgery") & err.predicted.isin(organs)).sum()), len(err))
    md = f"""# Specialty classification: representations, metadata, and repeated notes

**Group XXX** — Author 1 (xx%), Author 2 (xx%), Author 3 (xx%), Author 4 (xx%) ⚠ *mandatory: names and % contribution of each member*

## Introduction

We classify clinical cases into 12 specialties for the course task [6]. Our research question is: **How do lexical and pretrained representations compare when specialty metadata is available versus absent, and when notes are repeated versus unseen?** We separate model predictions from keyword and duplicate-label rules, because aggregate success may depend on corpus structure as well as clinical language.

## Data

### Data Analysis

The supplied training set contains 2,616 records. Three corrupted records with invalid labels and empty clinical fields are logged and excluded, leaving 2,613. Surgery has 876 cases versus Dermatology's 25 (35:1); {audit['missing_fields']['keywords']} cases lack keywords and {audit['missing_fields']['transcription']} lack transcriptions.

Two inspected patterns guide our experiments. First, {audit['recognized_first_keywords']} keyword lists begin with a recognizable specialty (e.g., `orthopedic`), agreeing with {audit['keyword_agreement']} supplied labels. Second, {audit['mixed_label_note_groups']} cleaned note groups have conflicting labels: “Left lower extremity venous Doppler ultrasound” appears in Cardiovascular-Pulmonary and Radiology. Identical cleaned three-field clinical tuples impose an empirical deterministic label-matching ceiling of {pct(audit['clinical_tuple_deterministic_empirical_accuracy_bound'])} on this corpus; differing keywords can resolve this ambiguity. Copies almost never repeat a label: in {len(multi) - repeat_label} of {len(multi)} cleaned note groups with several rows, every copy carries a different specialty. Cleaning removes embedded records from seven oversized descriptions and normalizes section separators.

### Data Augmentation

For Dermatology, Psychiatry-Psychology, Ophthalmology and Neurosurgery, we target about 150 cases per class, adding at most four variants per original. We compare resampling, matched field dropout, and section recombination. Dropout removes keywords, sample name with probability .5, and transcription with probability .35 when a description remains. Recombination replaces each eligible section with probability .5, drawing the same header from another fitting case of the same specialty. Donor pools are rebuilt within each fitting fold; all base/donor ids are saved. Shared specialty does not guarantee coherent synthetic histories.

## Models

Six families address the research question. The official **Naive Bayes baseline** uses lowercased, Porter-stemmed description TF-IDF and Multinomial NB (α=1). **Word TF-IDF + logistic regression (LR)** tests lexical evidence; **field word/character support vector machine (SVM)** concatenates normalized features per field to capture phrases and spelling fragments. **Latent semantic analysis (LSA) + LR** uses a 128-dimensional TF-IDF projection. **Frozen MiniLM + LR** averages normalized nonoverlapping 220-token chunk embeddings, renormalizes each field mean and concatenates four 384-dimensional vectors; empty fields use the empty-text embedding [1–3]. **ClinicalBERT** is fully fine-tuned [4]: head-only retains 510 word pieces; head+tail keeps the first 128 and last 382, plus special tokens.

Inputs either omit keywords, remove their first specialty clue, or retain all keywords. Separate rules use a fixed specialty mapping or a fitting-only mapping (≥2 observations, ≥95% purity); duplicate-label exclusion removes specialties already attached to fitting copies of a repeated note before choosing the remaining model score. Keyword decisions take precedence. Duplicate exclusion is a corpus hypothesis, not a clinical guarantee.

## Experimental Setup

Random partitions permit repeated notes across boundaries; grouped partitions keep each cleaned normalized note together, using clinical fields for empty notes. Both stratify by label. The first outer and first inner development folds of seeded five-fold divisions yield random train/validation/test {random_counts['train']}/{random_counts['validation']}/{random_counts['test']} and grouped {grouped_counts['train']}/{grouped_counts['validation']}/{grouped_counts['test']}, with all classes present. All compared models fit the same training rows. Development selects configurations, checkpoints and rules by macro-F1, then minimum class F1, then accuracy. The submission uses the declared random development winner, frozen before outer scoring and refitted on all valid labeled cases.

Earlier runs explored labeled data and inspected unlabeled official inputs; this is a retrospective rerun, not new external confirmation. The merged analysis uses labeled cases only, and official inputs are loaded solely for final prediction. Accuracy, macro-F1 and per-class precision/recall/F1 are computed with scikit-learn [5].

### Parameters/Hyperparameters

Partition/model seeds are 20261008/42. Inverse regularization C is .3/1/3 for SVM and 1/10 for LR; ordinary/balanced weights are compared (LSA balanced). SVM word 1–2-grams have min_df=2 and ≤40,000 features per field; character word-boundary 3–5-grams have min_df=3 and ≤30,000. Word LR/LSA use word 1–2-grams, min_df=2 and ≤60,000 features. These comparators use sublinear TF and L2 normalization; the reference uses TF-IDF defaults. BERT uses AdamW, learning rate 3e-5, weight decay .01, 10% warm-up, square-root inverse-frequency class weights, three epochs, and ≤512 tokens. Effective batches ≤16 accumulate microbatches ≤8 with exact weighted normalization, including partial batches. Pinned revisions, versions and full configurations accompany the code; computation uses M4/16 GB.

## Results

Table 1 compares development-selected bare models, removing the first specialty keyword (the reference uses description only).

{main_table}

*Table 1. Common-partition model comparisons; F1 is macro-averaged. Mem. rate: share of the {rep['repeated_test_notes']} random-test notes that repeat a fitting note for which the bare model outputs a specialty that note already carries in the fitting data.*

The submission choice is **{names[winner['spec']['family']]}** ({chosen_settings}), input `{winner['spec']['condition']}`, rules `{winner['policy']}`: random test accuracy **{pct(m['accuracy'])}**, macro-F1 **{m['macro_f1']:.3f}**, minimum class F1 **{m['minimum_f1']:.3f}**. The independently selected grouped system achieves {pct(grouped['test_metrics']['accuracy'])} accuracy and {grouped['test_metrics']['macro_f1']:.3f} macro-F1. Table 2 and Figure 1 describe the submission-selected system on the random test.

{perclass}

*Table 2. Full training distribution and internal-test support, precision (P), recall (R) and F1.*

![Random-test confusion matrix](../artifacts/random_confusion.png)

*Figure 1. Rows: gold; columns: predicted. Cardio=Cardiovascular-Pulmonary; Derm=Dermatology; Gastro=Gastroenterology; GenMed=General Medicine; Neuro=Neurology; Neurosurg=Neurosurgery; ObGyn=Obstetrics-Gynecology; Ophth=Ophthalmology; Ortho=Orthopedic; Psych=Psychiatry-Psychology; Radio=Radiology; Surg=Surgery.*

{metadata_table}

*Table 3. Metadata/rule ablations. Each input condition tunes SVM settings on bare development macro-F1; the last three rows share the no-keywords model. The submission tunes the full system, so its configuration can differ. Rules inspect original keywords even when model features omit them.*

## Discussion

The research question has two parts. For representation, {leading_models} attain the highest random bare scores (within .001). Fine-tuning does not consistently lead: ClinicalBERT reaches macro-F1 {bert_random:.3f} on random and {bert_grouped:.3f} on grouped partitions, against {svm_random:.3f} and {svm_grouped:.3f} for the SVM. For repeated versus unseen notes, a repeated note is the *hardest* case, not the easiest: {rep['repeated_test_notes']} of {rep['test_rows']} random-test notes repeat a fitting note, and for {rep['gold_label_new_for_repeated_notes'].split('/')[0]} of them the gold specialty is one the note does not yet carry. Bare models reproduce the fitting copy's label for {mem_rates[0]:.0%}–{mem_rates[-1]:.0%} of these notes (Table 1). The SVM does so for all of them (accuracy {rep['accuracy_on_repeated']:.1%} on repeated vs {rep['accuracy_on_unseen']:.1%} on unseen notes), which explains why every model scores lower on random than on grouped partitions. Memorisation and generalisation are separate: the SVM memorises most and is still among the best on unseen notes. For metadata, Table 3 shows its contribution; removing only the first clue still retains other keyword evidence. The selected system routes are {route_text}. Its success combines classification with metadata and corpus structure; transfer to hospital data lacking those clues is unproven.

Augmentation diagnostics pool three-fold predictions on outer training/development cases using previously selected SVM settings. Changes in macro-F1 are {'; '.join(aug_details)}. Matched resampling separates sampling weight from dropout; fold-local donors prevent heldout-section leakage. In an inspected variant of Ophthalmology source 1110, an eyelid-repair title is retained while donor 1143 supplies lens-removal procedure text: label preservation does not ensure consistency. These post-selection diagnostics are descriptive, and minority F1 is unstable. Full tables and synthetic examples accompany the draft.

{common_text} Across models the dominant bare error is a *Surgery* note predicted as an organ specialty ({surg_share['clinicalbert'][0]} of {surg_share['clinicalbert'][1]} ClinicalBERT and {surg_share['wordchar_svm'][0]} of {surg_share['wordchar_svm'][1]} SVM grouped-test errors), where the duplication described in Data Analysis makes both labels defensible. Four actual selected-system mistakes illustrate the limitations:

{chr(10).join(examples)}

These readings do not establish that supplied labels are wrong. Paired note-group bootstrap estimates compare BERT with SVM; they are not equivalence tests. The selected random system's conditional 95% macro-F1 interval is [{bootstrap['macro_f1_95_interval'][0]:.3f}, {bootstrap['macro_f1_95_interval'][1]:.3f}]. Prior exploration and selection uncertainty are excluded, and exact grouping misses near-duplicates. Official performance remains unknown.

## Future Work

Repeat across outer folds/seeds; test clinically coherent augmentation, whole-note encoding and multi-label targets for overlapping specialties. Evaluate a calibrated lexical/neural ensemble if development evidence supports it.

## Bibliography

[1] Reimers and Gurevych (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084).

[2] Wang et al. (2020). [MiniLM](https://arxiv.org/abs/2002.10957).

[3] Sentence Transformers. [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

[4] Alsentzer et al. (2019). [Publicly Available Clinical BERT Embeddings](https://aclanthology.org/W19-1909/); [Bio_ClinicalBERT checkpoint](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).

[5] Pedregosa et al. (2011). [Scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html).

[6] Course assignment and supplied template (2026); supplied corpus adapted from Kaggle. No external labeled cases were added.

Software: Python, NumPy, pandas, SciPy, scikit-learn, NLTK, Matplotlib, PyTorch, Transformers and Sentence Transformers (versions/revisions supplied).

**LLM declaration:** Claude assisted earlier design, code and drafting. OpenAI Codex reviewed both runs, merged/corrected code, executed experiments, inspected data/errors and drafted this paper. Metrics come from saved predictions; neither assistant supplied case predictions or generated synthetic clinical assertions. The group should amend this declaration to reflect actual review and contributions.

## Appendix A: Extra Figures and Tables

*Table A1. Augmentation diagnostics (three-fold pooled predictions on outer training/development cases).*

{aug_table}

*Table A2. Memorisation on the random test (bare models).*

{tab(["Model", "Mem. rate", "Acc. repeated", "Acc. unseen"], [[names[f], f"{mem_of[f]['memorisation_rate']:.2f}", f"{mem_of[f]['accuracy_on_repeated']:.3f}", f"{mem_of[f]['accuracy_on_unseen']:.3f}"] for f in names])}

![Grouped-test confusion matrix](../artifacts/grouped_confusion.png)

*Figure A1. Grouped-test confusion matrix of the separately selected grouped system.*

**Formatting:** English Markdown working draft. Final submission requires the supplied template, ≤3 main pages, bibliography allowance and ≤1 appendix page; group-supplied names/contributions and pagination require final checking.
"""
    (ROOT / "paper/draft.md").write_text(md)
    rules = {
        (r["regime"], r["spec"]["family"], r["policy"]): r
        for r in results
        if r["stage"] == "rules"
    }
    rule_table = tab(
        [
            "Regime",
            "Family",
            "Fixed base candidate",
            "Model F1",
            "+Keyword F1",
            "+Duplicate F1",
            "+Both F1",
        ],
        [
            [
                regime,
                names[family],
                rules[regime, family, "none"]["name"],
                *[
                    f"{rules[regime,family,p]['test_metrics']['macro_f1']:.3f}"
                    for p in ["none", "keyword", "memory", "both"]
                ],
            ]
            for regime in ["random", "grouped"]
            for family in names
        ],
    )
    bert_rows = []
    for regime in ["random", "grouped"]:
        for variant in ["head", "headtail"]:
            development = load(ART / f"{regime}_clinicalbert_{variant}_validation.json")
            bare = next(r for r in development if r["policy"] == "none")
            best = max(
                development,
                key=lambda r: (
                    r["metrics"]["macro_f1"],
                    r["metrics"]["minimum_f1"],
                    r["metrics"]["accuracy"],
                ),
            )
            bert_rows.append(
                [
                    regime,
                    variant,
                    bare["spec"]["epochs"],
                    f"{bare['metrics']['macro_f1']:.3f}",
                    best["policy"],
                    f"{best['metrics']['macro_f1']:.3f}",
                ]
            )
    bert_development = tab(
        [
            "Regime",
            "Variant",
            "Selected epoch",
            "Bare validation F1",
            "Best rule policy",
            "Full validation F1",
        ],
        bert_rows,
    )
    details = (
        "# Experiment details and supplementary results\n\n## Selected configurations\n\n"
        + configuration
        + "\n\n## BERT variant selection on development data\n\n"
        + bert_development
        + "\n\nCheckpoint epochs are selected by bare development performance. Bare and full-system comparisons can select different variants; outer scores do not change those choices.\n"
        + "\n\n## Augmentation diagnostics\n\n"
        + aug_table
        + "\n\n## Minority-class augmentation results\n\n"
        + aug_labels
        + "\n\n## Rules on fixed development-selected family candidates\n\n"
        + rule_table
        + "\n\nEach row keeps the representation and fitted model fixed. Family candidates were selected using full-system development performance; model-only columns need not match the separately selected bare models. Grouped testing has no fitting-note overlap, so the duplicate route should contribute nothing.\n"
        + "\n\n## Paired BERT minus SVM group bootstrap\n\n```json\n"
        + json.dumps(difference, indent=2)
        + "\n```\n\n## Synthetic examples\n\n"
    )
    for regime in ["random", "grouped"]:
        for kind in ["dropout", "sections"]:
            a = pd.read_csv(
                ROOT / f"data/augmented/{regime}_{kind}_development.csv",
                keep_default_na=False,
            )
            example = (
                a[a.donor_source_records != "[]"].head(1)
                if kind == "sections"
                else a.head(1)
            )
            for _, r in example.iterrows():
                details += f"- {regime}/{kind}: base {int(r.base_source_record)}, donors {r.donor_source_records}; description: {r.description[:180]}; transcription excerpt: {r.transcription[:380]}\n"
    details += "\n## Interpretation and source lineage\n\nFull donor ids and fold indices are preserved in data/augmented. Shared specialty cannot ensure coherent histories; section recombination may mix ages, medications and diagnoses. Dropout changes field availability rather than clinical assertions. These details complement the main paper and do not replace its required argument.\n"
    details += "\nSee [augmentation_review.md](augmentation_review.md) for inspected before/after examples with specific mismatched sections and donor ids.\n"
    (ROOT / "paper/experiment_details.md").write_text(details)
    print("Wrote paper/draft.md and paper/experiment_details.md", flush=True)


if __name__ == "__main__":
    main()
