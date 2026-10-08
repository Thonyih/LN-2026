# 6. Every code tool explained

All 21 Python modules in `src/` are explained below. A **program** is normally run from the terminal; a **helper module** supplies functions that another program calls.

The inputs and outputs listed here are the important ones. [The file index](09_FILE_BY_FILE.md) covers individual saved files. [Lesson 7](07_RUN_AND_DEFENSE.md) gives the commands.

## The launchers

### [run_all.py](../run_all.py): portable full reproduction

Runs the same stages as the shell pipeline using the current Python interpreter. This also supports native Windows environments, which store Python under Scripts rather than bin. Completed BERT comparisons are reused unless --force-bert is supplied. Stage logs and clear failure messages show progress. For classroom practice, start with the short tests in the main README instead of full reproduction.

### [run_all.sh](../run_all.sh): run the full sequence

**Reads:** the prepared Python environment, supplied data and existing experiment settings.

**Does:** tests → prepare → classical comparisons → fitting-data alignment → augmentation candidates → augmentation diagnostics → BERT queue → select/evaluate → official prediction → memorisation → paper → verification.

**Writes:** experiment artifacts, logs, the draft and official predictions.

**Why it matters:** connects the evidence to one reproducible pipeline. It can take substantial time and overwrite generated outputs; use the short demo in class.

### [run_training_queue.sh](../run_training_queue.sh): run neural jobs one at a time

**Does:** random/head, random/head+tail, grouped/head, grouped/head+tail ClinicalBERT runs.

**Writes:** BERT scores, training histories and logs.

**Why it matters:** covers both repetition regimes and both long-note strategies without running memory-heavy jobs simultaneously on the M4.

It reuses completed jobs by default. `--force` retrains them.

## Preparation and shared helpers

### [config.py](../src/config.py): the shared address book

**Helper.** Defines project/data/cache paths, the canonical order of 12 labels, seeds and pretrained model revisions.

`save` and `load` write/read JSON; `sha` calculates a file fingerprint.

**Why it matters:** every program agrees about where files live and which score column belongs to which specialty.

### [data.py](../src/data.py): read and clean cases

**Helper.** Reads semicolon CSV records, keeps original source IDs, logs corrupt labelled records, cleans fields and calculates note-group IDs.

`clean` chooses clinical, stripped or all inputs. `note_keys` identifies repeated notes. `write_labels` writes the official label format.

`load_train` reads labelled data. `load_official_for_prediction` is a separate entry point called only by final prediction.

**Why it matters:** comparisons need consistent inputs and trustworthy links back to source cases.

### [prepare.py](../src/prepare.py): make the study's partitions

**Reads:** `StudentsPack/train.csv` through the shared loader.

**Writes:** `random_splits.csv`, `grouped_splits.csv`, `data_audit.json` and `protocol.json`.

**Does:** fixes fitting/validation/internal-test partitions; counts classes, missing fields and conflicting note groups; calculates keyword agreement and the three-field ambiguity bound.

**Why it matters:** provides the concrete observations and fair comparisons needed to answer the research question.

## Building and comparing models

### [models.py](../src/models.py): build the non-BERT learners

**Helper.** `model(spec)` constructs NB, word LR, field word/character SVM, LSA or MiniLM's logistic classifier.

`stem_tokens` reduces word endings for the reference. `field` selects a field; `joined` joins fields. `score` orders prediction columns by the shared label list.

**Why it matters:** this is where the representation alternatives are implemented. Consistent column order prevents a numerical score from being assigned to the wrong specialty.

### [embeddings.py](../src/embeddings.py): make MiniLM text vectors

**Helper.** `embed` splits fields into 220-token chunks, uses the frozen encoder, averages/normalizes chunk vectors and joins the field vectors.

**Writes:** compressed vector caches under `cache/`. Their names depend on input text, condition and model revision.

**Why it matters:** lets us compare pretrained whole-note representations without updating the encoder or repeatedly computing the same vectors.

### [classical.py](../src/classical.py): compare settings on validation

**Reads:** saved partitions.

**Does:** fits the non-BERT configurations on fitting cases, scores validation, and tries the rule policies. `specs` lists candidates; `rank` applies the selection order; `name` makes readable identifiers.

**Writes:** validation scores/metrics, development-selected candidates, and test scores for the chosen comparisons.

**Why it matters:** chooses settings without using internal-test labels to rank them.

### [reconcile_classical_scores.py](../src/reconcile_classical_scores.py): check equal fitting data

**Reads:** selected configurations and score metadata.

**Does:** checks whether each outer comparison used exactly the fitting partition. It regenerates scores only when the recorded fitting IDs differ.

**Why it matters:** a model trained on more cases should not be presented as an equal-data comparison.

### [bert.py](../src/bert.py): fine-tune ClinicalBERT

**Reads:** one regime's fitting/validation/test partitions and pinned pretrained weights.

`texts` orders fields; `make_ids` applies head or head+tail truncation; `train_predict` fits the model and chooses a validation checkpoint.

**Writes:** validation/test score arrays, validation-policy metrics and a training history with epochs and optimizer update counts.

**Why it matters:** gives the fine-tuned language model the same fitting and validation data as the other families. Internal target labels do not select its checkpoint.

### [training_utils.py](../src/training_utils.py): get small-batch training right

**Helper.** `windows` divides fitting cases into effective batches of up to 16. `normalised_micro_loss` divides weighted loss by the correct total weight across that batch.

**Why it matters:** the last smaller batch contributes correctly, and two small GPU batches behave like the intended larger batch. This is an implementation correctness tool, not an extra model.

## Rules and augmentation

### [rules.py](../src/rules.py): apply the decision rules

**Helper.** `first_keywords` reads the first tag. `learn_keywords` learns reliable fitting-only mappings. `apply` combines model scores with a selected policy and records which route decided each case.

Policies: `none`, fixed `keyword`, `learned_keyword`, `memory`, fixed `both`, or `learned_both`.

**Why it matters:** separates the benefit of clinical representation from metadata and repeated-note structure.

### [augment.py](../src/augment.py): add fitting examples

**Helper.** `sections` recognizes headed sections. `generate` creates resampled, dropout or recombined minority cases using only the fitting data supplied to it.

Synthetic IDs are negative, so they cannot be mistaken for real source records. Each variant records its base source and all section donors.

**Why it matters:** implements mandatory augmentation while keeping held-out content out of fitting.

### [augmentation_candidates.py](../src/augmentation_candidates.py): test augmented candidates

**Reads:** the development-selected stripped SVM and partitions.

**Does:** fits resampling, dropout and section variants, then tries their rule policies on validation.

**Writes:** `augmentation_validation.json`, validation/test score arrays and `*_development.csv` synthetic cases.

**Why it matters:** gives augmented systems an opportunity to be selected by development performance.

### [augmentation_study.py](../src/augmentation_study.py): compare augmentation effects

**Reads:** outer fitting + validation cases and the previously chosen stripped SVM settings.

**Does:** three-fold diagnostic comparisons of none/resampling/dropout/sections, under random and grouped splitting. Each case receives a prediction from a fold that did not fit on it.

**Writes:** `augmentation_results.json`, `*_oof.csv` predictions and `*_lineage.csv` donor histories.

**Why it matters:** compares changes with matched controls. Settings were already selected, so these are descriptive diagnostics rather than new independent final estimates.

## Turning predictions into evidence

### [evaluation.py](../src/evaluation.py): calculate and preserve results

**Helper.** `metrics` calculates accuracy, macro-F1, minimum F1 and per-class results. `store_scores` saves score arrays with labels, source IDs and metadata. `read_scores` validates and aligns them. `plot_confusion` draws the matrix.

**Why it matters:** every number can be traced to actual predictions, rather than copied manually.

### [finalize.py](../src/finalize.py): freeze choices, then score them

**Reads:** classical, augmentation and BERT validation candidates.

`validation_rows` collects candidates. The program writes `selection_manifest.json` using validation ranking **before** scoring the outer tests.

It produces bare/full/metadata/rule/selected reports, predictions, error lists, routes, matrices and bootstrap intervals. `bootstrap` resamples whole note groups.

**Why it matters:** this is the boundary between choosing a system and reporting its performance.

### [memorisation.py](../src/memorisation.py): measure repeated-note behaviour

**Reads:** random fitting groups and bare-model prediction CSVs.

**Writes:** `memorisation.json`.

**Does:** counts repeated test notes, whether their gold labels are new, how often models predict an old fitting label, and repeated/unseen accuracy.

**Why it matters:** directly supplies the evidence for the “repeated versus unseen” part of our question.

This script runs its analysis at top level; execute it as a program rather than importing it as a helper.

### [report.py](../src/report.py): generate the paper

**Reads:** measured results, selections, data audit, augmentation, memorisation and reviewed error notes.

`tab` formats Markdown tables; `pct` formats percentages. `main` writes `paper/draft.md` and `paper/experiment_details.md`.

**Why it matters:** puts the evidence in the professor's section order. It does not discover official-test accuracy, invent error interpretations, or create a final PDF.

### [verify.py](../src/verify.py): audit the saved study

**Reads:** partitions, donor histories, candidate scores, selected reports, BERT histories and final-output metadata.

**Does:** recomputes selection and metrics; checks grouped isolation, donor labels, source order, label columns and output format/hash.

**Writes:** `integrity_verification.json`.

**Why it matters:** protects the reproducibility of the claims. Passing it does not certify PDF layout or erase past official-input inspection.

## Submission and demonstration

### [predict.py](../src/predict.py): generate official predictions

**Reads:** the frozen winner, all valid labelled cases and the official unlabelled cases.

**Does:** refits the winner on all 2,613 usable labelled cases, applies its policy, and checks 409 output records.

**Writes:** `results.txt`, `final_prediction_scores.npz` and `final_output_verification.json`.

**Why it matters:** completes the assignment's classification task. No official labels are available, so it cannot calculate official accuracy.

### [defense_demo.py](../src/defense_demo.py): show the selected SVM live

**Reads:** saved choice, labelled course data, partition IDs and archived evidence.

**Does:** refits on 1,672 fitting cases, predicts 523 internal cases, and checks new predictions against the archive. Prints metrics, routes and all four paper errors.

**Why it matters:** demonstrates real fitting and prediction in about seven seconds on the tested M4. The archive checks its answers; it does not supply them.

### [package_submission.py](../src/package_submission.py): build the ZIP

**Reads:** a completed PDF, real group number, final labels, code and needed data/evidence.

**Writes:** `NUM.zip` locally, excluding environments, caches and model weights.

**Why it matters:** assembles the professor's deliverables. It checks that the supplied paper is a PDF, but page count still needs human checking. It does not upload to Fenix.

It currently packages the project under the directory name `llm-codex-merged`. That is the archive layout, even though the working copy is `llm-final`.

## The test file

### [tests/test_integrity.py](../tests/test_integrity.py): six small correctness checks

| Test | The mistake it protects against |
| --- | --- |
| Reversed labels / reordered sources | Scores being attached to the wrong class or case |
| Donor isolation / original preservation | Using an outside donor or accidentally editing original cases |
| Keyword removal | Deleting useful remaining keywords |
| Duplicate exclusion / keyword precedence | Applying rules in the wrong way |
| Final partial window | Losing the last few training cases |
| Weighted microbatch gradient | Small-batch training producing the wrong gradient |

These tests check code behaviour. The full audit additionally checks actual saved experiments.
