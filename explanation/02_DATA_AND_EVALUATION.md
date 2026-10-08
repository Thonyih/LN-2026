# 2. Data and fair evaluation

## What is one case?

A row in `train.csv` contains:

| Field | Meaning | Example invented only to explain the format |
| --- | --- | --- |
| `medical_specialty` | Expected answer, called the label | Orthopedic |
| `description` | Short case summary | Shoulder operation |
| `sample_name` | Name of the procedure or consultation | Rotator cuff repair |
| `transcription` | Full clinical note | A longer description of the operation |
| `keywords` | Supplied tags | orthopedic, shoulder, repair |

Our official prediction file has the four input fields but no `medical_specialty` column. We must predict that missing answer.

The training CSV has a header. The official CSV does not. Both use semicolons, and a quoted field may contain newlines. Therefore “409 cases” means 409 parsed records, not necessarily 409 physical lines in the CSV.

## Why clean the data?

The supplied training file has 2,616 records. Three have invalid labels; we exclude them, keeping 2,613. Some oversized descriptions contain accidentally embedded records; cleaning cuts these fragments away. The same cleaning is used when fitting and predicting.

Labels are imbalanced: **876 Surgery cases versus 25 Dermatology cases**. A model could favour Surgery and neglect small classes. We measure every class and compare balanced training weights.

Each case retains a `source_record` number, its zero-based original data-record position. Source 361 means original record index 361; removing corrupt records does not renumber the remaining sources.

## The two surprising patterns

**Keywords often give away the answer.** A recognized specialty is the first keyword in 2,172 cases, agreeing with 2,171 labels. For example, the first tag may literally say “orthopedic.”

**Copies often have different answers.** The same clinical note can be filed under Surgery and Orthopedic. Remembering the fitting copy's label can then produce the wrong answer for its other copy.

These are corpus patterns. We do not assume that another hospital dataset behaves this way.

## Three partitions: learning, choosing, measuring

| Partition | Cases in each regime | What we use it for | Exam analogy |
| --- | ---: | --- | --- |
| Train / fitting | 1,672 | Learn word weights, features and model parameters | Study material |
| Validation / development | 418 | Choose settings, model family and rules | Practice exam |
| Internal / outer test | 523 | Measure the frozen choice | Final exam |

Validation and development mean the same thing here. “Fitting” is the exact training part used for a comparison.

All compared outer models fit the same 1,672 cases. We rank choices by validation macro-F1, then minimum class F1, then accuracy. We write the selected choice before calculating the final internal-test reports.

The code uses five-fold splitters, but the main experiment evaluates **only the first outer fold**, with one inner validation fold. It is not a five-fold average. Earlier work explored the dataset, so this is a retrospective rerun rather than completely new independent evidence.

## Random and grouped splitting

**Random:** individual rows are divided, while keeping class proportions reasonably similar. Copies of a note can appear in different partitions.

**Grouped:** every copy of a cleaned note goes into the same partition. This tests notes whose group is absent from fitting data.

Example:

| Two copies of note A | Random can do this | Grouped keeps them together |
| --- | --- | --- |
| A labelled Surgery | Train | Train |
| A labelled Orthopedic | Test | Train |

Groups are built from cleaned, lowercased transcription with non-word characters removed. Empty transcriptions use the other clinical fields as a fallback. A hash is a compact identifier for that grouping key. Exact grouping does not catch every near-copy.

## Three input conditions

| Code name | What the model receives | Why compare it? |
| --- | --- | --- |
| `clinical` | Description, procedure name, transcription; no keywords | Tests clinical text without keyword evidence |
| `stripped` | All four fields, with a recognized first specialty tag removed | Removes the obvious tag while keeping other keywords |
| `all` | All supplied fields after cleaning | Measures performance when the supplied clues are available |

The rules are a separate layer. A keyword rule can inspect original keywords even when the model uses `clinical` inputs.

## Data leakage: the mistake we avoid

Leakage means information from a held-out target influences learning. An example is building a synthetic training note using a section from an internal-test note.

Our augmentation uses only fitting cases as base cases and donors. Grouped partitions also keep fitting and validation note groups separate. Saved IDs and tests check these boundaries.

## The official test is a different file

The 523-case internal test comes from labelled training data, so we can score it. The 409-case official file has no supplied answers, so we cannot score it.

After freezing the choice, we refit the winner on all 2,613 labelled cases and generate the official output. The current pipeline reads official inputs only in this prediction stage. The paper honestly discloses that earlier versions inspected official inputs; that history is not erased by the current restriction.
