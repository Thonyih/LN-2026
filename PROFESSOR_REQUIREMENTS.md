# Professor requirements and implementation map

Sources: `../StudentsPack-v2/Project-2026-Description.pdf` and `../StudentsPack-v2/Project-Template/template.tex` (identical to the first pack). On 10 Oct 2026 the professor released a cleaned test set (398 rows), asked groups to run their best model on it, reminded that the whole training set may be used, and pointed to the template's information. Deadline: 16 October 2026, 23:59, via Fenix. This file distinguishes computed project deliverables from group-supplied final submission details.

| Requirement | Implementation / evidence |
| --- | --- |
| Classify each case into one of 12 specialties | Shared alphabetical LABELS in src/config.py; output validity checks |
| At least three models | Baseline NB, word LR, field word/character SVM, LSA, frozen MiniLM, ClinicalBERT |
| Explicit research question | paper/draft.md: representations, metadata availability, and repeated versus unseen notes |
| Data analysis with at least two concrete observations | Labeled-data audit: specialty-name keywords, conflicting duplicate labels, corruption, missing fields |
| Mandatory augmentation | Fold-local section recombination; matched resampling/dropout controls with complete lineage |
| Reproducible experimental setup | protocol.json, source-row split files, model revisions, requirements-lock.txt, seed and config manifests |
| Best model results by class | Selected-system reports in artifacts, regenerated paper table |
| Comprehensive confusion matrix | artifacts/random_confusion.png and grouped_confusion.png |
| At least three misclassification examples | Saved selected-system error records; reviewed excerpts and interpretations in paper/draft.md |
| Limitations and answer to the research question | Metadata subgroup results, ambiguity bound, group-bootstrap intervals, retrospective exploration disclosure |
| Papers, external sources, pretrained models, explicit LLM-use declaration | Bibliography and model revision records; declaration names both assistants accurately |
| Official test only for generating results.txt; never training | Only src/predict.py calls the official loader; other stages use train.csv alone |
| Same preprocessing at training and inference | Shared src/data.py::clean |
| results.txt, no header, one label per input record in original order | Final prediction checks and artifacts/final_output_verification.json |
| Accuracy >49%; no class F1 <25% for full automatic component | Internal scores are evidence, not official-test guarantees; official labels unavailable |
| Paper max 3 pages, supplied organization; bibliography excluded; appendix max 1 page | Working deliverable is English Markdown as requested; PDF pagination remains a final group task |
| Authors and contribution percentages mandatory | Group must supply these; no invented names or percentages |
| NUM.zip containing NUM.pdf, code, needed added data, results.txt; exclude model weights | package_submission.py prepares an archive once the group's final PDF and number are supplied |
| Be able to reproduce and explain the work orally | README and MODEL_EXPLANATIONS.md plus executed checks |

Historical qualification: previous runs inspected the unlabeled official inputs for data analysis. The merged analysis uses only labeled data and does not assert that this erases prior inspection. No official labels were available or used.

The template allocates the 16 paper points as follows: Introduction 1.5; Data Analysis 1.5; Data Augmentation 2; Models 4; Experimental Setup 1; Results 1.5; Discussion 3; Future Work 0.5; References 1. The four automatic points depend on external evaluation of results.txt. The assignment specifies a three-point deduction for instructions not followed. The final PDF must therefore preserve the required evidence within three main pages; extra tables do not compensate for an incomplete main paper.
