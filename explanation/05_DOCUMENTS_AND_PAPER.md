# 5. Every written document explained

The paths below are relative to `llm-final`. Click a file to open it. The main README now includes GitHub setup and commands from inside the checkout. Some historical documents still mention `llm-codex-merged`; [lesson 7](07_RUN_AND_DEFENSE.md) gives commands from the parent project folder.

## Documents in the main folder

| Document | What it is for | What to remember |
| --- | --- | --- |
| [README.md](../README.md) | Main overview, setup, protocol and results | Start-up instructions and the study's structure |
| [PROFESSOR_REQUIREMENTS.md](../PROFESSOR_REQUIREMENTS.md) | Maps assignment requirements to our evidence | What must appear in the paper and submission |
| [MODEL_EXPLANATIONS.md](../MODEL_EXPLANATIONS.md) | Short technical notes about models, rules and metrics | More detail after the beginner model lesson |
| [DEFENSE_GUIDE.md](../DEFENSE_GUIDE.md) | Original guide for the live demonstration | What to run and what it demonstrates |
| [MERGE_STATUS.md](../MERGE_STATUS.md) | Records completed work, corrections and limitations | The method was corrected and actually executed; some history refers to the base merge |
| [IMPROVEMENTS.md](../IMPROVEMENTS.md) | Lists what the final copy adds | Memorisation analysis, fourth error, paper additions and output formatting |
| [protocol.json](../protocol.json) | A structured record of experiment choices | Splits, settings, selection rule and experiment scope |
| [requirements.txt](../requirements.txt) | Points the installer to the locked dependency list | A convenient installation entry point |
| [requirements-lock.txt](../requirements-lock.txt) | Exact versions of installed Python packages | Helps another machine use the same software versions |
| [results.txt](../results.txt) | The official submission predictions | 409 labels, no header, matching original record order; not evaluation scores |
| [.gitignore](../.gitignore) | Tells Git which local files to ignore | Environments, caches and compiled/system files are excluded; archived logs are included |

JSON means a structured text file with named fields. You can read it in a text editor. It holds settings or evidence, rather than a narrative.

## Files in paper/

| Document | What it is for | Connection to the question |
| --- | --- | --- |
| [draft.md](../paper/draft.md) | Main English manuscript | Tells the complete research story |
| [experiment_details.md](../paper/experiment_details.md) | Longer tables, choices, comparisons and intervals | Supports the claims with fuller evidence |
| [augmentation_review.md](../paper/augmentation_review.md) | Inspected synthetic examples | Shows why same-class section replacement may create inconsistent cases |
| [error_notes.json](../paper/error_notes.json) | Source IDs, excerpts and human interpretations | Lets the report connect actual mistakes to the research question |
| [FORMAT_NOTES.md](../paper/FORMAT_NOTES.md) | Instructions for turning Markdown into the final PDF | Helps preserve required content within the page limits |

`report.py` regenerates `draft.md` and `experiment_details.md`. Editing generated prose and then running the report can overwrite those edits. The error interpretations are stored separately in `error_notes.json`.

## The paper, section by section

| Section | The question it must answer | Our content |
| --- | --- | --- |
| Introduction | What are we investigating? | Representation, metadata and repeated notes |
| Data Analysis | What did we find by inspecting cases? | Label imbalance, specialty tags, conflicting copies, corruption |
| Data Augmentation | How did we add training examples? | Resampling, missing fields and section replacement |
| Models | What did we compare, and why? | Six families and explicit rules |
| Experimental Setup | Is the comparison fair? | Shared fitting data, separate validation/test, random/grouped regimes |
| Parameters/Hyperparameters | Could someone reproduce it? | Seeds, feature settings, C, class weights and neural training settings |
| Results | What happened? | Bare comparisons, full-system scores, per-class results, confusion matrix |
| Discussion | Why did it happen? | Shortcut dependence, memorisation, augmentation and four actual errors |
| Future Work | What would we improve? | More folds/seeds, coherent augmentation and possible ensembles |
| Bibliography | Where did methods and help come from? | Papers, pretrained models, software and LLM declaration |
| Appendix | What extra evidence helps? | Supplementary tables and grouped confusion matrix |

### Table 1: compare representations

The model rows show bare performance. “Mem. rate” shows how often repeated-note predictions reproduce a fitting label. This table connects model type to both performance and memorisation.

### Table 2: do small classes work?

“Train n” means the **full labelled corpus class count**, not the count in the 1,672-case fitting partition. “Test n” is that class's random internal-test count. Precision, recall and F1 describe the final selected system.

### Table 3: what do metadata and rules add?

It changes keyword availability and decision rules. Settings are chosen on development data for each input condition. It is therefore not a table where every row necessarily shares exactly the same fitted model.

The last three rows do share the selected no-keywords base model. Rules can inspect original keywords even when the model omits them.

### Appendix

Table A1 reports augmentation diagnostics. Table A2 reports repeated/unseen performance. Figure A1 shows the independently selected grouped system's errors.

The Markdown appendix is still a working layout. It must fit one page after typesetting.

## Logs: receipts of earlier runs

A `.log` file is captured terminal output: progress, warnings, scores or failure messages. It is not a scientific argument or a trained model.

Each log has its own entry in [the complete index](09_FILE_BY_FILE.md). In particular:

- `training_queue.log` records an old launcher error after the neural jobs completed.
- `training_queue_resume.log` records the successful check of all four completed comparisons.
- `defense_demo.log` is an earlier demo transcript; the reviewed final copy reproduced the same metrics in about 6.8 seconds.
- `verification.log` records the audit. It supports reproducibility, but cannot prove final PDF pagination or erase historical data inspection.

Copied logs can mention the earlier directory name or an older run. For current scores, use the measured artifacts and final paper.

## Supporting folders

`artifacts/` contains experiment evidence. `data/augmented/` contains added training examples and donor histories. Lesson 9 explains every file in both.

`.venv/` contains installed Python software. `cache/` stores downloads and reusable calculations. In this final folder both are links into the earlier merged folder. They are machine support, not additional models to describe in the paper. Copying the final folder alone may break those links.

`__pycache__/` contains automatically compiled copies of Python modules. Read the corresponding `.py` source when learning the program.
