# LN-2026 — clinical specialty classification

This repository contains the contents of the final project folder, **llm-final**: code, explanations, the paper draft, saved results, added training data and run logs.

Our selected system is a **word/character SVM with a keyword rule and repeated-note label exclusion**. BERT and MiniLM are comparison models.

> **WARNING: BERT TAKES LONG.**
>
> Start with the main tests below. They do not train BERT or download pretrained model weights.
>
> Deliberately retraining all four BERT comparisons can take about an hour on the tested M4, and substantially longer on CPU. A fresh full experiment can take roughly 1–2 hours on that M4, plus downloads.

## 1. Get the project and course data

```sh
git clone https://github.com/Thonyih/LN-2026.git llm-final
cd llm-final
```

Download the professor's **StudentsPack** from the course materials and place it **beside** the checkout:

```text
your-working-directory/
├── StudentsPack/
│   ├── train.csv
│   ├── test_no_labels.csv
│   ├── Project-2026-Description.pdf
│   └── Project-Template/
│       └── template.tex
└── llm-final/
    ├── README.md
    ├── src/
    ├── artifacts/
    └── ...
```

Only the final project folder is uploaded here. The separate StudentsPack is required for the commands below. Keep its original files: the demo checks the training-file fingerprint against the archived experiment.

Saved partitions and added examples are included, but do not replace the original course files.

## 2. Create your own Python environment

The tested setup is **Python 3.14 on macOS / Apple M4**. Install Python 3.14, then use the commands for your system.

Every colleague creates a fresh environment. The author's local environment/cache links depend on another folder and are not uploaded.

### macOS or Linux

From inside the checkout:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

### Windows PowerShell

From inside the checkout:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

For subsequent commands use **.venv/Scripts/python.exe** instead of **python**. Activation is optional.

Installing dependencies needs internet and can take a few minutes. It installs libraries, including PyTorch, **not** pretrained model checkpoints. Exact versions are in [requirements-lock.txt](requirements-lock.txt). Cross-platform installation has not been tested; availability can depend on the OS and processor.

## 3. Main tests — start here

Run these from the checkout directory using its Python environment.

### A. Reproduce the selected model live

```sh
python src/defense_demo.py
```

This fits the selected SVM on **1,672 labelled cases**, predicts **523 internal test cases**, and checks the result against the archive.

Expected output:

```text
Accuracy: 95.79%
Macro-F1: 0.9402
Minimum class F1: 0.8000
PASS: live results match the archived selected-system results.
```

It also prints every class's F1, decision routes and the four paper mistakes. It took approximately **6–7 seconds** on the prepared M4; other computers can differ.

### B. Run six code correctness tests

```sh
python -m unittest discover -s tests -v
```

These check score ordering, donor isolation, keyword handling, rule priority, partial batches and weighted gradients. Expected: **six tests, OK**. They do not train a pretrained model.

### C. Audit the saved experiments

```sh
python src/verify.py
```

This recomputes validation selection/outer metrics and checks donor/group isolation and final-output integrity. It uses archived evidence without rerunning model fitting, and writes artifacts/integrity_verification.json.

Expected: the JSON summary prints with checks marked true and no exception.

### D. Generate the official predictions

```sh
python src/predict.py
```

This refits the frozen SVM on all **2,613 usable labelled cases** and rewrites **results.txt** with **409 labels**, no header, in original input order. It updates final-prediction evidence too.

The matching implementation took about ten seconds in an earlier M4 run. The file uses CRLF and no trailing newline.

**We have no official gold labels. The 95.79% score is an internal-test score, not official accuracy.**

These four commands are the recommended colleague/professor checks. They need no GPU or pretrained downloads after the environment is installed.

## 4. Understand the project

Start with [explanation/README.md](explanation/README.md). It explains the task, models, rules, augmentation, evaluation, results, code tools and individual files, with defense questions.

The complete file index is a dated snapshot. This GitHub preparation adds the portable launcher **run_all.py** and updates this README.

The paper is [paper/draft.md](paper/draft.md), with [experiment details](paper/experiment_details.md) and [augmentation inspection](paper/augmentation_review.md).

## 5. Full reproduction — optional and slower

> **WARNING: BERT TAKES LONG. Use the main tests first.**

The portable launcher uses your current Python interpreter:

```sh
python run_all.py
```

It runs tests, preparation, classical tuning, augmentation, neural-job checks, frozen evaluation, prediction, memorisation, reporting and verification. It prints progress and saves stage logs.

**Completed BERT runs are reused by default.** Their score files and training records are included. Other experiment stages run again.

MiniLM may still download its encoder and compute embeddings when the cache is empty. A fresh clone is therefore slower than a prepared machine even when BERT is reused.

With reusable BERT results and embeddings, estimate **10–30 minutes on the M4**. This is not a measured guarantee for other platforms.

To deliberately retrain all four BERT comparisons:

```sh
python run_all.py --force-bert
```

Original shell launchers are also included for macOS/Linux:

```sh
sh run_all.sh
sh run_training_queue.sh --force
```

They expect .venv/bin/python. On native Windows use the Python launcher.

Full reproduction overwrites generated artifacts and paper files. If choices/errors change, review paper/error_notes.json before relying on regenerated interpretations.

## 6. Research question, protocol and results

**Question:** how do lexical and pretrained representations compare when specialty metadata is available versus absent, and when notes are repeated versus unseen?

| Validation-selected full system | Internal test cases | Accuracy | Macro-F1 | Minimum class F1 |
| --- | ---: | ---: | ---: | ---: |
| Random: clinical SVM, C=0.3, balanced, keyword + duplicate rules | 523 | 95.79% | 0.9402 | 0.80 |
| Grouped: stripped-keyword SVM, C=1, ordinary weights, keyword rule | 523 | 93.50% | 0.9141 | 0.75 |

Both regimes use **1,672 fitting / 418 validation / 523 internal test cases**. Random splitting permits repeated notes across boundaries; grouped splitting keeps a cleaned note's copies together.

All compared outer models fit the same cases. Validation selects by macro-F1, then minimum class F1, then accuracy. We evaluate only the first outer fold, not a five-fold average. The submission choice was frozen before outer scoring.

Earlier work explored labelled data and inspected unlabelled official inputs; the paper discloses this. The current pipeline uses labelled cases for analysis/selection and official inputs only for final prediction.

Augmentation compares resampling, matched field dropout and section recombination. Every base/donor stays in its fitting fold. The selected winner is unaugmented.

MiniLM is frozen at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. ClinicalBERT is fine-tuned from revision d5892b39a4adaed74b92212a44081509db72f87b.

## 7. Other computers and common problems

- **The SVM runs on CPU.** Apple GPU hardware is not required for the main demo or predictions.
- **Neural acceleration:** current code uses Apple MPS when available, otherwise CPU; it does not currently select NVIDIA CUDA.
- **Missing StudentsPack:** place the original supplied folder beside the checkout.
- **No module named ...:** install requirements into the same environment used to run the command.
- **Windows:** use .venv/Scripts/python.exe and run_all.py.
- **Historical logs mentioning llm-codex-merged:** these are transcripts from the base merge; current code resolves its checkout location.
- **Keep artifacts and added data:** the full audit reads the saved score arrays and donor histories.

Only M4/16 GB execution has been tested. CPU scripts are expected to work on other desktop platforms with compatible dependencies; that has not been independently confirmed.

## 8. Final course submission

GitHub is separate from the professor's Fenix submission. You still need a template-based **NUM.pdf**, code/added data and **results.txt** inside **NUM.zip**. Main paper: at most three pages, bibliography excluded; appendix: at most one page. Add actual author names/contribution percentages and check the PDF.

Example for group 3 after its real PDF is ready:

```sh
python src/package_submission.py --group-number 3 --paper-pdf /path/to/3.pdf
```

The package builder currently uses llm-codex-merged as the directory name inside the ZIP. It does not upload anything.

See [PROFESSOR_REQUIREMENTS.md](PROFESSOR_REQUIREMENTS.md) and [paper/FORMAT_NOTES.md](paper/FORMAT_NOTES.md).
