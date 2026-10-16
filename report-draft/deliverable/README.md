# Help the doctor — clinical specialty classification (LN 2026, group 14)

## Contents
| Item | What it is |
|---|---|
| `14.pdf` | The paper |
| `results.txt` | Predictions for the 398 notes of `test_no_labels.csv`: one label per line, same order, no header |
| `code/` | Project code |
| `StudentPack/` | The course data used (`train.csv`, cleaned `test_no_labels.csv`); the code expects it next to `code/` |

## Reproduce results.txt (CPU only; installing takes a few minutes, the prediction under 30 seconds)
```sh
cd code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/predict.py
```
This fits the submitted model (word/character SVM with the keyword and duplicate rules, settings already chosen)
on all 2,613 valid training notes and writes `code/results.txt`, identical to the `results.txt` above.
No experiments or neural models are re-run. On Windows, use `.venv\Scripts\python.exe` instead of
`.venv/bin/python`.

The other scripts in `code/src/` are the experiments behind the paper (model comparison, ClinicalBERT,
augmentation study, figures); they are included for reference and are not needed to reproduce `results.txt`
(their full package list is in `code/requirements-lock.txt`).
