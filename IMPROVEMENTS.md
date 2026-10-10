# llm-final = llm-codex-merged + improvements (8 Oct 2026)

Base: an exact copy of `../llm-codex-merged` (its protocol, models, saved results and selected system are
unchanged). `.venv` and `cache` are symlinks to that folder. Changes:

| # | Change | Files | Why (professor / grade) |
|---|---|---|---|
| 1 | `results.txt` written with CRLF, no trailing newline, no header — same conventions as the official CSVs; labels unchanged (409/409 identical at the time; see change 7) | `src/data.py` (`write_labels`) | Minimal difference to the teacher's comparison file |
| 2 | Verification asserts that format (and no BOM, no stray spaces, 12 classes); negative test confirmed | `src/verify.py` | Catches regressions before zipping |
| 3 | Memorisation analysis from saved predictions (no retraining): 190/523 random-test notes repeat a fitting note, gold label new for 189; bare models reproduce the copy's label for 49–100% of them; SVM accuracy 0.0% repeated vs 64.6% unseen | `src/memorisation.py` → `artifacts/memorisation.json`; `run_all.sh` | Answers the "repeated vs unseen" half of the RQ with evidence (Discussion 3.0, creativity) |
| 4 | Paper: memorisation column in Table 1 + finding in Discussion; cross-model error pattern (Surgery → organ: 129/224 BERT, 95/213 SVM grouped errors); duplicate regularity (710/712 groups never repeat a label) | `src/report.py` → `paper/draft.md` | Stronger Data Analysis / Discussion |
| 5 | 4th misclassified example: Huntington note filed twice (858 Neurology, 1480 Psychiatry-Psychology) — irreducible error | `paper/error_notes.json` | Template asks for ≥ 3; a verified irreducible case strengthens the "labels set the ceiling" argument |
| 6 | Author/contribution line placeholder; "Appendix A" section (augmentation table, memorisation table, grouped confusion matrix); clearer BERT/SVM sentence; "official Naive Bayes baseline" wording | `src/report.py` | Mandatory author line; template organisation incl. appendix; formal wording |

Checks after the changes: 6/6 unit tests pass; `src/verify.py` passes (incl. the new format check); paper headings
follow the template; body ≈ 1,580 words (+ tables/figure).

## 10 Oct 2026 — cleaned test set from the professor

| # | Change | Files | Why |
|---|---|---|---|
| 7 | Data now read from `../StudentsPack-v2`. Its `test_no_labels.csv` is the old file with 11 broken rows deleted (old rows 17, 241, 324, 357: mega-rows; 18, 242, 325, 358: their overflow lines; 79: empty; 166: blank description; 376: shifted fields), order kept; train.csv, description and template byte-identical | `src/config.py` | Professor: "run your best model on the new, clean test set" |
| 8 | No hard-coded 409: predict/verify use the row count of the current test file, and verify checks that results.txt was produced from that file (SHA-256) | `src/predict.py`, `src/verify.py` | Prevents submitting predictions aligned to the wrong file |
| 9 | results.txt regenerated: 398 lines, CRLF, no trailing newline. It equals the previous 409-line file with exactly those 11 lines removed (no other prediction changed) | `results.txt` | Same model, same remaining notes |
| 10 | `report-draft/report.md` (hand-maintained Markdown report for the LaTeX conversion) incl. the test-set change; zip folder names updated | `report-draft/`, `src/package_submission.py` | Requested working deliverable |
