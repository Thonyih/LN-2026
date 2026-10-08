# 7. Run it and explain it in class

Run these commands from the **Main Project** directory containing `StudentsPack` and `llm-final`.

If your terminal is already inside `llm-final`, run `cd ..` first. You do not need to activate the environment because commands name its Python directly.

## The short live demonstration

```sh
llm-final/.venv/bin/python llm-final/src/defense_demo.py
```

It learns the selected SVM again on 1,672 cases and evaluates 523 internal cases. Expect:

```text
Accuracy: 95.79%
Macro-F1: 0.9402
Minimum class F1: 0.8000
PASS: live results match the archived selected-system results.
```

It also prints each class's F1, the decision routes and the four paper mistakes. The review measured 6.8 seconds for fitting/evaluation on the prepared M4; machine load can change the time.

Explain while it runs:

“We already chose the settings on validation data. This demonstration trains that fixed choice again and measures it on our internal held-out cases.”

## Generate the official output

```sh
llm-final/.venv/bin/python llm-final/src/predict.py
```

It refits the selected SVM on all 2,613 usable labelled cases and rewrites `llm-final/results.txt` with 409 predictions. It also updates final-output evidence.

The matching SVM implementation took about ten seconds in the earlier tested run. It runs offline with the prepared environment.

Explain:

“These cases have no supplied labels. We can show that predictions were produced correctly, but we cannot measure their official accuracy.”

## Optional correctness checks

```sh
llm-final/.venv/bin/python -m unittest discover -s llm-final/tests -v
llm-final/.venv/bin/python llm-final/src/verify.py
```

The first runs six code tests. The second audits the saved study and writes its verification summary. Neither retrains the BERT comparisons.

## Before class

1. Run the short demo once on the exact machine you will use.
2. Have `paper/draft.md`, the random confusion matrix and `selection_manifest.json` ready.
3. Ensure `StudentsPack` is next to the project folder.
4. Keep the full prepared project with its saved artifacts for the demo.
5. Remember that `.venv` and `cache` currently link to `llm-codex-merged`. Bring that folder too, or prepare an independent environment beforehand.

The original README/defense guide still use the older directory name. Use the commands on this page for the final working copy.

## On another Mac

Prepare software before the lesson. With Python 3.14 available, an independent copy without an existing `.venv` link can use:

```sh
python3.14 -m venv llm-final/.venv
llm-final/.venv/bin/python -m pip install -r llm-final/requirements-lock.txt
```

The pinned versions match the tested setup. A virtual environment is machine-specific; copying it is not a reliable installation method.

The two short SVM commands need no neural downloads. A fresh full experiment needs pretrained downloads if they are absent.

## Full experiment reproduction

```sh
llm-final/run_all.sh
```

This reruns the experiment sequence and overwrites generated artifacts and reports. Finished BERT jobs are reused by default. A complete fresh neural run can take roughly one to two hours, depending on machine and downloads.

To retrain the four BERT comparisons explicitly:

```sh
llm-final/run_training_queue.sh --force
llm-final/run_all.sh
```

If a new run changes the winner or its mistakes, review `paper/error_notes.json` before treating the generated paper as current.

## Packaging after the PDF is finished

Example for group 3, with a real final PDF at the path shown:

```sh
llm-final/.venv/bin/python llm-final/src/package_submission.py --group-number 3 --paper-pdf /path/to/3.pdf
```

Replace both the group number and path with your actual values. This creates `llm-final/3.zip`. Open and check it before manually submitting through Fenix. The code inside uses the packaged folder name `llm-codex-merged`.

## A one-minute explanation

“We classify clinical notes into 12 medical specialties. We ask whether word-based or pretrained representations work better, and how metadata and repeated notes change the answer. We compare six model families on the same fitting data, select settings on separate validation data, and evaluate both random and grouped partitions. The dataset has specialty keywords and repeated notes with different labels. Our selected submission system is a fast word/character SVM with keyword and duplicate-exclusion rules. It achieves 95.79% on our internal random test, but much of that comes from metadata. We also tested augmentation and analysed actual errors. Finally, we refit on all usable labelled cases and generate 409 official predictions, whose accuracy we do not know.”

## Practice questions and simple answers

**What is our research question?** Whether representation choice, specialty metadata and repeated notes explain the performance.

**What is the final model?** A field word/character SVM, C=0.3, balanced, using three clinical fields, with fixed keyword and duplicate-exclusion rules.

**Why did we choose it?** It ranked first by the declared validation criteria. Its speed is a practical benefit, rather than the reason we changed the selection.

**Did BERT fail?** It worked, but did not consistently beat the alternatives. Full-system and bare-model selection can choose different variants.

**Why do we need validation and test?** Validation chooses; internal test measures the frozen choice.

**Why is 95.79% not proof of strong clinical understanding?** The keyword rule decides 444 of 523 cases. SVM alone handles only 54 cases.

**Why can repeated notes hurt?** Their label often differs from the fitting copy's label, so recalling that label is wrong.

**What does the memory rule know?** Which labels have already appeared with fitting copies. It excludes them; it does not know the right remaining label.

**How do we avoid augmentation leakage?** Every base case and section donor belongs to that fitting fold.

**Why is the final model unaugmented?** We tested augmentation as required, but validation preferred an unaugmented candidate.

**Why macro-F1?** It gives small specialties the same importance as Surgery.

**What is a confusion matrix?** A count table: rows are gold classes, columns are predictions, diagonal entries are correct.

**What is the official score?** Unknown, because we have no official gold labels.

**What are our limits?** One outer fold per regime, one neural seed, prior exploration, small classes and corpus-specific shortcuts.

## Check your understanding

Try answering without this page, then compare with lessons 2–4:

1. Is “SVM gets 0% on repeated notes” a claim about our final system?
2. Does grouped accuracy of 93.5% use keywords?
3. Can the duplicate rule guarantee the right answer?
4. Can a system using keywords exceed the three-field ambiguity bound?
5. Should we switch to a different model because its internal test score is higher?

**Answers:** 1. No, it describes the stripped bare SVM. 2. Yes. 3. No. 4. Yes, the inputs differ. 5. No, that uses test outcomes for selection.
