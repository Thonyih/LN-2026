# Running the project during the defense

Use your prepared MacBook environment. Open a terminal in the **Main Project** folder containing both `StudentsPack` and `llm-codex-merged`. No environment activation is needed with the explicit Python paths below.

## 1. Demonstrate training and measured internal performance

```sh
llm-codex-merged/.venv/bin/python llm-codex-merged/src/defense_demo.py
```

This fits TF-IDF and the selected SVM again on the frozen 1,672-case fitting partition, then classifies its 523-case internal test partition. It prints accuracy, macro-F1, every class's F1, decision-route results and the three paper errors. It checks the newly computed results against the archived run. Archived predictions are used only for checking, not for producing the demo predictions.

Expected results: **95.79% accuracy, 0.9402 macro-F1 and 0.80 minimum class F1**, followed by `PASS`. The fit/evaluation/demo took **6.0 seconds** in the tested M4 run; terminal startup and machine load can add time. This demo uses labeled course data only and needs no pretrained model downloads or internet access.

## 2. Demonstrate generating the submission output

```sh
llm-codex-merged/.venv/bin/python llm-codex-merged/src/predict.py
```

This fits the frozen winning SVM on all **2,613 valid labeled cases**, applies the keyword and duplicate rules, and regenerates **409 predictions** in `llm-codex-merged/results.txt`. It checks labels and record order, with no header. The tested command took **9.5 seconds**, and its regenerated file was identical to the existing output. It runs offline in the prepared environment.

These predictions have **no available official gold labels**. The 95.79% score belongs to the internal test in step 1; do not present it as official-test accuracy.

## 3. Optional checks if requested

```sh
llm-codex-merged/.venv/bin/python -m unittest discover -s llm-codex-merged/tests -v
llm-codex-merged/.venv/bin/python llm-codex-merged/src/verify.py
```

The first runs six unit tests. The second audits all saved experiments, including donor isolation, grouped boundaries, partial BERT updates, validation-based selection, metrics, source/label ordering and final output integrity. It verifies archived evidence; it does not retrain the experiment grid. Keep the complete artifacts and augmentation data for this audit.

## Prepare before class

- Bring the full `llm-codex-merged` folder and its sibling `StudentsPack`, including your working `.venv`. Run steps 1 and 2 once before the lesson.
- On a different computer, create the environment and install `requirements-lock.txt` before class. Virtual environments are machine-specific; do not assume copying `.venv` to another computer works.
- Have `paper/draft.md`, `artifacts/random_confusion.png`, `artifacts/selection_manifest.json` and `MODEL_EXPLANATIONS.md` available to show and explain.
- The selected final model is an **SVM with rules**. BERT and MiniLM are comparison models. Their weights and neural retraining are not needed for these two short demonstrations.

## Full experiment reproduction

```sh
llm-codex-merged/run_all.sh
```

This reruns analysis, classical tuning, augmentation, evaluation, prediction, reporting and verification. Completed BERT comparisons are reused by default; missing comparisons are trained. A fresh run including all neural training can take roughly **1–2 hours**, depending on caches and machine load. Schedule it before class unless the lecturer explicitly requests it.

To explicitly repeat the four BERT comparisons on the existing fixed partitions, then regenerate the full experiment outputs:

```sh
llm-codex-merged/run_training_queue.sh --force
llm-codex-merged/run_all.sh
```

Pretrained downloads require internet if the local caches are absent. If new runs change the selected system or its mistakes, review `paper/error_notes.json` before generating the paper.

## Explain the main decisions

The winner was chosen on **development macro-F1**, not on its outer test score. The live demo reuses that frozen choice. Its SVM features come from description, procedure name and transcription; the keyword rule examines the separate original keyword field. Duplicate exclusion is a corpus-specific hypothesis that sometimes fails, as source 771 illustrates.

Random splitting permits repeated notes across partitions; grouped splitting tests unseen note groups. Missing-keyword cases are harder, and the high full-system score depends substantially on metadata. Augmentation was generated and evaluated, with donors confined to fitting folds; the final winner does not use augmentation because development selection did not prefer it. Read the paper's actual errors and synthetic inconsistency examples before the defense.
