# Merge status and professor alignment

The merger creates fresh code and artifacts in llm-codex-merged. Neither original run supplies headline scores to the new comparison tables.

## Implemented

- Shared labeled-data loader, canonical label order and source ids.
- Cleaned-note groups and common random/grouped train/validation/test partitions.
- Field-specific word/character SVM, word LR, LSA, frozen chunk-pooled MiniLM, baseline NB.
- ClinicalBERT head/head+tail code with shared validation and corrected partial accumulation.
- Fixed and fitting-learned keyword rules, corpus-specific duplicate exclusion, and separate rule ablations.
- Fold-local section donors with complete lineage, resampling and matched dropout controls.
- Validation-only family/configuration/rule selection, equal fitting sets for outer model comparisons.
- Labeled-data ambiguity bound and paired note-group bootstrap uncertainty.
- Markdown report generation, final ordered prediction path, and verification.
- Submission packaging code and professor-rubric map.

## Fixed experiment scope

The primary evaluation uses one predetermined outer fold per regime and an inner development partition. It is a retrospective train/development/test rerun rather than full nested outer cross-validation. Sparse augmentation diagnostics use three folds on outer training/development cases. BERT head/head+tail each train for up to three epochs with development checkpoint selection.

This scope provides fair shared fitting data on an M4/16 GB, covers at least three models and mandatory augmentation, and avoids treating repeated seed searches as external confirmation. Wider outer-fold and seed replication remains future work. The optional calibrated ensemble, Word2Vec retuning and new back-translation runs are omitted from the primary shortlist to keep the study focused; none is required by the professor.

## Execution status — completed

All six model families and all four ClinicalBERT comparisons completed. The controlled augmentation study, frozen outer evaluation, final all-data SVM refit, ordered predictions (398 since the professor's cleaned test set of 10 Oct 2026; previously 409) and English manuscript are complete. Six integrity unit tests passed. The final audit recomputed all development scores and outer metrics, verified per-class results against saved predictions, checked label/source order, isolated every donor and confirmed disjoint grouped partitions. See artifacts/integrity_verification.json and verification.log.

The random development winner is the clinical-field word/character SVM (C=0.3, balanced weights) with fixed keyword mapping and duplicate exclusion. Its 523-case random outer test has accuracy 95.79%, macro-F1 0.9402 and minimum class F1 0.80. The separately selected grouped system has accuracy 93.50%, macro-F1 0.9141 and minimum F1 0.75. Official performance is unknown; the final choice was not changed after outer scoring.

The required errors are actual random-test mistakes, source records 361, 2582 and 771. Fold-local section recombination and resampling give small descriptive macro-F1 gains; dropout is weaker. Inspected synthetic inconsistencies are recorded in paper/augmentation_review.md.

Operational note: changing the launcher during execution caused a shell error after all four BERT jobs completed. The outputs were intact; a fresh run of the corrected resume launcher passed and confirmed all four completed comparisons (training_queue_resume.log). This did not alter the trained scores.

## Working and final deliverables

The user's requested working paper is English Markdown. The professor's final submission still requires the group's NUM.pdf in the provided template, verified pagination, author contribution percentages and NUM.zip. These user-specific final details are not invented. See PROFESSOR_REQUIREMENTS.md.

Historical input-only official-test inspection is disclosed. The merged pipeline itself reads the official file only in the final prediction stage.
