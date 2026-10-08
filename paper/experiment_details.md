# Experiment details and supplementary results

## Selected configurations

| Regime | Comparison | Family | Condition | Parameters / variant | Rules |
| --- | --- | --- | --- | --- | --- |
| random | bare | Naive Bayes baseline | clinical | {} | none |
| random | bare | Word TF-IDF + LR | stripped | {"C": 1.0, "balanced": true} | none |
| random | bare | Field word/character SVM | stripped | {"C": 0.3, "balanced": true} | none |
| random | bare | LSA + LR | stripped | {"C": 1.0, "balanced": true} | none |
| random | bare | Frozen MiniLM + LR | stripped | {"C": 1.0, "balanced": true} | none |
| random | bare | Fine-tuned ClinicalBERT | stripped | {"variant": "headtail", "epochs": 3} | none |
| random | full | Naive Bayes baseline | clinical | {} | both |
| random | full | Word TF-IDF + LR | all | {"C": 10.0, "balanced": true} | both |
| random | full | Field word/character SVM | clinical | {"C": 0.3, "balanced": true} | both |
| random | full | LSA + LR | stripped | {"C": 10.0, "balanced": true} | both |
| random | full | Frozen MiniLM + LR | clinical | {"C": 1.0, "balanced": true} | both |
| random | full | Fine-tuned ClinicalBERT | stripped | {"variant": "head", "epochs": 3} | both |
| random | selected | Field word/character SVM | clinical | {"C": 0.3, "balanced": true} | both |
| grouped | bare | Naive Bayes baseline | clinical | {} | none |
| grouped | bare | Word TF-IDF + LR | stripped | {"C": 10.0, "balanced": true} | none |
| grouped | bare | Field word/character SVM | stripped | {"C": 0.3, "balanced": true} | none |
| grouped | bare | LSA + LR | stripped | {"C": 10.0, "balanced": true} | none |
| grouped | bare | Frozen MiniLM + LR | stripped | {"C": 1.0, "balanced": true} | none |
| grouped | bare | Fine-tuned ClinicalBERT | stripped | {"variant": "headtail", "epochs": 3} | none |
| grouped | full | Naive Bayes baseline | clinical | {} | keyword |
| grouped | full | Word TF-IDF + LR | clinical | {"C": 10.0, "balanced": true} | keyword |
| grouped | full | Field word/character SVM | stripped | {"C": 1.0, "balanced": false} | keyword |
| grouped | full | LSA + LR | stripped | {"C": 10.0, "balanced": true} | keyword |
| grouped | full | Frozen MiniLM + LR | clinical | {"C": 10.0, "balanced": true} | keyword |
| grouped | full | Fine-tuned ClinicalBERT | stripped | {"variant": "headtail", "epochs": 3} | keyword |
| grouped | selected | Field word/character SVM | stripped | {"C": 1.0, "balanced": false} | keyword |

## BERT variant selection on development data

| Regime | Variant | Selected epoch | Bare validation F1 | Best rule policy | Full validation F1 |
| --- | --- | --- | --- | --- | --- |
| random | head | 3 | 0.566 | both | 0.952 |
| random | headtail | 3 | 0.587 | both | 0.944 |
| grouped | head | 3 | 0.621 | keyword | 0.926 |
| grouped | headtail | 3 | 0.641 | keyword | 0.945 |

Checkpoint epochs are selected by bare development performance. Bare and full-system comparisons can select different variants; outer scores do not change those choices.


## Augmentation diagnostics

| Regime | Augmentation | Acc. | Macro-F1 | Minority mean F1 |
| --- | --- | --- | --- | --- |
| random | none | 43.2% | 0.441 | 0.440 |
| random | resample | 43.2% | 0.445 | 0.451 |
| random | dropout | 43.2% | 0.430 | 0.405 |
| random | sections | 43.0% | 0.444 | 0.450 |
| grouped | none | 59.3% | 0.608 | 0.618 |
| grouped | resample | 59.5% | 0.616 | 0.638 |
| grouped | dropout | 58.9% | 0.592 | 0.566 |
| grouped | sections | 59.4% | 0.614 | 0.633 |

## Minority-class augmentation results

| Regime | Augmentation | Derm F1 | Psych F1 | Ophth F1 | Neurosurg F1 | Minimum F1 |
| --- | --- | --- | --- | --- | --- | --- |
| random | none | 0.353 | 0.780 | 0.452 | 0.174 | 0.174 |
| random | resample | 0.353 | 0.820 | 0.459 | 0.171 | 0.171 |
| random | dropout | 0.303 | 0.800 | 0.424 | 0.095 | 0.095 |
| random | sections | 0.353 | 0.820 | 0.459 | 0.169 | 0.169 |
| grouped | none | 0.457 | 0.824 | 0.746 | 0.443 | 0.443 |
| grouped | resample | 0.514 | 0.848 | 0.738 | 0.452 | 0.452 |
| grouped | dropout | 0.457 | 0.825 | 0.672 | 0.311 | 0.311 |
| grouped | sections | 0.471 | 0.879 | 0.738 | 0.447 | 0.447 |

## Rules on fixed development-selected family candidates

| Regime | Family | Fixed base candidate | Model F1 | +Keyword F1 | +Duplicate F1 | +Both F1 |
| --- | --- | --- | --- | --- | --- | --- |
| random | Naive Bayes baseline | baseline_clinical | 0.099 | 0.871 | 0.284 | 0.884 |
| random | Word TF-IDF + LR | word_lr_all_10.0_True | 0.497 | 0.892 | 0.785 | 0.949 |
| random | Field word/character SVM | wordchar_svm_clinical_0.3_True | 0.415 | 0.897 | 0.706 | 0.940 |
| random | LSA + LR | lsa_stripped_10.0_True | 0.535 | 0.906 | 0.692 | 0.940 |
| random | Frozen MiniLM + LR | minilm_lr_clinical_1.0_True | 0.538 | 0.909 | 0.704 | 0.946 |
| random | Fine-tuned ClinicalBERT | clinicalbert_head | 0.562 | 0.919 | 0.729 | 0.957 |
| grouped | Naive Bayes baseline | baseline_clinical | 0.143 | 0.859 | 0.143 | 0.859 |
| grouped | Word TF-IDF + LR | word_lr_clinical_10.0_True | 0.590 | 0.918 | 0.590 | 0.918 |
| grouped | Field word/character SVM | wordchar_svm_stripped_1.0_False | 0.536 | 0.914 | 0.536 | 0.914 |
| grouped | LSA + LR | lsa_stripped_10.0_True | 0.565 | 0.916 | 0.565 | 0.916 |
| grouped | Frozen MiniLM + LR | minilm_lr_clinical_10.0_True | 0.623 | 0.936 | 0.623 | 0.936 |
| grouped | Fine-tuned ClinicalBERT | clinicalbert_headtail | 0.604 | 0.936 | 0.604 | 0.936 |

Each row keeps the representation and fitted model fixed. Family candidates were selected using full-system development performance; model-only columns need not match the separately selected bare models. Grouped testing has no fitting-note overlap, so the duplicate route should contribute nothing.


## Paired BERT minus SVM group bootstrap

```json
{
  "random": {
    "accuracy_95_interval": [
      0.08741396761133603,
      0.16604143199098848
    ],
    "macro_f1_95_interval": [
      0.07836473843003103,
      0.16868777051938016
    ],
    "replicates": 500,
    "unit": "normalized note group",
    "interpretation": "Conditional descriptive uncertainty; does not account for prior exploration or model-selection variability.",
    "point_difference": {
      "accuracy": 0.12810707456978965,
      "macro_f1": 0.11770361395408874
    }
  },
  "grouped": {
    "accuracy_95_interval": [
      -0.04645225821758262,
      0.007444599057093657
    ],
    "macro_f1_95_interval": [
      -0.0716117942391353,
      0.05270772330906525
    ],
    "replicates": 500,
    "unit": "normalized note group",
    "interpretation": "Conditional descriptive uncertainty; does not account for prior exploration or model-selection variability.",
    "point_difference": {
      "accuracy": -0.021032504780114647,
      "macro_f1": -0.011015929705633587
    }
  }
}
```

## Synthetic examples

- random/dropout: base 295, donors []; description:  A simple note on Acne Vulgaris.; transcription excerpt: ACNE VULGARIS,commonly referred to as just acne, is a chronic inflammation of the skin that occurs most often during adolescence but can occur off and on throughout life.  The skin eruptions most often appear on the face, chest, back and upper arms and are more common in males than females. SIGNS AND SYMPTOMS:*  Blackheads the size of a pinhead.,*  Whiteheads similar to blackhe
- random/sections: base 295, donors [298]; description:  A simple note on Acne Vulgaris.; transcription excerpt: ACNE VULGARIS,commonly referred to as just acne, is a chronic inflammation of the skin that occurs most often during adolescence but can occur off and on throughout life.  The skin eruptions most often appear on the face, chest, back and upper arms and are more common in males than females. SIGNS AND SYMPTOMS: *  Blackheads the size of a pinhead.,*  Whiteheads similar to blackh
- grouped/dropout: base 295, donors []; description:  A simple note on Acne Vulgaris.; transcription excerpt: ACNE VULGARIS,commonly referred to as just acne, is a chronic inflammation of the skin that occurs most often during adolescence but can occur off and on throughout life.  The skin eruptions most often appear on the face, chest, back and upper arms and are more common in males than females. SIGNS AND SYMPTOMS:*  Blackheads the size of a pinhead.,*  Whiteheads similar to blackhe
- grouped/sections: base 295, donors [296]; description:  A simple note on Acne Vulgaris.; transcription excerpt: ACNE VULGARIS,commonly referred to as just acne, is a chronic inflammation of the skin that occurs most often during adolescence but can occur off and on throughout life.  The skin eruptions most often appear on the face, chest, back and upper arms and are more common in males than females. SIGNS AND SYMPTOMS: *  Blackheads the size of a pinhead.,*  Whiteheads similar to blackh

## Interpretation and source lineage

Full donor ids and fold indices are preserved in data/augmented. Shared specialty cannot ensure coherent histories; section recombination may mix ages, medications and diagnoses. Dropout changes field availability rather than clinical assertions. These details complement the main paper and do not replace its required argument.

See [augmentation_review.md](augmentation_review.md) for inspected before/after examples with specific mismatched sections and donor ids.
