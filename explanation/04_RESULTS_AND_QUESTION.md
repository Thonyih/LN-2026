# 4. What do the results tell us?

## Learn these measures first

| Measure | Simple meaning |
| --- | --- |
| Accuracy | Fraction of cases with the supplied answer |
| Precision for a class | Of the cases we called that class, how many were right? |
| Recall for a class | Of the actual cases in that class, how many did we find? |
| F1 for a class | One score combining precision and recall |
| Macro-F1 | Average the 12 class F1 scores, giving every class equal importance |
| Minimum class F1 | The worst class F1 |
| Support / n | Number of actual examples of a class |

If ten cases are truly Dermatology and we predict Dermatology on ten cases, eight correctly, precision and recall are both 0.80, so F1 is 0.80.

F1 = 2 × precision × recall / (precision + recall), using zero when appropriate. It is not the fraction of all correct cases.

Macro-F1 matters because a small class such as Dermatology should not disappear behind the many Surgery cases.

## Compare models without the rules

These are the paper's **bare models**: no extra decision rules. The recognized first specialty keyword is removed; other keywords remain. The baseline uses description only.

| Model | Random accuracy | Random macro-F1 | Grouped accuracy | Grouped macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 36.5% | 0.099 | 40.7% | 0.143 |
| Word TF-IDF + LR | 46.7% | 0.502 | 57.2% | 0.594 |
| Field word/character SVM | 41.1% | 0.419 | 59.3% | 0.615 |
| LSA + LR | 51.8% | 0.542 | 54.7% | 0.565 |
| Frozen MiniLM + LR | 50.1% | 0.542 | 58.5% | 0.620 |
| ClinicalBERT | 53.9% | 0.537 | 57.2% | 0.604 |

For random bare macro-F1, LSA and MiniLM are almost tied at the top. For grouped bare macro-F1, MiniLM is highest and the SVM is close. BERT does not consistently win.

This is a finding about the saved experiment. It does not prove that one family always beats another.

## The full systems

| Selected system | Internal cases | Accuracy | Macro-F1 | Worst class F1 |
| --- | ---: | ---: | ---: | ---: |
| Random winner: clinical SVM + both rules | 523 | 95.79% | 0.9402 | 0.80 |
| Separately chosen grouped winner: stripped SVM + keyword rule | 523 | 93.50% | 0.9141 | 0.75 |

The random winner gets **501 right and 22 wrong**.

The grouped score also uses keywords. It is not evidence that clinical-text understanding alone achieves 93.5%.

We selected the random submission winner on **validation**, even though a different full model may have a better outer-test score. Switching after seeing test results would use the exam to choose the learner.

## Where does 95.79% come from?

| Final decision route | Cases | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Keyword | 444 | 444 | 100% |
| Repeated-note exclusion + SVM | 25 | 19 | 76% |
| SVM alone | 54 | 38 | 70.37% |
| Total | 523 | 501 | 95.79% |

The high total depends heavily on the metadata. It does not mean that the SVM reads every note correctly.

## Why can repeated notes be harder?

190 random-test notes repeat a fitting note group. For 189 of them, the correct specialty is absent from that group's fitting labels.

The stripped **bare** SVM predicts a fitting label on all 190 repeated cases and gets none right; on the other 333 cases its accuracy is 64.6%.

This result describes that particular bare SVM. The submitted clinical SVM with rules is a different configuration. Do not apply the “0%” claim to the final system.

Random and grouped partitions also have different test cases, so their score difference is not a perfectly isolated measurement of one cause.

## What did augmentation show?

In the three-fold diagnostic study:

- Resampling gives small macro-F1 gains: about +0.004 random and +0.008 grouped.
- Section recombination gives small gains: about +0.003 and +0.006.
- Dropout lowers macro-F1: about −0.011 and −0.016.

These use settings already chosen on development data. They are descriptive diagnostics, rather than another untouched final test. The conclusion is modest: repeating small-class examples helped slightly, and more elaborate synthetic text did not clearly help more.

## Four errors you should be able to explain

| Source | Supplied answer | Our answer | Plain explanation |
| --- | --- | --- | --- |
| 361 | Gastroenterology | General Medicine | A C. diff consultation is gastrointestinal, but the long general history may distract the model |
| 2582 | Surgery | Orthopedic | A shoulder repair strongly suggests Orthopedic, while the dataset uses the broader Surgery category |
| 771 | Neurology | Radiology | A fitting copy has Orthopedic; excluding it still leaves an imaging-related wrong choice |
| 1480 | Psychiatry-Psychology | Neurology | A psychiatric consult about Huntington disease mixes mental-health and neurological cues |

These are actual held-out mistakes. A plausible interpretation does not prove that the dataset label is wrong.

## How to read the confusion matrix

Open [the random matrix](../artifacts/random_confusion.png). Rows are supplied answers; columns are predictions. The diagonal contains correct predictions. An off-diagonal number counts a specific kind of mistake.

For example, row Gastro and column GenMed shows Gastroenterology cases predicted as General Medicine. This pair accounts for four of the final system's 22 errors.

## The answer to our question

**Representation:** pretrained models are useful, but fine-tuning is not consistently better than lexical models here.

**Metadata:** specialty keywords explain much of the full-system performance. Removing only the first keyword does not remove all keyword information.

**Repetition:** copying a fitting label often fails because copies tend to carry another specialty. Exclusion can help, but it cannot determine the correct remaining answer.

**Practical choice:** validation chose the fast SVM with rules for submission. Its speed helps us reproduce the work on the M4.

## The limits of the answer

Only one outer fold per regime and one neural seed were used. Tiny classes make F1 unstable: the random test has only five Dermatology cases. Bootstrap intervals describe variation in these saved note groups, not every possible retraining or past selection decision.

The 70.3% ambiguity bound concerns identical cleaned **three-field clinical inputs** in this labelled corpus. It does not limit a system allowed to use differing keywords.

Official test accuracy is unknown. Success on this corpus also does not establish success on hospital data without these shortcuts.
