<!--
WORKING DRAFT for the LaTeX conversion (template: StudentsPack-v2/Project-Template/template.tex).
- Every number comes from ../artifacts/ (saved predictions and metrics); see ../IMPROVEMENTS.md.
- Figures are in ./figures/, references in ./references.bib (BibTeX keys in [brackets] below).
- Before submitting: fill the author line; keep the main text ≤ 3 pages (bibliography excluded);
  appendix ≤ 1 page; delete the template's "Evaluation and tips" section.
- ../paper/draft.md is the auto-generated version of this text (src/report.py); THIS file is the one to edit.
- DECISIONS.md (same folder) = full decision story + likely oral questions; not part of the paper, not zipped.
-->

# Specialty Classification of Clinical Notes: Representations, Metadata and Repeated Notes

**Group XXX** — Author 1 (xx%), Author 2 (xx%), Author 3 (xx%), Author 4 (xx%) ⚠ *mandatory: names and % contribution of each member*

## 1 Introduction

We classify clinical notes into 12 medical specialties from their description, sample name, transcription and
keywords [course2026]. Inspecting the training data showed that a high score does not have to come from reading
the clinical text: the first keyword usually names the specialty, and many notes are verbatim copies of other
notes filed under a different specialty. Our research question is therefore: **How do lexical and pretrained
representations compare when specialty metadata is available versus absent, and when notes are repeated versus
unseen?** We keep model predictions separate from two explicit rules and measure what each contributes.

## 2 Data

### 2.1 Data Analysis

The training file contains 2,616 records. Three records carry note fragments instead of a label and have empty
clinical fields; they are logged and excluded, leaving 2,613. The classes are imbalanced: *Surgery* has 876
cases and *Dermatology* 25 (35:1). 441 cases have no keywords and 30 have no transcription.

Two observations guide the experiments. **(1) Specialty-naming keywords:** 2,172 keyword lists begin with a
specialty name (e.g. `orthopedic`), and it agrees with the supplied label in 2,171 of them. **(2) Repeated
notes with different labels:** 711 cleaned note groups carry conflicting labels; for example, the report "Left
lower extremity venous Doppler ultrasound" appears under both *Cardiovascular-Pulmonary* and *Radiology*. Copies
almost never repeat a label: in 710 of the 712 groups with several rows, every copy has a different specialty.
Consequently, on the labelled data, a system that gives identical clinical text (description, sample name,
transcription) a single label can reach at most 70.3% accuracy; only differing keywords can separate such copies. **(3) Formatting faults:** seven descriptions swallowed
other records (a quoting fault); cleaning cuts them at the first embedded record and normalises the `.,` section
separators, identically for training and test inputs.

### 2.2 Data Augmentation

We augment the four smallest classes (*Dermatology*, *Psychiatry-Psychology*, *Ophthalmology*, *Neurosurgery*)
towards about 150 cases each, adding at most four variants per original note. Three strategies are compared:
**resampling** (duplicating originals), **field dropout** (removing the keywords, the sample name with
probability 0.5, and the transcription with probability 0.35 when a description remains), and **section
recombination**: each headed section of a note (e.g. PROCEDURE, IMPRESSION) is replaced, with probability 0.5,
by the same-header section of another training note of the same specialty. Donors come only from the training
fold, so no held-out text enters training.

## 3 Models

Six model families address the research question. The official **Naive Bayes baseline** uses TF-IDF of the
lowercased, Porter-stemmed description with Multinomial NB (α = 1). **Word TF-IDF + logistic regression (LR)**
tests lexical evidence. The **field word/character support vector machine (SVM)** concatenates per-field word
and character TF-IDF features, capturing phrases and word fragments such as *-ectomy*. **Latent semantic
analysis (LSA) + LR** projects TF-IDF to 128 dimensions. **Frozen MiniLM + LR** [reimers2019, wang2020,
minilmcard] averages embeddings of 220-token chunks over the whole note and concatenates one 384-dimensional
vector per field. **ClinicalBERT** [alsentzer2019] is fully fine-tuned, either on the first 510 word pieces
(head) or on the first 128 and last 382 (head+tail).

Model inputs omit the keywords, remove only the first keyword, or keep them all. Two rules can sit on top of any
model: a **keyword rule** (a first keyword mapped to a specialty, from a fixed list or learned with ≥ 2
observations and ≥ 95% purity), which takes precedence, and a **duplicate rule** (for a verbatim copy of training
notes, exclude the specialties already attached to it and take the best remaining score). The duplicate rule
encodes a regularity of this corpus, not a clinical fact.

## 4 Experimental Setup

**How the design came about.** The design follows from the data. Because the first keyword names the specialty
in 2,171 of 2,172 rows, we removed it from the model input and turned it into a separate rule; otherwise we would
measure the keyword rather than how well a model reads clinical text. Because the same note is often filed under
several specialties, every model is tested twice: on a *random* partition, where copies of a note can fall on
both sides, and on a *grouped* partition, where every test note is new. Because the research question compares
representations, the models range from word counts to a fine-tuned clinical Transformer, and since 70.8% of the
notes exceed ClinicalBERT's 512-token limit, it also reads the end of each note. Section recombination takes
donors only from the training fold and is compared with plain resampling, to separate the effect of new text from
that of more examples. Every choice is made on a validation part and frozen before testing, so the test score is
not optimistic. We submit the validation winner, the SVM, rather than ClinicalBERT, which scored lower (full-system
macro-F1 0.952 vs. 0.969) and needed about 16 minutes of GPU training per run instead of seconds.

**Partitions.** Both regimes are stratified by label; empty notes are grouped by their other clinical fields. The
first outer and first inner fold of seeded five-fold divisions give train/validation/test sizes of 1,672/418/523
per regime, with all classes present, and all models are fitted on the same training rows. Validation ranks
configurations, checkpoints and rules by macro-F1, then minimum class F1, then accuracy. The submitted system is
refitted on **all 2,613 valid training cases**.

**Test set.** `results.txt` contains the predictions for the 398 official test notes; test inputs are used only
for this.

**Metrics.** Accuracy, macro-F1 and per-class precision, recall and F1 [pedregosa2011]; a paired bootstrap over
note groups gives 95% intervals.

### 4.1 Parameters/Hyperparameters

Partition and model seeds are 20261008 and 42. The inverse regularisation strength C is searched in {0.3, 1, 3}
for the SVM and {1, 10} for LR, with ordinary or balanced class weights. SVM word 1–2-grams use min_df = 2 and at
most 40,000 features per field; character 3–5-grams within word boundaries use min_df = 3 and at most 30,000.
Word LR and LSA use word 1–2-grams, min_df = 2, at most 60,000 features. All comparators use sublinear TF and L2
normalisation. Resulting input sizes: 114,257 sparse dimensions for the submitted SVM (about 3,200 non-zero per
note), 60,000 for word LR and 4,126 for the baseline; 128 (LSA) and 1,536 (MiniLM, 4 × 384) dense dimensions.
ClinicalBERT (BERT-base: 12 layers, 108 M parameters, all fine-tuned) reads at most 512 word pieces and classifies
a 768-dimensional [CLS] vector; it uses AdamW (learning rate 3e-5, weight decay 0.01), 10% warm-up, square-root
inverse-frequency class weights, 3 epochs and an effective batch of 16 (micro-batches of 8, partial batches
included). Model revisions are pinned; all runs used an Apple M4 with 16 GB.

## 5 Results

Table 1 compares the validation-selected bare models (first specialty keyword removed; the baseline uses only
the description).

| Model | Random acc. | Random macro-F1 | Grouped acc. | Grouped macro-F1 | Mem. rate |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes baseline | 36.5% | 0.099 | 40.7% | 0.143 | 0.52 |
| Word TF-IDF + LR | 46.7% | 0.502 | 57.2% | 0.594 | 0.72 |
| Field word/character SVM | 41.1% | 0.419 | 59.3% | 0.615 | 1.00 |
| LSA + LR | 51.8% | 0.542 | 54.7% | 0.565 | 0.49 |
| Frozen MiniLM + LR | 50.1% | 0.542 | 58.5% | 0.620 | 0.66 |
| Fine-tuned ClinicalBERT | 53.9% | 0.537 | 57.2% | 0.604 | 0.58 |

*Table 1. Bare models on common partitions; F1 is macro-averaged. Mem. rate: among the 190 random-test notes
that repeat a training note, the share for which the model outputs a specialty that note already carries in the
training data.*

The submitted system is the **field word/character SVM** (C = 0.3, balanced weights, keywords omitted from the
input) with **both rules**: random-test accuracy **95.8%**, macro-F1 **0.940**, minimum class F1 **0.800** (95%
macro-F1 interval [0.897, 0.969]). The separately selected grouped system reaches 93.5% accuracy and 0.914
macro-F1. Table 2 and Figure 1 describe the submitted system on the random test.

| Specialty | Train n | Test n | P | R | F1 |
| --- | --- | --- | --- | --- | --- |
| Cardiovascular-Pulmonary | 299 | 60 | 0.95 | 0.98 | 0.97 |
| Dermatology | 25 | 5 | 0.80 | 0.80 | 0.80 |
| Gastroenterology | 192 | 39 | 1.00 | 0.90 | 0.95 |
| General Medicine | 207 | 41 | 0.81 | 0.95 | 0.88 |
| Neurology | 181 | 36 | 0.97 | 0.92 | 0.94 |
| Neurosurgery | 78 | 16 | 0.89 | 1.00 | 0.94 |
| Obstetrics-Gynecology | 135 | 27 | 1.00 | 0.93 | 0.96 |
| Ophthalmology | 71 | 14 | 1.00 | 1.00 | 1.00 |
| Orthopedic | 292 | 58 | 0.96 | 0.95 | 0.96 |
| Psychiatry-Psychology | 40 | 8 | 1.00 | 0.88 | 0.93 |
| Radiology | 217 | 44 | 0.96 | 1.00 | 0.98 |
| Surgery | 876 | 175 | 0.99 | 0.97 | 0.98 |

*Table 2. Submitted system on the random test: training size, test support, precision (P), recall (R), F1.*

![Submitted system, random test](figures/fig1_submitted_random.png)

*Figure 1. Confusion matrix of the submitted system on the random test (rows: gold; columns: predicted; numbers:
counts; colour: share of the row). Cardio = Cardiovascular-Pulmonary, Derm = Dermatology, Gastro =
Gastroenterology, GenMed = General Medicine, Neuro = Neurology, NSurg = Neurosurgery, ObGyn =
Obstetrics-Gynecology, Ophth = Ophthalmology, Ortho = Orthopedic, Psych = Psychiatry-Psychology, Radio =
Radiology, Surg = Surgery.*

## 6 Discussion

**Representations.** LSA + LR and frozen MiniLM + LR reach the highest bare random scores (within 0.001).
Fine-tuning does not consistently lead: ClinicalBERT reaches macro-F1 0.537 (random) and 0.604 (grouped), against
0.419 and 0.615 for the SVM.

**Repeated versus unseen notes.** A repeated note is the *hardest* case, not the easiest. 190 of the 523
random-test notes repeat a training note, and for 189 of them the gold specialty is one the note does not yet
carry. Bare models reproduce the training copy's label for 49–100% of these notes (Table 1); the SVM does so for
all of them (accuracy 0.0% on repeated vs. 64.6% on unseen notes). This explains why every model scores lower on
the random than on the grouped partition. Memorisation and generalisation are separate abilities: the SVM
memorises most and is still among the best on unseen notes.

**Metadata.** The keywords carry most of the signal (Table A3): on the random partition, SVM macro-F1 rises from
0.399 without keywords to 0.885 with them; the keyword rule alone gives 0.896, the duplicate rule alone 0.702 and
both rules 0.945. Removing only the first keyword (0.419) still leaves other keyword evidence. In the submitted system, 444 random-test notes are decided by the keyword
rule (100.0% correct), 25 by the duplicate rule (76.0%) and 54 by the model (70.4%). Transfer to hospital data without these clues is
unproven.

**Augmentation.** Effects are small (Table A1): macro-F1 changes by +0.004 / +0.008 (random / grouped) for
resampling, +0.003 / +0.006 for section recombination and −0.011 / −0.016 for dropout. Since recombination is no
better than plain resampling, its small benefit probably comes from more minority examples rather than new text; a variant of
Ophthalmology note 1110 even pairs an eyelid-repair title with lens-removal text from donor 1143.

**Errors.** The largest residual confusion is Gastroenterology → General Medicine (4 of 22 errors). Across models,
the dominant bare error is a *Surgery* note predicted as an organ specialty (129 of 224 ClinicalBERT and 95 of 213
SVM grouped-test errors; Fig. A1), where the duplication described in Section 2.1 makes both labels defensible:
bare ClinicalBERT labels only 45 of 175 unseen *Surgery* notes as *Surgery*. Four
actual mistakes of the submitted system:

- **Source 361:** "I was asked to see the patient for C. diff colitis." Predicted *General Medicine*, gold
  *Gastroenterology*. Keywords are absent; the long admission history and general consultation language may
  outweigh the gastrointestinal cue. This is the most frequent confusion pair (four cases).
- **Source 2582:** "Primary right shoulder arthroscopic rotator cuff repair with subacromial decompression."
  Predicted *Orthopedic*, gold *Surgery*. The procedure belongs to both specialties; anatomical detail cannot
  identify which of the two labels this corpus assigned.
- **Source 771:** "Left third digit numbness and wrist pain." Predicted *Radiology*, gold *Neurology*. A training
  copy is labelled *Orthopedic*; the duplicate rule correctly excludes it but then picks *Radiology*, cued by the
  sample name "MRI C3 - Cord Compression". Excluding a known label does not identify the correct one.
- **Source 1480:** "69-year-old white woman with Huntington disease, who presents with the third suicide attempt
  in the past two months." Predicted *Neurology*, gold *Psychiatry-Psychology*. The note is a psychiatric consult
  ("suicide" 11 times, "depression" 10 times) about a neurological disease, and the corpus files it twice:
  source 858 as *Neurology* and source 1480 as *Psychiatry-Psychology*. Any single prediction is wrong for one
  copy.

These readings do not prove that the supplied labels are wrong, and exact grouping misses near-duplicates.

## 7 Future Work

Repeat the comparison across outer folds and seeds; test clinically coherent augmentation, whole-note encoding
and multi-label targets for overlapping specialties; evaluate a calibrated lexical/neural ensemble if validation
evidence supports it.

## Bibliography

*(BibTeX in `references.bib`; does not count towards the 3 pages)*

- [reimers2019] Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP*.
- [wang2020] Wang, W. et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS*.
- [minilmcard] Sentence Transformers. all-MiniLM-L6-v2 model card (revision 1110a24).
- [alsentzer2019] Alsentzer, E. et al. (2019). Publicly Available Clinical BERT Embeddings. *Clinical NLP Workshop* — checkpoint `emilyalsentzer/Bio_ClinicalBERT` (revision d5892b3).
- [pedregosa2011] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.
- [course2026] Course project description and template (2026); corpus adapted from Kaggle (MTSamples). No external labelled cases were added.

Software: Python, NumPy, pandas, SciPy, scikit-learn, NLTK, Matplotlib, PyTorch, Transformers, Sentence Transformers.

**Declaration of LLM use.** ⚠ *Adapt to what the group actually did.* Claude (Anthropic) assisted the earlier
design, code and drafting. OpenAI Codex reviewed both runs, merged and corrected the code, executed the
experiments, inspected data and errors, and drafted the paper; Claude then added the memorisation analysis and the
fourth error example. All metrics come from saved predictions; neither assistant
supplied case predictions or generated synthetic clinical content.

## Appendix A: Extra Figures and Tables

*Table A1. Augmentation diagnostics (three-fold pooled predictions on outer training/validation cases).*

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

*Table A2. Memorisation on the random test (bare models).*

| Model | Mem. rate | Acc. repeated | Acc. unseen |
| --- | --- | --- | --- |
| Naive Bayes baseline | 0.52 | 0.332 | 0.384 |
| Word TF-IDF + LR | 0.72 | 0.247 | 0.592 |
| Field word/character SVM | 1.00 | 0.000 | 0.646 |
| LSA + LR | 0.49 | 0.426 | 0.571 |
| Frozen MiniLM + LR | 0.66 | 0.279 | 0.628 |
| Fine-tuned ClinicalBERT | 0.58 | 0.353 | 0.646 |

*Table A3. Metadata and rule ablations for the SVM (macro-F1). Rules read the original keywords even when the
model input omits them.*

| SVM input / rules | Random | Grouped |
| --- | --- | --- |
| No keywords | 0.399 | 0.610 |
| First specialty keyword removed | 0.419 | 0.615 |
| All keywords | 0.885 | 0.911 |
| No keywords + keyword rule | 0.896 | 0.920 |
| No keywords + duplicate rule | 0.702 | 0.610 |
| No keywords + both rules | 0.945 | 0.920 |

![Bare ClinicalBERT, grouped test](figures/figA1_bert_grouped.png)

*Figure A1. Confusion matrix of bare ClinicalBERT (no rules) on the grouped test, i.e. unseen notes (same
conventions as Fig. 1). Outlined cells: Surgery notes predicted as an organ specialty (129 of 224 errors).*
