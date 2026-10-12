# Decision log — the story behind the project (for the oral; not part of the paper)

The paper tells the short version (Section 4, "How the design came about"). This is the longer version, the way
you would tell it to someone: what we did, in what order, and why. Every number comes from the saved results.

## 1. How the project unfolded

**It started with looking at the data.** Before building any model, we read through `train.csv`, and three things
jumped out. First, the keywords almost give the answer away: in 2,171 of 2,172 rows, the first keyword simply *is*
the specialty ("orthopedic, ORIF, fracture…"). Second, many notes appear more than once, and each copy has a
*different* specialty: a leg venous Doppler report is filed under both Cardiovascular-Pulmonary and Radiology,
for example. In 710 of 712 such
groups, the copies never share a label. Third, the file had some damage: three rows had clinical text where the
label should be, and seven descriptions had swallowed other records, most likely because the file passed through a
spreadsheet that cuts cells at 32,767 characters.

That changed the question we wanted to answer. Getting a high score clearly did not require understanding the
notes, so the interesting question became: *how much of the score comes from the clinical text, and how much from
the keywords and the repeated notes?*

**Our first attempt (`llm-claude`)** tested every model in two ways: on a random split, where copies of a note
can be on both sides, and on a grouped split, where every test note is new. That led to the most surprising
finding of the project: a copy of a note the model has already seen is the *hardest* case, not the easiest,
because the copy almost always carries a different label. A model that "remembers" the note gives exactly the
wrong answer.

**A second, independent attempt (`llm-codex`)** was built separately, as a cross-check. It found the keyword
shortcut too, but it missed the repeated-note effect because it only used grouped splits. It then reviewed our
first attempt and found real mistakes: our best-looking augmentation was secretly borrowing text from test
notes, the paper quoted numbers from the official test file (which the rules do not allow), we had picked our
model using a test split, and two details of the ClinicalBERT training were off.

**So we started over, cleanly (`llm-codex-merged`, then `llm-final`).** We kept what worked from both attempts and
fixed what did not: a proper train / validation / test split, every choice made on validation and frozen before
looking at the test, augmentation built only from training notes, and the corrected BERT training. The model that
won on validation, a word-and-character SVM with the two rules, became our submission, refitted on all 2,613
usable training notes.

**Then the professor released a cleaned test set (10 October).** It is the old file with the 11 broken rows
removed. We switched to it and checked line by line that our predictions for the remaining 398 notes did not
change.

## 2. Decisions, alternatives and reasons

| # | Decision | Alternatives considered | Why we chose it | Evidence |
|---|---|---|---|---|
| D1 | Remove the specialty keyword from model input; use a separate keyword rule | Keep keywords as normal features; drop keywords entirely | Kept in, the model learns "orthopedic → Orthopedic" and we cannot see what it learns from text; dropped entirely, we lose a legitimate given field | First keyword = label in 2,171/2,172 rows; SVM macro-F1 0.399 without keywords vs 0.885 with (Table A3) |
| D2 | Two partitions: random and grouped | One random split (usual); one grouped split (Codex's first run) | Random alone mixes memorisation with generalisation; grouped alone hides the case that dominates a random test | 190/523 random-test notes repeat a training note; 189 have a new label; SVM gets 0% of them right without the duplicate rule |
| D3 | Six families: NB baseline, word LR, word+char SVM, LSA, frozen MiniLM, fine-tuned ClinicalBERT (head and head+tail) | Only classical models; only BERT; Word2Vec; back-translation-heavy setups | The research question compares representations from the lectures (T5 sparse, T7 dense/pre-trained); head+tail because notes are long | 70.8% of notes exceed 512 tokens (median 740) |
| D4 | Augmentation compared as resampling vs field dropout vs section recombination, all generated inside each fold | EDA (WordNet synonyms), back-translation (tried in the first run) | Resampling is the control: it shows whether a gain comes from *more* minority examples or from *new text*. EDA produced wrong senses ("upper arms" → "upper berth"); back-translation took ~1.5 h and did not help | First recombination version leaked held-out sections and looked best; fold-local, it is no better than resampling (Table A1) |
| D5 | Choose everything on a validation part; freeze before testing; refit on all training data | Choose on the test split (first run); full nested cross-validation | Choosing on the test split makes the test score optimistic; nested CV would cost ~5× the BERT GPU time | Professor: "you can use the entire training dataset" → final refit on all 2,613 valid rows |
| D6 | Submit the SVM (+ both rules), not ClinicalBERT | Submit the "most advanced" model | Validation picks the winner, not reputation; BERT scored lower and costs far more | Random validation full-system macro-F1: SVM 0.969, MiniLM 0.958, BERT 0.952; BERT ≈ 16 min per run vs seconds |
| D7 | Keep the duplicate rule, disclosed | Drop it as a "trick" | It uses only training labels and given test *text*; it encodes a measured corpus regularity; the paper reports every model with and without it | Without it, repeated notes are almost always wrong; with it, 76% of those routed to it are right |
| D8 | Drop the 3 label-less rows; cut mega-rows at the first swallowed record | Try to repair them | They have no valid label (cannot be used); cutting is applied identically to train and test | The professor's cleaned test set removed exactly the equivalent broken test rows |
| D9 | `results.txt` in the official format: one label per line, no header, CRLF, no trailing newline | Default Unix format (LF + final newline) | The teacher compares it line by line with an answer file made like the given CSVs | `src/verify.py` checks it |

## 3. Things we deliberately did NOT do (and why)

| Not done | Why | Where it appears |
|---|---|---|
| Full 5-fold (nested) cross-validation | ~5× the GPU time on a 16 GB laptop; one pre-registered split instead | Limitation + Future Work |
| Adapters / LoRA for ClinicalBERT | Similar accuracy expected; would not change the submitted SVM; time | Could be one Future Work line |
| Model pruning | Not needed at this size (BERT-base, 108 M parameters); only vocabulary filtering (min_df, feature caps) | Parameters section |
| Ensemble BERT + SVM | No untouched validation set left to choose it fairly | Future Work |
| Head+tail BERT as the submission | It looked best on a *test* split in the first run; choosing it there would be tuning on test | Discussion (first run) |
| Using the original Kaggle dataset | It contains the labels of these exact notes: that would be cheating | Never used |

## 4. Likely oral questions — short answers

- **"Isn't the keyword rule cheating?"** No: keywords are a given input field, the mapping is learned from
  training data, and every model is reported with and without it. It would not work in a real hospital, and the
  paper says so.
- **"Why is accuracy on repeated notes 0% for the SVM?"** Because the copy in training has a *different* label
  (189 of 190 cases); a model that recognises the copy repeats the old label.
- **"Why not ClinicalBERT if it is the most advanced?"** It was not better on validation (0.952 vs 0.969) and costs
  ~16 minutes per run vs seconds.
- **"What went wrong along the way?"** The first augmentation leaked held-out sections and looked helpful; built
  fold-locally, the gain disappeared. We kept the honest result.
- **"How many folds?"** One pre-defined split from 5-fold divisions (64/16/20%); the augmentation study uses full
  3-fold cross-validation.
- **"What changed with the new test set?"** 11 broken rows removed; our predictions for the other 398 are
  identical; `results.txt` now has 398 lines.
