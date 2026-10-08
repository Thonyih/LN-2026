# Notes for understanding and discussing the project

**TF-IDF:** documents become word or character-fragment vectors. TF weights repeated terms within a case; IDF downweights terms common across fitting cases. IDF is fitted on fitting documents only.

**Naive Bayes:** estimates class-specific feature frequencies and combines their evidence under a conditional-independence assumption. This reproduces the professor's weak description-only reference.

**Linear SVM:** learns a linear boundary for each class. Word features capture clinical terms and phrases; character fragments help with morphology and spelling. Balanced class weights increase the influence of minority cases. C controls regularization: higher C penalizes fitting errors more strongly.

**Logistic regression:** learns linear class scores and normalized class probabilities. LSA first compresses TF-IDF to 128 latent dimensions; its projection is fitted on fitting documents.

**Frozen MiniLM:** a pretrained language encoder maps each 220-token chunk to a 384-dimensional vector. Chunk vectors are normalized, averaged and normalized per field; four fields are concatenated. Only the logistic classifier is trained on course labels. Empty fields use the encoder's empty-text embedding.

**ClinicalBERT:** begins with clinically pretrained BERT weights and adjusts all parameters for 12-way classification. Head-only keeps the first 510 word pieces plus special tokens. Head+tail keeps 128 beginning and 382 ending word pieces. Effective batches contain up to 16 cases, computed as microbatches of up to 8. Summed weighted losses are divided by the total class-weight mass of the effective batch, including the last partial batch. Square-root inverse-frequency weights reduce class imbalance without the full inverse-frequency correction.

**Keyword rule:** some lists begin with the specialty. A fixed mapping or a mapping learned with minimum count/purity checks can use this input directly. Report this benefit separately from clinical comprehension.

**Duplicate exclusion:** under this corpus's almost-unique (note,label) pattern, a repeated note's current label may differ from labels already attached to fitting copies. The rule excludes those known labels before choosing the remaining model score. This is a corpus hypothesis and can fail if the same (note,label) legitimately repeats.

**Random versus grouped splits:** random splits model repeated-note conditions. Grouped splits move every copy of a normalized note together and ask about unseen notes. These measure different questions; neither regime is an automatic prediction of official-test performance.

**Augmentation:** resampling repeats minority cases; dropout omits auxiliary fields; recombination replaces a section with the same-header section from another fitting case of the same specialty. Matching sampling counts separates dropout's field changes from resampling. Recombination may create contradictory cases, so inspect examples. All base and donor ids must stay in the fitting partition.

**Metrics:** accuracy counts correct cases. Macro-F1 gives each specialty equal weight. Minimum class F1 tracks the worst class and the professor's per-class requirement. Tiny class supports make F1 volatile. Group bootstrap intervals treat note groups as the sampling unit and are descriptive after retrospective exploration.

**Ambiguity bound:** identical clinical input tuples with different labels cannot all be classified correctly by a deterministic model seeing only those tuples. Sum the most common label count per identical tuple and divide by cases. This diagnostic does not apply unchanged to inputs with different specialty keywords.
