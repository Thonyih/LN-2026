# 1. What are we doing?

Imagine a case that says “right shoulder repair.” Our program must choose a specialty such as **Orthopedic** or **Surgery**. The dataset supplies one expected answer per case, even when several specialties seem reasonable.

This is **classification**: choose one category from a fixed list. We predict a specialty, rather than generating a diagnosis or treatment.

## The 12 possible answers

| Dataset label | Everyday meaning |
| --- | --- |
| Cardiovascular-Pulmonary | Heart and lungs |
| Dermatology | Skin |
| Gastroenterology | Digestive system |
| General Medicine | Broad medical care |
| Neurology | Nervous system |
| Neurosurgery | Surgery involving the nervous system |
| Obstetrics-Gynecology | Pregnancy and reproductive health |
| Ophthalmology | Eyes |
| Orthopedic | Bones, joints and related structures |
| Psychiatry-Psychology | Mental health |
| Radiology | Medical imaging |
| Surgery | Broad surgical category |

Those are simplified meanings. The exact spelling in the first column is what our output must contain.

## The task and the research question are different

**The task:** produce a specialty for each official case.

**The research question:** “How do lexical and pretrained representations compare when specialty metadata is available versus absent, and when notes are repeated versus unseen?”

Translate that into three questions:

1. Are word/character patterns or pretrained language representations more useful?
2. How much do the supplied keywords help?
3. What happens when we encounter a note already represented in fitting data?

We use several models and controlled comparisons to answer these. A single high score would not explain why the system works.

## What the professor expects

The assignment and template in `StudentsPack` are the authority.

| Requirement | How we address it |
| --- | --- |
| At least three models | We compare six families |
| An explicit research question | The question above appears in the Introduction |
| At least two concrete data observations | Specialty keywords and repeated notes with different labels |
| Mandatory augmentation | Resampling, field dropout and section recombination |
| Explain evaluation and settings | Fixed partitions, metrics, parameters and model revisions |
| Results by specialty and a confusion matrix | Paper Table 2 and Figure 1 |
| At least three actual errors, with interpretation | Four selected-system mistakes in the paper |
| Explain what the results say about the question | Discussion separates models, metadata and repetition |
| Cite sources and declare LLM assistance | Bibliography and LLM declaration |
| Be able to reproduce and explain the work | Tested short demo and these study notes |

## What earns the grade?

The paper is worth **16 points**. The official evaluation is worth **4 points**: two for accuracy above 49%, and two more if additionally every class has F1 of at least 25%.

Our internal scores are promising, but they do not guarantee those four official points.

The final deliverables are `NUM.pdf`, code/added data and `results.txt` inside `NUM.zip`. NUM is the group number. The PDF has at most three main pages; bibliography is excluded from that limit and the appendix has at most one page. Author contribution percentages are required. The deadline is 16 October 2026, 23:59, through Fenix.

## Say it in one sentence

“We classify medical notes into specialties and test whether our success comes from the text representation, specialty keywords, or the unusual repeated-note structure.”
