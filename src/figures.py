"""Paper figures (from saved predictions only; no retraining, no official-test data).

Figure 1  : submitted system (SVM + keyword + duplicate rules) on the random held-out set -> required confusion matrix
Figure A1 : bare ClinicalBERT (no rules) on the grouped held-out set, i.e. unseen notes     -> shows the error pattern
            discussed in the paper: Surgery notes predicted as an organ specialty (cells outlined)

Colour = share of the row (recall-normalised), so small classes are as visible as Surgery; numbers = counts.
Sized for one column of the two-column template; saved as PDF (vector, for LaTeX) and PNG (for Markdown).
Writes report-draft/figures/fig1_submitted_random.{pdf,png} and figA1_bert_grouped.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from config import ART, LABELS, ROOT

OUT = ROOT / "report-draft" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"Cardiovascular-Pulmonary": "Cardio", "Dermatology": "Derm", "Gastroenterology": "Gastro",
         "General Medicine": "GenMed", "Neurology": "Neuro", "Neurosurgery": "NSurg",
         "Obstetrics-Gynecology": "ObGyn", "Ophthalmology": "Ophth", "Orthopedic": "Ortho",
         "Psychiatry-Psychology": "Psych", "Radiology": "Radio", "Surgery": "Surg"}
ORGANS = ["Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology", "Neurology", "Neurosurgery",
          "Obstetrics-Gynecology", "Ophthalmology", "Orthopedic"]


def confusion(df):
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((len(LABELS), len(LABELS)), int)
    for g, p in zip(df.medical_specialty, df.predicted):
        cm[idx[g], idx[p]] += 1
    return cm


def draw(cm, title, path, outline=()):
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.25))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    n = len(LABELS)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.55 else "#1a1a1a",
                        fontweight="bold" if i == j else "normal")
    for (i, j) in outline:
        if not cm[i, j]:
            continue                      # outline only non-empty cells
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#c0392b", linewidth=0.9))
    names = [SHORT[l] for l in LABELS]
    ax.set_xticks(range(n), names, rotation=60, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticks(range(n), names, fontsize=6)
    ax.set_xlabel("Predicted specialty", fontsize=6.5)
    ax.set_ylabel("Gold specialty", fontsize=6.5)
    ax.set_title(title, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{path}.{ext}", dpi=300)
    plt.close(fig)
    errors = int(cm.sum() - np.trace(cm))
    print(f"{path}: {int(cm.sum())} notes, {errors} errors")
    return errors


def main():
    sel = pd.read_csv(ART / "random_selected_predictions.csv", keep_default_na=False)
    draw(confusion(sel), "Submitted system, random held-out set (n = 523)", "fig1_submitted_random")

    bert = pd.read_csv(ART / "grouped_bare_clinicalbert_headtail_none_predictions.csv", keep_default_na=False)
    cm = confusion(bert)
    s = LABELS.index("Surgery")
    cells = [(s, LABELS.index(o)) for o in ORGANS]
    errors = draw(cm, "Bare ClinicalBERT, grouped held-out set (unseen notes)", "figA1_bert_grouped", outline=cells)
    surg_to_organ = sum(int(cm[i, j]) for i, j in cells)
    print(f"  Surgery -> organ specialty (outlined cells): {surg_to_organ} of {errors} errors")


if __name__ == "__main__":
    main()
