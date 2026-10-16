import numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from config import LABELS, ART, save


def metrics(gold, pred):
    r = classification_report(
        gold, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(
            gold, pred, labels=LABELS, average="macro", zero_division=0
        ),
        "minimum_f1": float(
            f1_score(gold, pred, labels=LABELS, average=None, zero_division=0).min()
        ),
        "per_class": {l: r[l] for l in LABELS},
    }


def store_scores(path, frame, scores, metadata):
    assert frame.source_record.is_unique
    assert scores.shape == (len(frame), len(LABELS)) and np.isfinite(scores).all()
    np.savez_compressed(
        path,
        scores=scores,
        source_record=frame.source_record.to_numpy(),
        labels=np.array(LABELS),
        metadata=np.array(__import__("json").dumps(metadata, default=str)),
    )


def read_scores(path, frame):
    z = np.load(path)
    assert z["labels"].tolist() == LABELS
    ids = z["source_record"].tolist()
    assert len(ids) == len(set(ids)) == len(frame)
    assert (
        z["scores"].shape == (len(frame), len(LABELS))
        and np.isfinite(z["scores"]).all()
    )
    lookup = {x: i for i, x in enumerate(ids)}
    assert set(ids) == set(frame.source_record)
    return z["scores"][[lookup[x] for x in frame.source_record]]


def plot_confusion(frame, pred, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    short = [
        "Cardio",
        "Derm",
        "Gastro",
        "GenMed",
        "Neuro",
        "Neurosurg",
        "ObGyn",
        "Ophth",
        "Ortho",
        "Psych",
        "Radio",
        "Surg",
    ]
    cm = confusion_matrix(frame.medical_specialty, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ConfusionMatrixDisplay(cm, display_labels=short).plot(
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=60, values_format="d"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
