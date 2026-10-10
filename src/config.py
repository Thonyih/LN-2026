"""All paths are relative to this independent project; nothing imports the original runs."""

import os, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CACHE = ROOT / "cache"
# Course data. Since 10 Oct 2026 the professor provides a cleaned test set (398 rows: the original 409 minus
# 11 broken rows); train.csv, the description and the template are byte-identical to the first pack.
PACK = ROOT.parent / "StudentsPack-v2"
EXPECTED_TEST_ROWS = 398
for p in [ART, CACHE, ROOT / "paper", ROOT / "data/augmented"]:
    p.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE / "huggingface")
os.environ["HF_HUB_CACHE"] = str(CACHE / "huggingface/hub")
os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
LABELS = sorted(
    [
        "Surgery",
        "Cardiovascular-Pulmonary",
        "Orthopedic",
        "Radiology",
        "General Medicine",
        "Gastroenterology",
        "Neurology",
        "Obstetrics-Gynecology",
        "Neurosurgery",
        "Ophthalmology",
        "Psychiatry-Psychology",
        "Dermatology",
    ]
)
FIELDS = ["description", "sample_name", "transcription", "keywords"]
SEED = 20261008
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BERT = "emilyalsentzer/Bio_ClinicalBERT"
BERT_REV = "d5892b39a4adaed74b92212a44081509db72f87b"


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
    )
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
