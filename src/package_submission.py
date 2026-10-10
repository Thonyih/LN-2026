"""Build the professor's NUM.zip from a supplied final NUM.pdf, code, data and results; never publish."""

import argparse, zipfile
from pathlib import Path
from config import ROOT, PACK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-number", required=True, type=int)
    ap.add_argument("--paper-pdf", required=True, type=Path)
    a = ap.parse_args()
    if a.group_number < 1 or not a.paper_pdf.is_file():
        raise ValueError("Provide a positive group number and the completed paper PDF.")
    if a.paper_pdf.suffix.lower() != ".pdf" or a.paper_pdf.read_bytes()[:5] != b"%PDF-":
        raise ValueError(
            "The supplied paper must be an actual PDF; pagination remains a group check."
        )
    if not (ROOT / "results.txt").exists():
        raise FileNotFoundError("Final predictions have not been generated.")
    destination = ROOT / f"{a.group_number}.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(a.paper_pdf, f"{a.group_number}.pdf")
        z.write(ROOT / "results.txt", "results.txt")
        for p in (ROOT / "src").glob("*.py"):
            z.write(p, "llm-final/src/" + p.name)
        for p in (ROOT / "tests").glob("*.py"):
            z.write(p, "llm-final/tests/" + p.name)
        for p in (ROOT / "data/augmented").glob("*.csv"):
            z.write(p, "llm-final/data/augmented/" + p.name)
        for p in (ROOT / "artifacts").glob("*_splits.csv"):
            z.write(p, "llm-final/artifacts/" + p.name)
        # Include the reviewed error interpretations needed by report.py on a rerun.
        for n in [
            "error_notes.json",
            "draft.md",
            "experiment_details.md",
            "augmentation_review.md",
            "FORMAT_NOTES.md",
        ]:
            p = ROOT / "paper" / n
            if p.exists():
                z.write(p, "llm-final/paper/" + n)
        for n in [
            "selection_manifest.json",
            "data_audit.json",
            "outer_results.json",
            "integrity_verification.json",
            "final_output_verification.json",
        ]:
            p = ROOT / "artifacts" / n
            if p.exists():
                z.write(p, "llm-final/artifacts/" + n)
        for n in ["random_confusion.png", "grouped_confusion.png"]:
            p = ROOT / "artifacts" / n
            if p.exists():
                z.write(p, "llm-final/artifacts/" + n)
        # Preserve the documented relative paths without including model weights.
        for n in ["train.csv", "test_no_labels.csv", "Project-2026-Description.pdf"]:
            z.write(PACK / n, "StudentsPack-v2/" + n)
        z.write(
            PACK / "Project-Template/template.tex",
            "StudentsPack-v2/Project-Template/template.tex",
        )
        for p in sorted((ROOT / "report-draft").rglob("*")):
            if p.is_file():
                z.write(p, "llm-final/" + str(p.relative_to(ROOT)))
        for n in [
            "README.md",
            "MODEL_EXPLANATIONS.md",
            "DEFENSE_GUIDE.md",
            "PROFESSOR_REQUIREMENTS.md",
            "MERGE_STATUS.md",
            "IMPROVEMENTS.md",
            "tests.log",
            "requirements.txt",
            "requirements-lock.txt",
            "protocol.json",
            "run_all.py",
            "run_all.sh",
            "run_training_queue.sh",
        ]:
            z.write(ROOT / n, "llm-final/" + n)
    print(destination)


if __name__ == "__main__":
    main()
