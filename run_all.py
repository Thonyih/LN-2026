"""Portable full-experiment launcher. Reuses completed BERT jobs unless explicitly forced."""

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def run(title, arguments, log_name=None):
    print(f"\n{title}", flush=True)
    started = time.perf_counter()
    command = [sys.executable, *map(str, arguments)]
    if log_name:
        with (ROOT / log_name).open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"{title} failed (exit {result.returncode}). Read {ROOT / log_name}."
            )
    else:
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Completed in {time.perf_counter() - started:.1f} seconds.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full experiment. WARNING: BERT TAKES LONG."
    )
    parser.add_argument(
        "--force-bert",
        action="store_true",
        help="Retrain all four BERT comparisons instead of reusing completed runs.",
    )
    args = parser.parse_args()
    print(
        "WARNING: BERT TAKES LONG. For quick checks, run src/defense_demo.py, "
        "the unit tests, and src/verify.py.",
        flush=True,
    )
    run("Unit tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], "tests.log")
    stages = [
        ("Prepare partitions and labelled-data audit", "prepare", "prepare.log"),
        ("Classical model comparisons", "classical", "classical.log"),
        ("Align fitting-data provenance", "reconcile_classical_scores", "alignment.log"),
        ("Augmentation candidates", "augmentation_candidates", "augmentation_candidates.log"),
        ("Augmentation diagnostics", "augmentation_study", "augmentation.log"),
    ]
    for title, script, log_name in stages:
        run(title, [ROOT / "src" / f"{script}.py"], log_name)
    for regime in ["random", "grouped"]:
        for variant in ["head", "headtail"]:
            name = f"{regime}_clinicalbert_{variant}"
            completed = all(
                (ROOT / "artifacts" / f"{name}_{suffix}").is_file()
                for suffix in ["training.json", "validation.json", "test.npz"]
            )
            if completed and not args.force_bert:
                print(f"Reusing completed BERT comparison: {regime} {variant}", flush=True)
                continue
            arguments = [
                ROOT / "src/bert.py", "--regime", regime, "--variant", variant
            ]
            if args.force_bert:
                arguments.append("--force")
            run(
                f"WARNING: BERT TAKES LONG — training {regime}/{variant}",
                arguments,
                f"bert_{regime}_{variant}.log",
            )
    for title, script, log_name in [
        ("Freeze selection and evaluate", "finalize", "finalize.log"),
        ("Generate official predictions", "predict", "predict.log"),
        ("Memorisation analysis", "memorisation", "memorisation.log"),
        ("Generate paper", "report", None),
        ("Verify saved evidence", "verify", None),
    ]:
        run(title, [ROOT / "src" / f"{script}.py"], log_name)
    print("\nFull experiment completed.", flush=True)


if __name__ == "__main__":
    main()
