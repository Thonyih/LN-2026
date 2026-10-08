"""The only execution path that loads official test inputs: final selected-model inference."""

import numpy as np
from config import ART, ROOT, CACHE, LABELS, PACK, load, save, sha
from data import load_train, load_official_for_prediction, clean, write_labels
from models import model, score
from augment import generate
from rules import apply
from evaluation import store_scores


def main():
    manifest = load(ART / "selection_manifest.json")
    winner = manifest["winner"]
    spec = winner["spec"]
    assert sha(PACK / "train.csv") == load(ART / "data_audit.json")["train_sha256"]
    training = load_train()
    # Official inputs enter only after the model configuration and rule policy have been frozen.
    official = load_official_for_prediction()
    if spec["family"] == "clinicalbert":
        from bert import train_predict

        s, _, metadata = train_predict(
            training,
            None,
            official,
            spec["variant"],
            epochs=spec["epochs"],
            fixed_epochs=True,
        )
        save(ART / "final_bert_training.json", metadata)
    else:
        tr = clean(training, spec["condition"])
        te = clean(official, spec["condition"])
        if spec.get("augmentation"):
            tr, synthetic = generate(tr, spec["augmentation"])
            synthetic.to_csv(ROOT / "data/augmented/final_training.csv", index=False)
        if spec["family"] == "minilm_lr":
            from embeddings import embed

            x = embed(tr, spec["condition"])
            xt = embed(te, spec["condition"])
        else:
            x, xt = tr, te
        est = model(spec)
        est.fit(x, tr.medical_specialty)
        s = score(est, xt)
    pred, routes = apply(official, s, training, winner["policy"])
    assert len(pred) == len(official) == 409
    write_labels(ROOT / "results.txt", pred)
    assert (ROOT / "results.txt").read_text().splitlines() == pred.tolist()
    # Store output provenance, never test-set analysis or labels.
    store_scores(
        ART / "final_prediction_scores.npz",
        official,
        s,
        {"selected_spec": spec, "policy": winner["policy"]},
    )
    save(
        ART / "final_output_verification.json",
        {
            "rows": len(pred),
            "all_labels_valid": set(pred) <= set(LABELS),
            "header": False,
            "order": "original official record order",
            "selected_model": winner["name"],
            "policy": winner["policy"],
            "train_sha256": sha(PACK / "train.csv"),
            "official_sha256": sha(PACK / "test_no_labels.csv"),
            "results_sha256": sha(ROOT / "results.txt"),
        },
    )
    print("Generated results.txt:", len(pred), "valid ordered predictions.", flush=True)


if __name__ == "__main__":
    main()
