"""Corrected ClinicalBERT: shared development partitions and exact partial-window accumulation."""

import argparse, math, time, hashlib
import numpy as np, pandas as pd
from config import ART, CACHE, LABELS, BERT, BERT_REV, load, save
from data import clean
from evaluation import metrics, store_scores
from rules import apply, POLICIES
from classical import rank
from training_utils import windows, normalised_micro_loss


def texts(frame):
    c = clean(frame, "stripped")
    return (
        c.sample_name.str.strip()
        + " ; "
        + c.description.str.strip()
        + " ; "
        + c.keywords.str.strip()
        + " ; "
        + c.transcription.str.strip()
    ).tolist()


def make_ids(tokenizer, frame, variant):
    out = []
    for t in texts(frame):
        ids = tokenizer.encode(
            t, add_special_tokens=False, truncation=False, verbose=False
        )
        if len(ids) > 510:
            ids = ids[:510] if variant == "head" else ids[:128] + ids[-382:]
        out.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    return out


def train_predict(
    training, validation, target, variant, epochs=3, fixed_epochs=False, checkpoint=None
):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT, revision=BERT_REV)
    ids = make_ids(tok, training, variant)
    vi = make_ids(tok, validation, variant) if validation is not None else None
    ti = make_ids(tok, target, variant)
    y = np.array([LABELS.index(l) for l in training.medical_specialty])
    counts = np.bincount(y, minlength=12)
    weights = torch.tensor(
        np.sqrt(len(y) / (12 * np.maximum(counts, 1))),
        device=device,
        dtype=torch.float32,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT, revision=BERT_REV, num_labels=12
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    updates = math.ceil(len(y) / 16)
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(0.1 * updates * epochs), updates * epochs
    )

    def batch(source, indices):
        return tok.pad(
            {"input_ids": [source[j] for j in indices]},
            padding=True,
            return_tensors="pt",
        ).to(device)

    @torch.no_grad()
    def predict(source):
        model.eval()
        out = []
        for j in range(0, len(source), 8):
            enc = batch(source, list(range(j, min(j + 8, len(source)))))
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    best = None
    best_state = None
    best_validation = None
    best_epoch = epochs
    log = []
    start = time.time()
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y))
        loss_sum = 0.0
        t0 = time.time()
        for step, window in enumerate(windows(order), 1):
            optimiser.zero_grad(set_to_none=True)
            denominator = weights[torch.tensor(y[window], device=device)].sum()
            for j in range(0, len(window), 8):
                indices = window[j : j + 8]
                enc = batch(ids, indices)
                yy = torch.tensor(y[indices], device=device)
                normalised = normalised_micro_loss(
                    model(**enc).logits, yy, weights, denominator
                )
                normalised.backward()
                loss_sum += float((normalised.detach() * denominator).cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            if step % 20 == 0:
                print(
                    {
                        "variant": variant,
                        "epoch": epoch,
                        "update": step,
                        "updates": updates,
                        "elapsed_seconds": round(time.time() - t0),
                    },
                    flush=True,
                )
        entry = {
            "epoch": epoch,
            "weighted_training_loss": loss_sum
            / float(weights[torch.tensor(y, device=device)].sum().cpu()),
            "updates": updates,
            "seconds": time.time() - t0,
        }
        if validation is not None:
            vs = predict(vi)
            m = metrics(
                validation.medical_specialty,
                np.array(LABELS, dtype=object)[vs.argmax(1)],
            )
            entry["validation_metrics"] = m
            if best is None or rank(m) > best:
                best = rank(m)
                best_epoch = epoch
                best_validation = vs.copy()
                best_state = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in model.state_dict().items()
                }
        print(
            {
                "variant": variant,
                "epoch_complete": epoch,
                "seconds": round(entry["seconds"]),
                "validation_macro_f1": entry.get("validation_metrics", {}).get(
                    "macro_f1"
                ),
            },
            flush=True,
        )
        log.append(entry)
    if best_state is not None:
        model.load_state_dict(best_state)
    # Evaluation uses the development-selected checkpoint, never the target gold.
    target_scores = predict(ti)
    if checkpoint:
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint
        )
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return (
        target_scores,
        best_validation,
        {
            "selected_epoch": best_epoch,
            "variant": variant,
            "log": log,
            "seconds": time.time() - start,
            "revision": BERT_REV,
            "gradient_accumulation": "weighted sum loss normalised over each effective batch of up to 16; every partial window stepped",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["random", "grouped"], required=True)
    ap.add_argument("--variant", choices=["head", "headtail"], required=True)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Repeat a completed training job explicitly.",
    )
    args = ap.parse_args()
    frame = pd.read_csv(ART / f"{args.regime}_splits.csv", keep_default_na=False)
    tr = frame[frame.split == "train"]
    va = frame[frame.split == "validation"]
    te = frame[frame.split == "test"]
    title = "clinicalbert_" + args.variant
    path = ART / f"{args.regime}_{title}_training.json"
    if path.exists() and not args.force:
        raise RuntimeError(
            "Completed run already exists; pass --force to repeat it or use a new version directory."
        )
    ts, vs, metadata = train_predict(tr, va, te, args.variant, epochs=3)
    store_scores(ART / f"{args.regime}_{title}_validation.npz", va, vs, metadata)
    store_scores(ART / f"{args.regime}_{title}_test.npz", te, ts, metadata)
    rows = []
    spec = {
        "family": "clinicalbert",
        "condition": "stripped",
        "variant": args.variant,
        "epochs": metadata["selected_epoch"],
    }
    for policy in POLICIES:
        pred, _ = apply(va, vs, tr, policy)
        rows.append(
            {
                "regime": args.regime,
                "name": title,
                "spec": spec,
                "policy": policy,
                "metrics": metrics(va.medical_specialty, pred),
            }
        )
    save(path, metadata)
    save(ART / f"{args.regime}_{title}_validation.json", rows)
    print("COMPLETED", args.regime, title, metadata["selected_epoch"], flush=True)


if __name__ == "__main__":
    main()
