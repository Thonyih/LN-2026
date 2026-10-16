"""Frozen label-independent, chunk-pooled embeddings. Cache includes model revision and input hash."""

import hashlib, numpy as np
from config import CACHE, FIELDS, MINILM, MINILM_REV


def embed(frame, condition):
    digest = hashlib.sha256(
        (MINILM_REV + condition + frame[FIELDS].to_json()).encode()
    ).hexdigest()
    path = CACHE / f"embeddings_{digest}.npz"
    if path.exists():
        return np.load(path)["values"]
    from sentence_transformers import SentenceTransformer
    import torch

    encoder = SentenceTransformer(
        MINILM,
        revision=MINILM_REV,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )
    vectors = []
    for f in FIELDS:
        texts = []
        owners = []
        for i, text in enumerate(frame[f]):
            ids = encoder.tokenizer.encode(
                text, add_special_tokens=False, truncation=False, verbose=False
            )
            for chunk in [ids[j : j + 220] for j in range(0, len(ids), 220)] or [[]]:
                texts.append(encoder.tokenizer.decode(chunk, skip_special_tokens=True))
                owners.append(i)
        unique = list(dict.fromkeys(texts))
        lookup = {t: i for i, t in enumerate(unique)}
        u = encoder.encode(
            unique,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        e = u[[lookup[t] for t in texts]]
        pooled = np.zeros((len(frame), e.shape[1]), dtype=np.float32)
        np.add.at(pooled, owners, e)
        pooled /= np.bincount(owners, minlength=len(frame))[:, None]
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        vectors.append(pooled)
    out = np.stack(vectors, axis=1).reshape(len(frame), -1)
    np.savez_compressed(path, values=out)
    del encoder
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out
