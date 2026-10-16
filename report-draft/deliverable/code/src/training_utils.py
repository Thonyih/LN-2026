"""Exact accumulation windows, including the final partial effective batch."""


def windows(order, effective_batch=16):
    for start in range(0, len(order), effective_batch):
        yield order[start : start + effective_batch]


def normalised_micro_loss(logits, labels, weights, window_denominator):
    import torch

    return (
        torch.nn.functional.cross_entropy(
            logits, labels, weight=weights, reduction="sum"
        )
        / window_denominator
    )
