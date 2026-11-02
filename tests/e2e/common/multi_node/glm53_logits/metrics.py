# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compare aligned FULL-vocabulary raw logits. Never accept missing captures."""

import numpy as np


def compare(reference, candidate):
    for key in ("prompt_ids", "token_ids"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Unaligned {key}; logits comparison is invalid")
    a, b = reference["logits"].astype(np.float64), candidate["logits"].astype(np.float64)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        raise ValueError("Logits dimensions differ or are empty")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite logits")
    ac, bc = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    norm = np.linalg.norm(ac, axis=-1)
    rel = np.linalg.norm(ac - bc, axis=-1) / np.maximum(norm, 1e-12)
    cosine = (ac * bc).sum(-1) / np.maximum(norm * np.linalg.norm(bc, axis=-1), 1e-12)

    def log_softmax(x):
        shifted = x - x.max(-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(la) * (la - lb)).sum(-1)
    a_top = np.argpartition(a, -2, axis=-1)[:, -2:]
    top_values = np.sort(np.take_along_axis(a, a_top, axis=-1), axis=-1)
    margin = top_values[:, 1] - top_values[:, 0]
    agreement = a.argmax(-1) == b.argmax(-1)
    high_margin = margin >= 0.5
    top20a, top20b = np.argpartition(a, -20, axis=-1)[:, -20:], np.argpartition(b, -20, axis=-1)[:, -20:]
    overlap = [len(set(x) & set(y)) / 20 for x, y in zip(top20a, top20b)]
    return {
        "positions": a.shape[0],
        "vocab_size": a.shape[1],
        "max_absolute_error": float(np.abs(a - b).max()),
        "mean_absolute_error": float(np.abs(a - b).mean()),
        "max_centered_relative_l2": float(rel.max()),
        "min_centered_cosine": float(cosine.min()),
        "max_kl_ref_to_candidate": float(kl.max()),
        "mean_kl_ref_to_candidate": float(kl.mean()),
        "top1_agreement": float(agreement.mean()),
        "mean_top20_overlap": float(np.mean(overlap)),
        "high_margin_top1_mismatches": int((high_margin & ~agreement).sum()),
        "high_margin_positions": int(high_margin.sum()),
        "per_position_relative_l2": rel.tolist(),
        "per_position_kl": kl.tolist(),
    }
