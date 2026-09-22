# SPDX-License-Identifier: Apache-2.0
"""Strict full-vocabulary numerical regression comparison."""

import numpy as np


def check_logits(reference, current, atol):
    if reference.shape != (8, 154880) or current.shape != reference.shape:
        raise ValueError("Unexpected logits shape")
    if reference.dtype != np.float32 or current.dtype != np.float32:
        raise ValueError("Expected float32 logits storage")
    if not np.isfinite(reference).all():
        raise ValueError("Non-finite reference logits")
    if not np.isfinite(current).all():
        return {"status": "FAIL", "reason": "Non-finite current logits", "max_abs": None}
    delta = float(np.abs(reference.astype(np.float64) - current.astype(np.float64)).max())
    top1 = bool(np.array_equal(reference.argmax(-1), current.argmax(-1)))
    return {"status": "PASS" if delta <= atol and top1 else "FAIL", "max_abs": delta, "top1_equal": top1}
