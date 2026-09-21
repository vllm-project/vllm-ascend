# SPDX-License-Identifier: Apache-2.0
"""Building blocks for an instrumented GLM prefix comparison, not a baseline.

The caller must capture layer 7's actual (hidden, residual) outputs on every
TP rank. This helper neither truncates the reference model nor substitutes a
reduced checkpoint for the full reference. Only eager, TP-only instrumentation
is supported initially; graph capture and MTP require separate qualification.
"""

from __future__ import annotations

import math

import numpy as np


def project_boundary(model, hidden, residual, *, token_count, row_indices, gather_rows):
    """Project captured rows using the loaded model's real norm and head.

    All TP ranks must call this function, including ranks without returned logits.
    gather_rows must be the runtime's TP all-gather along the token dimension.
    Clones protect the original forward from fused norm implementations which
    modify their arguments. The caller owns serialization and rank selection.
    """
    if token_count <= 0 or not row_indices or len(set(row_indices)) != len(row_indices):
        raise ValueError("expected positive token count and unique selected rows")
    if any(type(index) is not int or not 0 <= index < token_count for index in row_indices):
        raise ValueError("selected row outside valid tokens")
    if residual is None or hidden.shape != residual.shape or len(hidden.shape) != 2:
        raise ValueError("expected matching two-dimensional hidden and residual tensors")
    hidden = hidden.clone()
    residual = residual.clone()
    if hidden.shape[0] != token_count:
        hidden = gather_rows(hidden)[:token_count]
        residual = gather_rows(residual)[:token_count]
    if hidden.shape[0] != token_count or hidden.shape != residual.shape:
        raise ValueError("TP reconstruction did not recover valid token rows")
    # Keep the same token-batch shape as the normal final norm. Selecting one
    # row before RMSNorm can select a different NPU reduction kernel.
    hidden = hidden.contiguous()
    residual = residual.contiguous()
    merged = (hidden[row_indices] + residual[row_indices]).contiguous()
    normalized, _ = model.model.norm(hidden, residual)
    normalized = normalized[row_indices].contiguous()
    logits = model.compute_logits(normalized)
    return {"hidden": merged, "normalized": normalized, "logits": logits}


def compare_arrays(reference, candidate, *, atol=None, rtol=None):
    """Report full-vector errors; without preselected tolerances, no PASS.

    Tolerance selection belongs to the qualification protocol, never to a retry
    loop. A failed result does not change tolerances or launch another experiment.
    """
    if (atol is None) != (rtol is None):
        raise ValueError("supply both atol and rtol, or neither")
    if atol is not None and any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 2 or not ref.size:
        raise ValueError("expected nonempty matching [positions, features] arrays")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("non-finite values in captured arrays")
    delta = np.abs(cand - ref)
    ref_norm = np.linalg.norm(ref, axis=1)
    cand_norm = np.linalg.norm(cand, axis=1)
    denom = ref_norm * cand_norm
    cosine = np.divide(np.sum(ref * cand, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    cosine[(ref_norm == 0) & (cand_norm == 0)] = 1
    result = {
        "status": "UNASSESSED",
        "shape": list(ref.shape),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rms_error": float(np.sqrt(np.mean(delta**2))),
        "min_cosine_similarity": float(np.clip(cosine, -1, 1).min()),
        "argmax_agreement": float(np.mean(ref.argmax(axis=1) == cand.argmax(axis=1))),
        "atol": atol,
        "rtol": rtol,
    }
    if atol is not None:
        mismatch = delta > atol + rtol * np.abs(ref)
        result["status"] = "FAIL" if mismatch.any() else "PASS"
        result["mismatched_elements"] = int(mismatch.sum())
        result["first_mismatch"] = np.argwhere(mismatch)[0].tolist() if mismatch.any() else None
    return result
