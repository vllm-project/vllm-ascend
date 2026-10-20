# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 acceptance oracle for entropy verification."""

import numpy as np


def accepted_prefix(target_probs, draft_probs, draft_tokens, uniforms, *, thresholds=None):
    """Return accepted draft count, excluding the recovery/bonus token.

    Inputs contain only draft rows. A missing draft distribution is one-hot.
    Placeholder and zero-probability proposals stop the scan.
    """
    target_probs = np.asarray(target_probs, dtype=np.float32)
    draft_probs = None if draft_probs is None else np.asarray(draft_probs, dtype=np.float32)
    uniforms = np.asarray(uniforms, dtype=np.float32)
    thresholds = np.ones(len(draft_tokens), dtype=np.float32) if thresholds is None else thresholds
    for step, token in enumerate(draft_tokens):
        if token < 0:
            return step
        q = np.float32(1) if draft_probs is None else draft_probs[step, token]
        if q <= 0 or target_probs[step, token] / q < np.float32(thresholds[step] * uniforms[step]):
            return step
    return len(draft_tokens)


def entropy_thresholds(probs, posterior_threshold=0.95, posterior_alpha=0.4):
    probs = np.asarray(probs, dtype=np.float32)
    entropy = -np.sum(probs * np.log(probs + np.float32(1e-10)), axis=-1, dtype=np.float32)
    return np.minimum(np.exp(-entropy * np.float32(posterior_alpha)), np.float32(posterior_threshold))
