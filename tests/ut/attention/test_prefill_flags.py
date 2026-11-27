# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import apply_prefill_as_decode_mask, has_requests_to_run_as_prefill


@pytest.mark.parametrize("source", ["numpy", "cpu_tensor"])
@pytest.mark.parametrize(
    "override,expected",
    [(None, [False, True, True]), ([False, False, True], [False, True, False]), ([False, True, True], [False] * 3)],
)
def test_effective_prefill_preserves_request_lifecycle(source, override, expected):
    phases = np.array([False, True, True])
    mask = None if override is None else np.array(override)
    is_prefilling = torch.from_numpy(phases) if source == "cpu_tensor" else phases
    prefill_runs_as_decode = torch.from_numpy(mask) if source == "cpu_tensor" and mask is not None else mask

    np.testing.assert_array_equal(apply_prefill_as_decode_mask(is_prefilling, prefill_runs_as_decode), expected)
    batch = SimpleNamespace(is_prefilling_np=phases)
    if mask is not None:
        batch.prefill_runs_as_decode_np = mask
    assert has_requests_to_run_as_prefill(batch) is any(expected)
    np.testing.assert_array_equal(phases, [False, True, True])
    if mask is not None:
        np.testing.assert_array_equal(mask, override)


@pytest.mark.parametrize("override", [[True], [[True, False]]])
def test_effective_prefill_rejects_mismatched_request_mask(override):
    with pytest.raises(RuntimeError, match="mask must match the request phase mask"):
        apply_prefill_as_decode_mask(np.array([True, False]), np.array(override))
