# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import get_effective_prefill_flags, has_effective_prefill


@pytest.mark.parametrize("source", ["numpy", "common_metadata"])
@pytest.mark.parametrize(
    "override,expected",
    [(None, [False, True, True]), ([False, False, True], [False, True, False]), ([False, True, True], [False] * 3)],
)
def test_effective_prefill_preserves_request_lifecycle(source, override, expected):
    phases = np.array([False, True, True])
    batch = (
        SimpleNamespace(is_prefilling_np=phases)
        if source == "numpy"
        else SimpleNamespace(is_prefilling=torch.from_numpy(phases))
    )
    if override is not None:
        batch.prefill_runs_as_decode_np = np.array(override)

    np.testing.assert_array_equal(get_effective_prefill_flags(batch), expected)
    assert has_effective_prefill(batch) is any(expected)
    np.testing.assert_array_equal(phases, [False, True, True])


@pytest.mark.parametrize("override", [[True], [[True, False]]])
def test_effective_prefill_rejects_mismatched_request_mask(override):
    batch = SimpleNamespace(is_prefilling_np=np.array([True, False]), prefill_runs_as_decode_np=np.array(override))
    with pytest.raises(RuntimeError, match="mask must match the request phase mask"):
        get_effective_prefill_flags(batch)
