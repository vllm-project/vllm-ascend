# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark sampling dispatch through the NPU gumbel_sample binding.

The gumbel_sample fan-out used to be installed (and unit-tested) by the
retired patch_v2/patch_triton.py; it is now registered through the
official kernel dispatcher, whose table is covered by
tests/ut/test_triton_kernel_overrides.py.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator

from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample


@pytest.mark.parametrize("probabilistic", [False, True])
def test_dspark_sample_logits_dispatch(monkeypatch, probabilistic):
    speculator = dspark_speculator.DSparkSpeculator.__new__(dspark_speculator.DSparkSpeculator)
    speculator.model = MagicMock()
    speculator.model.map_draft_to_target.side_effect = lambda tokens: tokens + 10
    speculator._d2t_scatter_index = None
    speculator.temperature = torch.tensor([0.5, 1.5])
    speculator.seeds = torch.tensor([3, 7])
    speculator._step_cols = torch.arange(2, dtype=torch.int32)
    speculator.draft_logits = torch.empty(2, 2, 3) if probabilistic else None
    speculator.use_fp64_gumbel = False

    # This fixture bypasses __init__, so provide defaults used by current main.
    # They are harmless extra instance attributes on the v0.28-compatible lane.
    speculator.acceptance_estimator = None
    speculator.draft_watermarker = None

    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]])
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    sample_pos = torch.tensor([8, 12])
    sampled = torch.tensor([2, 1])
    sample = create_autospec(categorical_sample, return_value=sampled)
    monkeypatch.setattr(dspark_speculator, "gumbel_sample", sample)

    result = speculator._sample_logits(
        logits,
        idx_mapping,
        sample_pos,
        step=1,
    )

    if probabilistic:
        assert result is sampled
        sample.assert_called_once()
        args, kwargs = sample.call_args
        assert args[0] is logits
        assert args[1] is idx_mapping
        torch.testing.assert_close(args[4], sample_pos - 1)
        assert kwargs["apply_temperature"] is True
        assert kwargs["is_drafting"] is True
        assert kwargs["logits_cache"] is speculator.draft_logits
        torch.testing.assert_close(
            kwargs["logits_cache_col"],
            speculator._step_cols[1],
        )
        assert kwargs["use_fp64"] is False
    else:
        sample.assert_not_called()
        torch.testing.assert_close(
            result,
            logits.argmax(dim=-1) + 10,
        )
