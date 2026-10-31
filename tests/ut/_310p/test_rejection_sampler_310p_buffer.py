# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression: MTP rejection sampler must not shrink host buffers across steps."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def sampler_310v2():
    """Build RejectionSampler310V2 with NPU / parent init mocked out."""

    def _parent_init(self, sampler, spec_config, device, **kwargs):
        del sampler, device, kwargs
        self.num_speculative_steps = int(getattr(spec_config, "num_speculative_tokens", 1))

    with (
        patch("vllm_ascend._310p.worker.v2.rejection_sampler.torch.npu.Stream", MagicMock),
        patch("vllm_ascend._310p.worker.v2.rejection_sampler.torch.npu.Event", MagicMock),
        patch(
            "vllm_ascend._310p.worker.v2.rejection_sampler.is_pin_memory_available",
            return_value=False,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler.RejectionSampler.__init__",
            _parent_init,
        ),
    ):
        from vllm_ascend._310p.worker.v2.rejection_sampler import RejectionSampler310V2

        mock_sampler = SimpleNamespace(max_num_reqs=32)
        spec = MagicMock()
        spec.num_speculative_tokens = 1
        spec.method = "mtp"

        return RejectionSampler310V2(mock_sampler, spec, device=torch.device("cpu"))


def test_host_buffers_keep_max_num_reqs_capacity(sampler_310v2):
    rs = sampler_310v2
    assert rs.sampled_tokens_cpu.shape[0] == 32

    # Temperature path used to assign a 1-row tensor and shrink capacity.
    sampled_1 = torch.zeros((1, rs.num_speculative_steps + 1), dtype=torch.int32)
    num_1 = torch.ones(1, dtype=torch.int32)
    rej_1 = torch.zeros(1, dtype=torch.int32)
    rs.sampled_tokens_cpu[:1].copy_(sampled_1)
    rs.num_sampled_cpu[:1].copy_(num_1)
    rs.num_rejected_cpu[:1].copy_(rej_1)

    assert rs.sampled_tokens_cpu.shape[0] == 32
    assert rs.num_sampled_cpu.shape[0] == 32
    assert rs.num_rejected_cpu.shape[0] == 32
    assert torch.equal(rs.sampled_tokens_cpu[:1], sampled_1)

    # Later greedy step with 4 requests (CI: Target [1, 2] vs Tensor [4, 2]).
    sampled_4 = torch.arange(8, dtype=torch.int32).view(4, 2)
    num_4 = torch.full((4,), 2, dtype=torch.int32)
    rej_4 = torch.zeros(4, dtype=torch.int32)
    rs.sampled_tokens_cpu[:4].copy_(sampled_4)
    rs.num_sampled_cpu[:4].copy_(num_4)
    rs.num_rejected_cpu[:4].copy_(rej_4)

    assert rs.sampled_tokens_cpu.shape[0] == 32
    assert torch.equal(rs.sampled_tokens_cpu[:4], sampled_4)
