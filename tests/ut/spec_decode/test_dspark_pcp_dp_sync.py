# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for replicated DSpark requests exceeding target PCP DP capacity."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("is_profile", [False, True])
def test_dspark_global_requests_do_not_reuse_local_pcp_dp_capacity(monkeypatch, pcp_enabled, is_profile):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec.pcp_manager = object() if pcp_enabled else None
    spec.enable_adaptive_verification = False
    spec.use_dcp = False
    spec.attn_architecture = "MLA"
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(64, dtype=torch.int64))
    spec.max_num_tokens = 64
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    # A target batch can have four PCP-local requests while the replicated
    # DSpark drafter consumes the restored 32-request batch.
    draft_reqs = 32 if pcp_enabled else 4
    batch = SimpleNamespace(num_reqs=draft_reqs, is_prefilling_np=np.zeros(draft_reqs, dtype=np.bool_))
    target_sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([20, 20]), uniform_token_count=5, eager=False, num_reqs=4
    )
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    seen = []

    def parent(self, *args, **kwargs):
        sync = args[11]
        seen.append(sync)
        if sync is not None:
            # Use the real upstream request-capacity check, rather than
            # duplicating its assertion in this probe.
            DraftModelSpeculator._build_uniform_batch_dp_sync(self, sync, batch.num_reqs, 5)
        return torch.empty(batch.num_reqs, 5, dtype=torch.int64)

    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    spec.propose(
        batch,
        {},
        {},
        torch.empty(0),
        None,
        *[torch.empty(0) for _ in range(6)],
        dp_sync_state=target_sync,
        is_profile=is_profile,
    )
    assert seen[0] is (None if pcp_enabled else target_sync)
