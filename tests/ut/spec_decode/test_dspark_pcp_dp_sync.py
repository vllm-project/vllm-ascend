# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for replicated DSpark DP capacity and PCP proposal consistency."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("embedding_tp,lmhead_tp", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("skip_drafts", [False, True, None])
@pytest.mark.parametrize("is_profile,sync_present", [(False, True), (True, True), (False, False)])
def test_dspark_vocab_tp_agrees_on_query_pass(
    monkeypatch, embedding_tp, lmhead_tp, skip_drafts, is_profile, sync_present
):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec.pcp_manager = None
    spec.use_dcp = False
    spec.draft_logits = None
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(4, dtype=torch.int64))
    spec.max_num_tokens = 4
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
    batch = SimpleNamespace(num_reqs=1, is_prefilling_np=np.zeros(1, dtype=np.bool_))
    # A missing field is the supported release lane's pre-skip_drafts state.
    sync = SimpleNamespace(num_reqs=1)
    if skip_drafts is not None:
        sync.skip_drafts = skip_drafts
    monkeypatch.setattr(module, "embedding_tp_enable", lambda: embedding_tp)
    monkeypatch.setattr(module, "lmhead_tp_enable", lambda: lmhead_tp)
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    query_passes = []
    forwarded_k = []

    def parent(self, *args, **kwargs):
        # Upstream only skips the query branch for exactly K=0.
        forwarded_k.append(kwargs["num_speculative_tokens"])
        query_passes.append(kwargs["num_speculative_tokens"] != 0)
        return torch.zeros(1, 3, dtype=torch.int64)

    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    applies = (embedding_tp or lmhead_tp) and sync_present and not is_profile
    # Exercise active K0, active K3, and idle arguments against the shared
    # decision, including when it overrides a caller's local default.
    peers = [(0, False), (3, False), (None, True)]
    for local_k, dummy in peers:
        spec.propose(
            batch,
            {},
            {},
            torch.empty(0),
            None,
            *[torch.empty(0) for _ in range(6)],
            dp_sync_state=sync if sync_present else None,
            dummy_run=dummy,
            is_profile=is_profile,
            num_speculative_tokens=local_k,
        )
    expected_k: list[int | None] = ([0, 0, 0] if skip_drafts is True else [None, 3, None]) if applies else [0, 3, None]
    assert forwarded_k == expected_k
    assert query_passes == [k != 0 for k in expected_k]


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("is_profile", [False, True])
def test_dspark_global_requests_do_not_reuse_local_pcp_dp_capacity(monkeypatch, pcp_enabled, is_profile):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec.pcp_manager = object() if pcp_enabled else None
    spec.draft_logits = None
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
    batch = SimpleNamespace(
        num_reqs=draft_reqs,
        idx_mapping=torch.arange(draft_reqs),
        is_prefilling_np=np.zeros(draft_reqs, dtype=np.bool_),
    )
    sync_kwargs = {} if vllm_version_is("0.33.0") else {"max_query_len": None}
    target_sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([20, 20]), uniform_token_count=5, eager=False, num_reqs=4, **sync_kwargs
    )
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        module, "get_pcp_group", lambda: SimpleNamespace(broadcast=lambda tensor, src: tensor), raising=False
    )
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


@pytest.mark.parametrize("upstream_sync", [False, True])
@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("probabilistic", [False, True])
def test_dspark_syncs_finished_proposals_only_without_upstream_hook(
    monkeypatch, upstream_sync, pcp_enabled, dummy_run, probabilistic
):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec.pcp_manager = object() if pcp_enabled else None
    spec.enable_adaptive_verification = False
    spec.use_dcp = False
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(4, dtype=torch.int64))
    spec.max_num_tokens = 4
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
    spec.draft_logits = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) if probabilistic else None
    original_logits = spec.draft_logits.clone() if probabilistic else None
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([4, 1]), is_prefilling_np=np.zeros(2, dtype=np.bool_))
    local_tokens = torch.tensor([[9, 8], [7, 6]])
    shared_tokens = torch.tensor([[1, 2], [3, 4]])
    shared_logits = torch.full((2, 2, 2), 0.5)
    calls = []
    drafted = False

    def parent(self, *args, **kwargs):
        nonlocal drafted
        drafted = True
        return local_tokens

    def broadcast(tensor, src):
        assert drafted  # Synchronize completed proposals, not backbone hidden states.
        assert src == 0
        calls.append(tensor)
        tensor.copy_(shared_tokens if len(calls) == 1 else shared_logits)
        return tensor

    monkeypatch.setattr(
        module, "PCPManager", SimpleNamespace(sync_drafts=object()) if upstream_sync else object, raising=False
    )
    monkeypatch.setattr(module, "get_pcp_group", lambda: SimpleNamespace(broadcast=broadcast), raising=False)
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    result = spec.propose(batch, {}, {}, torch.empty(0), None, *[torch.empty(0) for _ in range(6)], dummy_run=dummy_run)

    needs_sync = pcp_enabled and not upstream_sync and not dummy_run
    assert len(calls) == (1 + int(probabilistic) if needs_sync else 0)
    assert result is local_tokens
    torch.testing.assert_close(result, shared_tokens if needs_sync else torch.tensor([[9, 8], [7, 6]]))
    if probabilistic:
        if needs_sync:
            original_logits[batch.idx_mapping] = shared_logits
        # Noncontiguous request slots must be written back; unrelated slots stay intact.
        torch.testing.assert_close(spec.draft_logits, original_logits)
