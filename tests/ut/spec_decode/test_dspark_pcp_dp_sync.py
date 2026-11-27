# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for replicated DSpark DP capacity and PCP proposal consistency."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

import vllm_ascend.worker.v2.pcp_manager as pcp_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module


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
    spec.pcp_manager = object.__new__(pcp_module.AscendPCPManager) if pcp_enabled else None
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
    sync_kwargs = {"max_query_len": None}
    target_sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([20, 20]), uniform_token_count=5, eager=False, num_reqs=4, **sync_kwargs
    )
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.pcp_manager.get_pcp_group", lambda: SimpleNamespace(broadcast=lambda tensor, src: tensor)
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


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("logits_kind", ["none", "cache"])
def test_dspark_syncs_finished_proposals_once_across_runner_versions(monkeypatch, pcp_enabled, dummy_run, logits_kind):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec.pcp_manager = object.__new__(pcp_module.AscendPCPManager) if pcp_enabled else None
    spec.enable_adaptive_verification = False
    spec.use_dcp = False
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(4, dtype=torch.int64))
    spec.max_num_tokens = 4
    spec.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
    logits_buffer = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) if logits_kind != "none" else None
    spec.draft_logits = SimpleNamespace(values=logits_buffer) if logits_kind == "cache" else logits_buffer
    original_logits = logits_buffer.clone() if logits_buffer is not None else None
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([4, 1]), is_prefilling_np=np.zeros(2, dtype=np.bool_))
    token_buffer = torch.tensor([[9, 8], [7, 6], [5, 4]])
    local_tokens = token_buffer[:2]
    assert local_tokens.is_contiguous()
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
        assert tensor.is_contiguous()
        calls.append(tensor)
        tensor.copy_(shared_tokens if len(calls) == 1 else shared_logits)
        return tensor

    sync_method = pcp_module.AscendPCPManager.sync_drafts
    monkeypatch.setitem(
        pcp_module.PCPManager.sync_drafts.__globals__,
        "get_pcp_group",
        lambda: SimpleNamespace(broadcast=broadcast),
    )
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    result = spec.propose(batch, {}, {}, torch.empty(0), None, *[torch.empty(0) for _ in range(6)], dummy_run=dummy_run)

    broadcasts = 1 + int(logits_buffer is not None)
    assert len(calls) == 0
    if pcp_enabled and not dummy_run:
        # The upstream runner invokes the manager after propose returns.
        assert sync_method(spec.pcp_manager, result, batch.idx_mapping, spec.draft_logits) is None
    needs_sync = pcp_enabled and not dummy_run
    assert len(calls) == (broadcasts if needs_sync else 0)
    assert result is local_tokens
    assert result.is_contiguous()
    if needs_sync:
        assert calls[0] is local_tokens
    torch.testing.assert_close(token_buffer[2], torch.tensor([5, 4]))
    torch.testing.assert_close(result, shared_tokens if needs_sync else torch.tensor([[9, 8], [7, 6]]))
    if logits_buffer is not None:
        if needs_sync:
            original_logits[batch.idx_mapping] = shared_logits
        # Noncontiguous request slots must be written back; unrelated slots stay intact.
        torch.testing.assert_close(logits_buffer, original_logits)


def test_owner_drafts_merge_in_scheduler_order_after_owner_changes(monkeypatch):
    managers = [object.__new__(pcp_module.AscendPCPManager) for _ in range(2)]
    draft_buffers = [torch.full((6, 2), -9, dtype=torch.int64) for _ in managers]
    for manager in managers:
        manager.device = torch.device("cpu")
    monkeypatch.setattr(pcp_module, "async_copy_to_gpu", lambda array, **kwargs: torch.from_numpy(array))
    group = object()
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: SimpleNamespace(device_group=group))

    # State 11 changes owners when state 7 finishes ahead of it.
    for states in (np.array([7, 11, 4, 19, 3]), np.array([11, 4, 19, 3])):
        global_batch = SimpleNamespace(idx_mapping_np=states, num_reqs=len(states))
        expected = torch.tensor(np.stack((states, states + 1), axis=1), dtype=torch.int64)
        expected[0, 0] = 0
        expected[-1, 1] = -1
        outputs = []
        staged = []
        for rank, (manager, buffer) in enumerate(zip(managers, draft_buffers)):
            local_states = states[rank::2]
            manager._local_batch = SimpleNamespace(idx_mapping_np=local_states, num_reqs=len(local_states))
            local_batch, query_rows, owned_rows = manager.owner_draft_rows(global_batch)
            assert local_batch is manager._local_batch
            torch.testing.assert_close(owned_rows, torch.arange(rank, len(states), 2))
            torch.testing.assert_close(query_rows[owned_rows], torch.arange(len(local_states)))
            assert torch.all(query_rows[1 - rank :: 2] == -1)
            buffer[: len(local_states)].copy_(expected[rank::2])
            output = buffer[: len(states)]
            assert manager.stage_owner_drafts(output, owned_rows) is output
            torch.testing.assert_close(output[: len(local_states)], expected[rank::2])
            outputs.append(output)
            staged.append(manager._staged_owner_drafts)

        merged = torch.maximum(*staged)

        def all_reduce(tokens, *, op, group, merged=merged):
            assert op == torch.distributed.ReduceOp.MAX
            assert group is pcp_module.get_pcp_group().device_group
            assert tokens.dtype == torch.int32
            tokens.copy_(merged)

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        for manager, buffer, result in zip(managers, draft_buffers, outputs):
            tail = buffer[len(states) :].clone()
            manager.sync_drafts(result, torch.from_numpy(states), None)
            torch.testing.assert_close(result, expected)
            assert result.dtype == torch.int64
            assert result.data_ptr() == buffer.data_ptr()
            result[:, 1:] = -1  # The runner truncates unused draft columns in place.
            assert torch.all(buffer[: len(states), 1:] == -1)
            torch.testing.assert_close(buffer[len(states) :], tail)
            assert manager._staged_owner_drafts is None


@pytest.mark.parametrize(
    ("has_prefill", "num_spec_tokens", "cg_mode", "overlap"),
    [
        (False, 2, CUDAGraphMode.FULL, True),
        (True, 2, CUDAGraphMode.FULL, False),
        (False, 0, CUDAGraphMode.FULL, False),
        (False, 2, CUDAGraphMode.NONE, False),
    ],
)
def test_wkv_gather_is_joined_after_target_sampling(monkeypatch, has_prefill, num_spec_tokens, cg_mode, overlap):
    manager = object.__new__(pcp_module.AscendPCPManager)
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec._owner_pcp_size = 2
    spec.num_query_per_req = 3
    spec.num_speculative_steps = 2
    spec.max_num_tokens = 16
    desc = SimpleNamespace(cg_mode=cg_mode)
    spec._owner_graph_manager = SimpleNamespace(dispatch=lambda *_: desc)
    manager.pcp_world_size = 2
    batch = SimpleNamespace(num_reqs=2, num_tokens=3, is_dummy=False, is_prefilling_np=np.array([False, has_prefill]))
    manager._global_batch = batch
    manager._hidden_restore_idx = torch.tensor([0, 2, 1, 3])
    manager._restore_unpadded_global_batch = lambda: True
    manager._expose_unpadded_global_batch = lambda batch: batch
    logits_stream = object()
    device_group = object()
    calls = []
    wkv_stream = SimpleNamespace(wait_stream=lambda producer: calls.append(("producer", producer)))
    manager._draft_wkv_stream = wkv_stream
    monkeypatch.setattr(torch.npu, "current_stream", lambda: logits_stream)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: SimpleNamespace(device_group=device_group))

    def restore(manager, hidden, aux):
        calls.append(("restore", aux is not None))
        return hidden, aux, batch

    def gather(output, local, *, group, async_op):
        assert group is device_group and async_op
        calls.append("gather")

        def wait():
            calls.append("wait")
            output.copy_(torch.cat((local, local + 10)))

        return SimpleNamespace(wait=wait)

    def project(context_states):
        calls.append("project")
        return context_states.unsqueeze(1)

    spec.model = SimpleNamespace(combine_hidden_states=lambda states: states, project_context_wkv=project)
    monkeypatch.setattr(pcp_module.PCPManager, "restore_for_sampling", restore)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    aux = [torch.tensor([[1], [2]]), torch.tensor([[3], [4]])]
    manager.draft_wkv_batch_desc = spec.dispatch_owner_graph(batch, num_spec_tokens)
    assert manager.draft_wkv_batch_desc is (desc if overlap else None)
    _, eager_aux, global_batch = manager.restore_for_sampling(torch.zeros(3, 1), aux)
    assert global_batch is batch
    assert calls == [("restore", not overlap)]

    # The target's logits hook runs before sampling, while the draft joins later.
    manager.start_draft_wkv_gather(spec.project_owner_wkv)
    calls.append("sample")
    if overlap:
        assert eager_aux is None
        expected_calls = [("restore", False), ("producer", logits_stream), "project", "gather", "sample"]
        assert calls == expected_calls
        staging = torch.full((4, 1, 2), -1)
        manager.finish_draft_wkv_gather(staging[: batch.num_tokens])
        assert calls == [*expected_calls, "wait"]
        torch.testing.assert_close(staging, torch.tensor([[[1, 3]], [[11, 13]], [[2, 4]], [[-1, -1]]]))
        assert manager.draft_wkv_batch_desc is None
    else:
        torch.testing.assert_close(eager_aux, aux)
        assert calls == [("restore", True), "sample"]
