# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""FULL replay metadata reuse keeps builder updates and padding semantics."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as upstream_dflash

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph_module
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


def _metadata_speculator(monkeypatch, cls=AscendDFlashSpeculator, query_len=4):
    speculator = cls.__new__(cls)
    speculator.num_query_per_req = query_len
    speculator._group_causal = {0: False, 1: True}
    speculator.arange_np = np.arange(9, dtype=np.int32)
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_cpu_upper_bound=torch.tensor([10, 20]))
    built = []

    def build(**kwargs):
        # Emulate a builder with state that must advance once per proposal.
        query_lens = kwargs["query_start_loc_np"][1:].tolist()
        metadata = {
            name: SimpleNamespace(actual_seq_lengths_q=query_lens.copy(), generation=len(built))
            for name in ("draft.0", "draft.1")
        }
        built.append(metadata)
        return metadata

    speculator._build_attn_metadata = MagicMock(side_effect=build)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", nullcontext)
    return speculator, built


def _build(speculator, desc, **overrides):
    args = dict(
        batch_desc=desc,
        num_reqs=speculator.input_batch.num_reqs,
        num_query_per_req=speculator.num_query_per_req,
        seq_lens_cpu_upper_bound=speculator.input_batch.seq_lens_cpu_upper_bound,
        step=speculator.num_query_per_req,
        causal=speculator._group_causal,
    )
    args.update(overrides)
    return speculator._build_uniform_attn_metadata(**args)


@pytest.mark.parametrize("query_len", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("padded_reqs", [2, 4])
def test_reuse_pads_queries_and_is_consumed_once(monkeypatch, query_len, padded_reqs):
    speculator, built = _metadata_speculator(monkeypatch, query_len=query_len)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, padded_reqs * query_len, padded_reqs)
    with speculator._reuse_full_graph_attn_metadata():
        metadata = _build(speculator, desc)
        assert speculator.take_draft_attn_metadata(desc) is metadata
        assert speculator.take_draft_attn_metadata(desc) is None
    assert len(built) == 1
    for layer in metadata.values():
        assert layer.actual_seq_lengths_q == list(range(query_len, desc.num_tokens + 1, query_len))


@pytest.mark.parametrize("change", ["descriptor", "uniform_count", "request_count", "seq_lens", "layout"])
def test_changed_batch_rejects_pending_metadata(monkeypatch, change):
    speculator, _ = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    with speculator._reuse_full_graph_attn_metadata():
        _build(speculator, desc)
        replay_desc = desc
        if change == "descriptor":
            replay_desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 2)
        elif change == "uniform_count":
            replay_desc = replace(desc, uniform_token_count=speculator.num_query_per_req)
        elif change == "request_count":
            speculator.input_batch.num_reqs = 1
        elif change == "seq_lens":
            # Equal values in a new buffer must not accidentally reuse metadata.
            speculator.input_batch.seq_lens_cpu_upper_bound = speculator.input_batch.seq_lens_cpu_upper_bound.clone()
        else:
            speculator.num_query_per_req = 5
        assert speculator.take_draft_attn_metadata(replay_desc) is None
        assert speculator.take_draft_attn_metadata(desc) is None


@pytest.mark.parametrize("case", ["eager", "outside_scope", "query_len", "step", "causal", "dcp", "no_metadata"])
def test_ineligible_metadata_is_not_reused(monkeypatch, case):
    speculator, _ = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE if case == "eager" else CUDAGraphMode.FULL, 16, 4)
    overrides = {
        "query_len": {"num_query_per_req": 3},
        "step": {"step": 1},
        "causal": {"causal": True},
        "dcp": {"dcp_local_seq_lens": torch.tensor([5, 10])},
    }.get(case, {})
    if case == "no_metadata":
        speculator._build_attn_metadata.side_effect = None
        speculator._build_attn_metadata.return_value = None
    context = nullcontext() if case == "outside_scope" else speculator._reuse_full_graph_attn_metadata()
    with context:
        _build(speculator, desc, **overrides)
        assert speculator.take_draft_attn_metadata(desc) is None


@pytest.mark.parametrize("failure", ["none", "after_build", "builder"])
def test_scope_clears_metadata_on_return_and_failure(monkeypatch, failure):
    speculator, _ = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    expected = nullcontext() if failure == "none" else pytest.raises(RuntimeError, match="failed")
    with expected, speculator._reuse_full_graph_attn_metadata():
        _build(speculator, desc)
        if failure == "after_build":
            raise RuntimeError("replay failed")
        if failure == "builder":
            speculator._build_attn_metadata.side_effect = RuntimeError("builder failed")
            _build(speculator, desc)
    assert speculator.take_draft_attn_metadata(desc) is None
    with speculator._reuse_full_graph_attn_metadata():
        assert speculator.take_draft_attn_metadata(desc) is None


def test_failed_rebuild_invalidates_previous_metadata_inside_scope(monkeypatch):
    speculator, _ = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    with speculator._reuse_full_graph_attn_metadata():
        _build(speculator, desc)
        speculator._build_attn_metadata.side_effect = RuntimeError("builder failed")
        with pytest.raises(RuntimeError, match="builder failed"):
            _build(speculator, desc)
        assert speculator.take_draft_attn_metadata(desc) is None


def _graph_manager(monkeypatch, speculator, desc, updatable):
    manager = graph_module.DFlashAclGraphManager.__new__(graph_module.DFlashAclGraphManager)
    manager.speculator = speculator
    manager.vllm_config = object()
    manager.update_stream = MagicMock()
    speculator.attn_backends = {"draft.0": object()}
    speculator.dp_size = 1
    speculator.model_state = SimpleNamespace(attn_metadata={"target": object()})
    speculator.speculative_config = object()
    monkeypatch.setattr(graph_module, "use_updatable_graph", lambda backend: updatable)
    monkeypatch.setattr(graph_module, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(graph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(graph_module, "_EXTRA_CTX", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: object()), raising=False)
    monkeypatch.setattr(graph_module, "logger", SimpleNamespace(info_once=MagicMock()), raising=False)
    replay = MagicMock(return_value=object())
    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "run_fullgraph", replay)
    update = MagicMock()
    monkeypatch.setattr(graph_module, "update_full_graph_params", update)
    graph = MagicMock(spec=graph_module.UpdatableGraph)
    manager.graphs = {desc: graph}

    def delivered_metadata():
        if updatable:
            return graph.resolve_tasks.call_args.args[0].context
        return update.call_args.kwargs["draft_attn_metadatas"][0]

    return manager, replay, graph, update, delivered_metadata


@pytest.mark.parametrize("updatable", [False, True])
@pytest.mark.parametrize("has_reuse_api", [False, True])
def test_graph_replay_fallback_preserves_existing_builders(monkeypatch, updatable, has_reuse_api):
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 2)
    metadata = {"draft.0": object()}
    speculator = SimpleNamespace(
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([10, 20])),
        build_draft_attn_metadatas=MagicMock(return_value=[metadata]),
    )
    if has_reuse_api:
        speculator.take_draft_attn_metadata = MagicMock(return_value=None)
    manager, replay, _, _, delivered = _graph_manager(monkeypatch, speculator, desc, updatable)
    assert manager.run_fullgraph(desc) is replay.return_value
    speculator.build_draft_attn_metadatas.assert_called_once_with(2, speculator.input_batch.seq_lens_cpu_upper_bound)
    assert delivered() is metadata


@pytest.mark.parametrize("updatable", [False, True])
def test_graph_replay_outside_proposal_builds_fresh_padded_metadata(monkeypatch, updatable):
    speculator, built = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    manager, _, _, _, delivered = _graph_manager(monkeypatch, speculator, desc, updatable)
    manager.run_fullgraph(desc)
    assert len(built) == 1
    assert delivered() is built[0]
    assert built[0]["draft.0"].actual_seq_lengths_q == [4, 8, 12, 16]
    assert speculator.take_draft_attn_metadata(desc) is None


def _prepare_proposal(monkeypatch, speculator, desc, manager):
    speculator.num_speculative_steps = speculator.num_query_per_req - 1
    speculator.max_model_len = 100
    speculator.max_num_tokens = 16
    speculator.hidden_states = torch.zeros(16, 2)
    speculator.context_positions = torch.zeros(16, dtype=torch.int64)
    speculator._context_slot_mappings = torch.zeros(1, 16, dtype=torch.int32)
    speculator.prepare_context_anchor = MagicMock()
    speculator._prepare_eplb_forward = MagicMock()
    speculator._precompute_context_kv = MagicMock()
    speculator._generate_draft = MagicMock()
    speculator.model = SimpleNamespace(precompute_and_store_context_kv=MagicMock())
    speculator.pcp_manager = None
    speculator.draft_kv_cache_group_id = 0
    speculator.draft_kv_cache_group_ids = []
    speculator.block_tables = SimpleNamespace(slot_mappings=torch.zeros(1, 16, dtype=torch.int32))
    speculator.kv_cache_config = object()
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.query_cudagraph_manager = manager
    speculator.draft_tokens = torch.arange(12).reshape(4, 3)
    speculator.input_batch.num_tokens = 2
    monkeypatch.setattr(upstream_dflash, "dispatch_cg_and_sync_dp", lambda *args, **kwargs: (desc, None))
    monkeypatch.setattr(upstream_dflash, "build_slot_mappings_by_layer", lambda *args: {})


def _propose(speculator, **kwargs):
    return speculator.propose(speculator.input_batch, {}, {}, torch.ones(2, 2), None, *[None] * 6, **kwargs)


@pytest.mark.skipif(vllm_version_is("0.31.0"), reason="Uses main's graph context-KV preparation")
@pytest.mark.parametrize("cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("updatable", [False, True])
def test_real_propose_builds_once_per_full_replay(monkeypatch, cls, updatable):
    speculator, built = _metadata_speculator(monkeypatch, cls)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    manager, replay, graph, update, delivered = _graph_manager(monkeypatch, speculator, desc, updatable)
    _prepare_proposal(monkeypatch, speculator, desc, manager)
    for index in range(2):
        speculator.input_batch.seq_lens_cpu_upper_bound.add_(1)
        result = _propose(speculator)
        torch.testing.assert_close(result, speculator.draft_tokens[:2])
        assert len(built) == index + 1
        assert delivered() is built[index]
        assert built[index]["draft.0"].actual_seq_lengths_q == [4, 8, 12, 16]
        assert speculator.take_draft_attn_metadata(desc) is None
    assert replay.call_count == 2
    assert graph.update.call_count == (2 if updatable else 0)
    assert update.call_count == (0 if updatable else 2)
    speculator._generate_draft.assert_not_called()


@pytest.mark.skipif(vllm_version_is("0.31.0"), reason="Uses main's graph context-KV preparation")
@pytest.mark.parametrize("updatable", [False, True])
@pytest.mark.parametrize("failure_stage", ["before_replay", "replay"])
def test_failed_propose_cannot_leak_metadata_into_next_proposal(monkeypatch, updatable, failure_stage):
    speculator, built = _metadata_speculator(monkeypatch, AscendDFlash2Speculator)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    manager, replay, _, _, delivered = _graph_manager(monkeypatch, speculator, desc, updatable)
    _prepare_proposal(monkeypatch, speculator, desc, manager)
    if failure_stage == "before_replay":
        monkeypatch.setattr(
            upstream_dflash, "build_slot_mappings_by_layer", MagicMock(side_effect=RuntimeError("proposal failed"))
        )
    else:
        replay.side_effect = RuntimeError("proposal failed")
    with pytest.raises(RuntimeError, match="proposal failed"):
        _propose(speculator)
    assert len(built) == 1
    assert speculator.take_draft_attn_metadata(desc) is None

    monkeypatch.setattr(upstream_dflash, "build_slot_mappings_by_layer", lambda *args: {})
    replay.side_effect = None
    _propose(speculator)
    assert len(built) == 2
    assert delivered() is built[1]
    assert speculator.take_draft_attn_metadata(desc) is None


@pytest.mark.skipif(vllm_version_is("0.31.0"), reason="Uses main's graph context-KV preparation")
@pytest.mark.parametrize("profile", [False, True])
def test_real_propose_eager_and_memory_profile_do_not_cache(monkeypatch, profile):
    speculator, built = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 8, 2)
    manager = SimpleNamespace(run_fullgraph=MagicMock())
    _prepare_proposal(monkeypatch, speculator, desc, manager)
    _propose(speculator, dummy_run=profile, skip_attn_for_dummy_run=profile)
    assert len(built) == (0 if profile else 1)
    manager.run_fullgraph.assert_not_called()
    speculator._generate_draft.assert_called_once()
    assert speculator.take_draft_attn_metadata(desc) is None


def test_capture_does_not_enter_proposal_reuse_scope(monkeypatch):
    speculator, built = _metadata_speculator(monkeypatch)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 4)
    manager = graph_module.DFlashAclGraphManager.__new__(graph_module.DFlashAclGraphManager)
    manager.speculator = speculator
    monkeypatch.setattr(graph_module, "communicator_switch", nullcontext)
    monkeypatch.setattr(graph_module, "model_capture_wrapper", lambda *args: nullcontext())

    def capture(self, *args, **kwargs):
        _build(speculator, desc)
        assert speculator.take_draft_attn_metadata(desc) is None

    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "capture", capture)
    manager.capture(*[object()] * 6)
    assert len(built) == 1
