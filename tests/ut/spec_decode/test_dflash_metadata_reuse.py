# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph_module
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


def test_dflash_graph_reuses_same_proposal_metadata_without_rebuilding(monkeypatch):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.num_query_per_req = 4
    group_cache = object()
    metadata = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8, 8, 8], block_tables=group_cache)}
    speculator._prepared_draft_attn_metadata = (4, metadata)
    build = MagicMock(side_effect=AssertionError("Same proposal must reuse metadata"))
    monkeypatch.setattr(speculator, "_build_uniform_attn_metadata", build)
    result = speculator.build_draft_attn_metadatas(4, object())
    assert result[0] is metadata
    assert metadata["draft"].actual_seq_lengths_q == [4, 8, 12, 16]
    assert metadata["draft"].block_tables is group_cache
    build.assert_not_called()


@pytest.mark.parametrize("cached_count", [None, 2])
def test_dflash_graph_builds_metadata_when_cache_missing_or_shape_changes(monkeypatch, cached_count):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.num_query_per_req = 4
    speculator.input_batch = SimpleNamespace(num_reqs=2)
    speculator._group_causal = False
    stale = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8])}
    speculator._prepared_draft_attn_metadata = None if cached_count is None else (cached_count, stale)
    metadata = {"draft": SimpleNamespace(actual_seq_lengths_q=[4, 8, 8, 8])}
    build = MagicMock(return_value=metadata)
    monkeypatch.setattr(speculator, "_build_uniform_attn_metadata", build)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", lambda: nullcontext())
    bounds = object()
    result = speculator.build_draft_attn_metadatas(4, bounds)
    assert result[0] is metadata
    assert metadata["draft"].actual_seq_lengths_q == [4, 8, 12, 16]
    assert stale["draft"].actual_seq_lengths_q == [4, 8]
    build.assert_called_once()
    assert build.call_args.kwargs["seq_lens_cpu_upper_bound"] is bounds
    assert build.call_args.kwargs["batch_desc"].num_tokens == 16


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("padded_count", [None, 4])
def test_dflash_caches_full_metadata_after_upstream_builder_refresh(monkeypatch, mode, padded_count):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator._prepared_draft_attn_metadata = None
    descriptor = SimpleNamespace(cg_mode=mode, num_reqs=padded_count)
    query, bounds, local_lengths = object(), object(), object()
    metadata = {"draft": object()}
    calls = []

    def build(self, *args):
        assert self is speculator
        calls.append(args)
        return metadata

    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", build)
    result = speculator._build_attn_metadata(2, descriptor, query, bounds, 4, False, local_lengths)
    assert calls == [(2, descriptor, query, bounds, 4, False, local_lengths)]
    assert result is metadata
    if mode == CUDAGraphMode.FULL:
        assert speculator._prepared_draft_attn_metadata == (padded_count or 2, metadata)
    else:
        assert speculator._prepared_draft_attn_metadata is None


def test_proposal_invalidates_previous_metadata_before_upstream_execution(monkeypatch):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator._prepared_draft_attn_metadata = (4, object())
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", lambda: nullcontext())
    batch, result = object(), object()

    def propose(self, *args, **kwargs):
        assert self.input_batch is batch
        assert self._prepared_draft_attn_metadata is None
        return result

    monkeypatch.setattr(DFlashSpeculator, "propose", propose)
    assert speculator.propose(batch, *([None] * 10)) is result


@pytest.mark.parametrize("num_reqs,padded", [(0, 4), (1, 1), (1, 4), (2, 4)])
def test_dflash_cpu_mirror_reads_valid_rows_and_zeroes_padding(num_reqs, padded):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor([73, 22666, 99999, 99999], dtype=torch.int32))
    actual = spec._prepare_draft_seq_lens_cpu(num_reqs, padded)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int32
    assert actual.tolist() == [73, 22666][:num_reqs] + [0] * (padded - num_reqs)
    assert spec.input_buffers.seq_lens.tolist() == [73, 22666, 99999, 99999]


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_dflash_metadata_shares_exact_lengths_and_refreshes_after_rejection(monkeypatch, mode):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._use_cpu_seq_lens = True
    spec._prepared_draft_attn_metadata = None
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([73, 22666, 99999, 99999], dtype=torch.int32),
        positions=torch.arange(32),
    )
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    descriptor = SimpleNamespace(cg_mode=mode, num_reqs=4, num_tokens=32)
    query = np.array([0, 8, 16], dtype=np.int32)
    upper_bound = torch.tensor([99, 99999], dtype=torch.int32)
    supplied = []

    @contextmanager
    def factory(positions, pad, **kwargs):
        assert positions is spec.input_buffers.positions
        assert pad == (32 if mode == CUDAGraphMode.FULL else 16)
        assert kwargs["seq_lens_cpu_is_exact"]
        supplied.append(kwargs["seq_lens_cpu"])
        yield

    monkeypatch.setattr(dflash_module, "build_attn_metadata_factory", factory)
    parent = MagicMock(return_value={"draft": object()})
    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", parent)
    spec._build_attn_metadata(2, descriptor, query, upper_bound, 8, False)
    spec.input_buffers.seq_lens[:2].copy_(torch.tensor([74, 22674]))
    spec._build_attn_metadata(2, descriptor, query, upper_bound, 8, False)
    assert [value.tolist() for value in supplied] == [[73, 22666, 0, 0], [74, 22674, 0, 0]]
    assert upper_bound.tolist() == [99, 99999]
    assert parent.call_count == 2


def test_prefetched_lengths_wait_on_copy_event_and_clear_old_padding():
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._draft_seq_lens_cpu = torch.tensor([73, 81, 99, 99], dtype=torch.int32)
    spec._draft_seq_lens_copy_count = 2
    spec._draft_seq_lens_copy_event = MagicMock()
    # A copy from the compute stream here would deadlock after early replay.
    spec.input_buffers = SimpleNamespace(seq_lens=MagicMock(side_effect=AssertionError("Unexpected D2H")))
    mirror = spec._prepare_draft_seq_lens_cpu(2, 4)
    assert mirror.tolist() == [73, 81, 0, 0]
    assert mirror.data_ptr() == spec._draft_seq_lens_cpu.data_ptr()
    assert spec._draft_seq_lens_copy_count is None
    spec._draft_seq_lens_copy_event.synchronize.assert_called_once_with()


def test_snapshot_is_started_after_grouped_input_launch():
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._in_draft_proposal = spec._use_cpu_seq_lens = True
    spec._draft_inputs_prepared = False
    spec.draft_kv_cache_group_ids = [2, 5]
    spec.input_batch = SimpleNamespace(num_reqs=3)
    spec._start_draft_seq_lens_copy = MagicMock()
    spec._on_draft_inputs_prepared()
    spec._start_draft_seq_lens_copy.assert_called_once_with(3)
    assert spec._draft_inputs_prepared


@pytest.mark.parametrize("use_cpu_lengths", [False, True])
def test_grouped_preparation_launches_once_per_proposal_and_refreshes_views(monkeypatch, use_cpu_lengths):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._in_draft_proposal = True
    spec._use_cpu_seq_lens = use_cpu_lengths
    spec._draft_inputs_prepared = False
    spec.draft_kv_cache_group_ids = [2, 0]
    spec.input_batch = SimpleNamespace(num_reqs=3)
    spec._start_draft_seq_lens_copy = MagicMock()
    spec.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((3, 16), dtype=torch.int32),
        input_block_tables=[torch.full((4, 8), i, dtype=torch.int32) for i in range(3)],
        kernel_block_sizes=[64, 128, 32],
    )
    spec._context_slot_mappings = torch.empty((2, 16), dtype=torch.int32)
    launch = MagicMock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", launch)
    prepare = dflash_module.prepare_dflash_inputs_factory(
        128, spec._on_draft_inputs_prepared, spec._get_draft_input_groups
    )
    prepare("common_inputs")
    prepare("common_inputs")
    launch.assert_called_once()
    groups = launch.call_args.kwargs["kv_groups"]
    for index, gid in enumerate(spec.draft_kv_cache_group_ids):
        assert groups[index][0].data_ptr() == spec.block_tables.slot_mappings[gid].data_ptr()
        assert groups[index][1].data_ptr() == spec._context_slot_mappings[index].data_ptr()
        assert groups[index][2] is spec.block_tables.input_block_tables[gid]
        assert groups[index][3] == spec.block_tables.kernel_block_sizes[gid]
    assert spec._start_draft_seq_lens_copy.call_count == int(use_cpu_lengths)
    # A new proposal must pick up current table views and launch again.
    spec._draft_inputs_prepared = False
    spec.block_tables.input_block_tables[2] = torch.full((4, 8), 99, dtype=torch.int32)
    prepare("next_proposal")
    assert launch.call_count == 2
    assert launch.call_args.kwargs["kv_groups"][0][2] is spec.block_tables.input_block_tables[2]
    # Dummy/standalone calls outside a proposal retain the single-group API.
    spec._in_draft_proposal = False
    prepare("standalone")
    assert launch.call_count == 3
    assert launch.call_args.kwargs["kv_groups"] is None


def test_full_layout_reuse_refreshes_lengths_and_each_groups_physical_cache(monkeypatch):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._use_cpu_seq_lens = spec._reuse_draft_layout = True
    spec._prepared_draft_attn_metadata = spec._draft_metadata_template = None
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor([73, 81, 999, 999]), query_start_loc=torch.arange(5), positions=torch.arange(32)
    )
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    spec.attn_groups = [[SimpleNamespace(layer_names=["a", "b"])], [SimpleNamespace(layer_names=["c"])]]
    spec.block_tables = SimpleNamespace(
        input_block_tables=[torch.full((4, 2), i) for i in (1, 2)],
        slot_mappings=torch.arange(64).reshape(2, 32),
    )
    original = {
        "a": AscendMetadata(query_start_loc=torch.arange(5), actual_seq_lengths_q=[8, 16, 16, 16]),
        "c": AscendMetadata(query_start_loc=torch.arange(5), actual_seq_lengths_q=[8, 16, 16, 16]),
    }
    original["b"] = original["a"]
    parent = MagicMock(return_value=original)
    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", parent)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_factory", lambda *a, **kw: nullcontext())
    desc = SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_reqs=4, num_tokens=32)
    args = (2, desc, np.array([0, 8, 16]), torch.tensor([100, 100]), 8, False)
    spec._build_attn_metadata(*args)
    # Simulate rejection, a new block allocation and changed slot mappings.
    spec.input_buffers.seq_lens[:2] = torch.tensor([60, 82])
    spec.block_tables.input_block_tables = [torch.full((4, 2), i) for i in (10, 20)]
    spec.block_tables.slot_mappings = torch.arange(64, 128).reshape(2, 32)
    order = []
    refresh = spec._refresh_draft_layout
    prepare_lengths = spec._prepare_draft_seq_lens_cpu

    def refresh_layout(*args):
        order.append("layout")
        return refresh(*args)

    def prepare_current_lengths(*args):
        order.append("length_wait")
        return prepare_lengths(*args)

    monkeypatch.setattr(spec, "_refresh_draft_layout", refresh_layout)
    monkeypatch.setattr(spec, "_prepare_draft_seq_lens_cpu", prepare_current_lengths)
    refreshed = spec._build_attn_metadata(*args)
    assert order == ["layout", "length_wait"]
    parent.assert_called_once()
    assert refreshed["a"] is refreshed["b"]
    assert refreshed["a"] is not refreshed["c"]
    assert refreshed["a"].query_start_loc is original["a"].query_start_loc
    for name, index in (("a", 0), ("c", 1)):
        assert refreshed[name].seq_lens_list == [60, 82, 0, 0]
        assert refreshed[name].block_tables.data_ptr() == spec.block_tables.input_block_tables[index].data_ptr()
        assert refreshed[name].slot_mapping.data_ptr() == spec.block_tables.slot_mappings[index].data_ptr()
        assert refreshed[name].qfa_metadata_cache == {}
    # A different query layout must rebuild instead of reusing the old shape.
    spec._build_attn_metadata(2, desc, np.array([0, 4, 16]), args[3], 8, False)
    assert parent.call_count == 2


def test_graph_replay_precedes_deferred_build_but_update_stream_wait_precedes_replay(monkeypatch):
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    calls = []
    graph = MagicMock()
    monkeypatch.setattr(graph_module, "UpdatableGraph", MagicMock)
    metadata = {"draft": object()}
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=4)
    manager.graphs = {desc: graph}

    def build_metadata(*args):
        calls.append("build")
        return [metadata]

    def replay(*args):
        calls.append("replay")
        return 7

    def resolve(source):
        calls.append("resolve")
        return iter(())

    manager.speculator = SimpleNamespace(
        _deferred_draft_attn_metadata=object(),
        attn_backends={"draft": object()},
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=build_metadata,
    )
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: calls.append("wait_inputs"))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "run_fullgraph", replay)
    graph.iter_resolved_tasks.side_effect = resolve
    graph.update.side_effect = lambda *a: calls.append("update")
    assert manager.run_fullgraph(desc) == 7
    assert calls == ["wait_inputs", "replay", "build", "resolve", "update"]


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_deferred_metadata_requires_prefetch_and_captured_fia_events(monkeypatch, mode):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec._in_draft_proposal = spec._reuse_draft_layout = True
    spec._materializing_draft_metadata = False
    spec._draft_seq_lens_copy_count = 2
    desc = BatchExecutionDescriptor(cg_mode=mode, num_tokens=32, num_reqs=4)
    graph = MagicMock()
    graph.tasks = [object()]
    monkeypatch.setattr(dflash_module, "UpdatableGraph", MagicMock)
    spec.query_cudagraph_manager = SimpleNamespace(graphs={desc: graph})
    assert spec._can_defer_draft_metadata(desc) == (mode == CUDAGraphMode.FULL)
    graph.tasks = []
    assert not spec._can_defer_draft_metadata(desc)
    graph.tasks = [object()]
    spec._draft_seq_lens_copy_count = None
    assert not spec._can_defer_draft_metadata(desc)


def test_deferred_build_materializes_current_proposal_once(monkeypatch):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.num_query_per_req = 8
    spec._prepared_draft_attn_metadata = None
    spec._materializing_draft_metadata = False
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=4)
    args = (2, desc, np.array([0, 8, 16]), torch.tensor([100, 100]), 8, False, None)
    spec._deferred_draft_attn_metadata = args
    metadata = {"draft": SimpleNamespace(actual_seq_lengths_q=None)}

    def build(*actual):
        assert spec._materializing_draft_metadata
        assert actual == args
        assert spec._deferred_draft_attn_metadata is None
        spec._prepared_draft_attn_metadata = (4, metadata)

    builder = MagicMock(side_effect=build)
    monkeypatch.setattr(spec, "_build_attn_metadata", builder)
    assert spec.build_draft_attn_metadatas(4, args[3]) == [metadata]
    assert metadata["draft"].actual_seq_lengths_q == [8, 16, 24, 32]
    assert spec.build_draft_attn_metadatas(4, args[3])[0] is metadata
    builder.assert_called_once()
    assert not spec._materializing_draft_metadata


@pytest.mark.parametrize("has_context_prefix", [False, True])
def test_capture_preserves_upstream_context_kv_callback(monkeypatch, has_context_prefix):
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = object()
    parent = MagicMock()
    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "capture", parent)
    monkeypatch.setattr(graph_module, "communicator_switch", lambda: nullcontext())
    monkeypatch.setattr(graph_module, "model_capture_wrapper", lambda *args: nullcontext())
    callback = MagicMock() if has_context_prefix else None
    manager.capture(*([None] * 6), False, "Capture draft", precompute_context_kv=callback)
    parent.assert_called_once()
    assert parent.call_args.kwargs["progress_bar_desc"] == "Capture draft"
    if has_context_prefix:
        assert parent.call_args.kwargs["precompute_context_kv"] is callback
    else:
        assert "precompute_context_kv" not in parent.call_args.kwargs
