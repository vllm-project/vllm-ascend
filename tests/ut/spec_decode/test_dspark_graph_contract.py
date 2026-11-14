# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.attn_utils as attn_utils_module
import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class _BackendA:
    @staticmethod
    def get_impl_cls():
        return object


def test_flash_attn_graph_replay_uses_device_metadata_without_host_updates(monkeypatch):
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        attn_backends={f"draft.{i}": AscendFlashAttentionBackend for i in range(6)},
        build_draft_attn_metadatas=MagicMock(side_effect=AssertionError("unexpected host metadata rebuild")),
    )
    replay = MagicMock(return_value=object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", replay)
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=24, num_reqs=3)

    assert manager.run_fullgraph(desc) is replay.return_value
    replay.assert_called_once_with(desc)
    manager.speculator.build_draft_attn_metadatas.assert_not_called()


def _speculator(**attributes):
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator.attn_architecture = None
    for name, value in attributes.items():
        setattr(speculator, name, value)
    return speculator


def test_set_attn_preserves_cache_group_order(monkeypatch):
    draft_config = object()
    active_context = []

    @contextmanager
    def config_context(config):
        assert config is draft_config
        active_context.append(config)
        try:
            yield
        finally:
            active_context.pop()

    def parent_set_attn(self, *_args):
        assert active_context == [draft_config]
        self._context_slot_mappings = torch.zeros(2, dtype=torch.int64)
        self.attn_groups = [[SimpleNamespace(backend=_BackendA)]]

    def get_layers(config, layer_type, names):
        assert active_context == [draft_config]
        return {name: SimpleNamespace(get_attn_backend=lambda: _BackendA) for name in names}

    monkeypatch.setattr(speculator_module, "set_current_vllm_config", config_context)
    monkeypatch.setattr(speculator_module.DSparkSpeculator, "set_attn", parent_set_attn)
    monkeypatch.setattr(speculator_module, "get_layers_from_vllm_config", get_layers)
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: draft_config))
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    speculator = _speculator(vllm_config=vllm_config, draft_attn_layer_names={"draft.2", "draft.0"})
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.2", "target.0", "draft.0"])])

    speculator.set_attn(None, cache, None, None, None)

    assert list(speculator.attn_backends) == ["draft.2", "draft.0"]
    assert speculator._context_slot_mappings.dtype == torch.int32
    assert active_context == []


@pytest.mark.parametrize("architecture", ["GQA", "MLA"])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE])
def test_draft_metadata_matches_full_or_actual_query_shape(monkeypatch, architecture, graph_mode):
    metadata = SimpleNamespace(actual_seq_lengths_q=[8, 16, 24, 24])
    layer_metadata = SimpleNamespace(decode=metadata) if architecture == "MLA" else metadata
    parent_build = MagicMock(return_value={"draft.0": layer_metadata})
    metadata_factory = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(DSparkSpeculator, "_build_attn_metadata", parent_build)
    monkeypatch.setattr(speculator_module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(speculator_module, "build_attn_metadata_factory", metadata_factory)
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))

    speculator = _speculator(
        attn_architecture=architecture,
        num_query_per_req=8,
        arange_np=np.arange(5, dtype=np.int32),
        input_buffers=SimpleNamespace(positions=object()),
        _graph_query_start_loc_cpu=torch.arange(5, dtype=torch.int32) * 8,
        _graph_query_start_loc_gpu=torch.arange(5, dtype=torch.int32) * 8,
        vllm_config=SimpleNamespace(parallel_config=object()),
    )
    speculator._prepare_draft_dcp_metadata_inputs = MagicMock(return_value=(None, torch.zeros(4, dtype=torch.bool)))
    batch_desc = BatchExecutionDescriptor(cg_mode=graph_mode, num_tokens=32, num_reqs=4)

    result = speculator._build_uniform_attn_metadata(
        batch_desc=batch_desc,
        num_reqs=3,
        num_query_per_req=8,
        seq_lens_cpu_upper_bound=torch.tensor([8, 8, 8]),
        step=8,
    )

    assert result == {"draft.0": layer_metadata}
    assert parent_build.call_args.kwargs["batch_desc"] is batch_desc
    np.testing.assert_array_equal(parent_build.call_args.kwargs["query_start_loc_np"], [0, 8, 16, 24])
    assert metadata_factory.call_args.args[1] == (32 if graph_mode == CUDAGraphMode.FULL else 24)
    for name in ("query_start_loc_cpu", "query_start_loc_gpu"):
        boundaries = metadata_factory.call_args.kwargs[name]
        if graph_mode == CUDAGraphMode.FULL:
            assert boundaries.tolist() == [0, 8, 16, 24, 32]
        else:
            assert boundaries is None
    assert metadata.actual_seq_lengths_q == [8, 16, 24, 24]


def test_sfa_draft_metadata_passes_through_upstream(monkeypatch):
    metadata = {"draft.0": object()}
    parent_build = MagicMock(return_value=metadata)
    monkeypatch.setattr(DSparkSpeculator, "_build_attn_metadata", parent_build)
    monkeypatch.setattr(speculator_module, "build_attn_metadata_factory", MagicMock(side_effect=AssertionError))
    speculator = _speculator(attn_architecture="SFA", arange_np=np.arange(5, dtype=np.int32))
    batch_desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=4)

    result = speculator._build_uniform_attn_metadata(
        batch_desc=batch_desc,
        num_reqs=3,
        num_query_per_req=8,
        seq_lens_cpu_upper_bound=torch.tensor([8, 8, 8]),
        step=8,
    )

    assert result is metadata
    assert parent_build.call_args.kwargs["batch_desc"] is batch_desc


def test_replay_installs_forward_context_before_accessing_extra_ctx(monkeypatch):
    class _ContextProxy:
        active = False

        def __getattr__(self, name):
            if not self.active:
                raise AssertionError("forward context accessed before installation")
            return self.__dict__.get(name, False)

        def __setattr__(self, name, value):
            if name != "active" and not self.active:
                raise AssertionError("forward context accessed before installation")
            object.__setattr__(self, name, value)

    proxy = _ContextProxy()

    @contextmanager
    def _set_forward_context(*args, **kwargs):
        counts = kwargs["num_tokens_across_dp"]
        assert counts.device.type == "cpu"
        assert counts.tolist() == [7]
        proxy.active = True
        try:
            yield
        finally:
            proxy.active = False

    speculator = SimpleNamespace(
        num_query_per_req=7,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=lambda *args: {"draft.0": object()},
        attn_backends={"draft.0": _BackendA},
        dp_size=1,
        model_state=SimpleNamespace(attn_metadata={}),
        speculative_config=object(),
    )
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: None)
    manager.device = torch.device("cpu")
    manager.vllm_config = object()

    monkeypatch.setattr(aclgraph_module, "_EXTRA_CTX", proxy)
    monkeypatch.setattr(aclgraph_module, "set_forward_context", _set_forward_context)
    monkeypatch.setattr(aclgraph_module, "get_forward_context", lambda: object())
    monkeypatch.setattr(aclgraph_module, "update_full_graph_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", lambda self, desc: "replayed")

    desc = SimpleNamespace(num_tokens=7, num_reqs=1, cg_mode=object())
    assert manager.run_fullgraph(desc) == "replayed"
    assert proxy.active is False


@pytest.mark.parametrize("query_count", [7, 8])
def test_dispatcher_pads_uniform_draft_descriptors(query_count):
    manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[16, 32, 48, 64, 80, 96, 112, 128],
        max_cudagraph_capture_size=128,
    )
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.max_num_reqs = 16
    manager.decode_query_len = query_count
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.varlen_decode = False
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = True
    # vLLM #51700 added ubatch_runner to the cudagraph manager.
    manager.ubatch_runner = None
    manager._init_candidates()

    for num_reqs in (1, 2, 3, 4, 8, 16):
        desc = manager.dispatch(num_reqs, num_reqs * query_count, query_count, 0)
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.num_reqs >= num_reqs
        assert desc.num_tokens == desc.num_reqs * query_count
        assert desc.uniform_token_count == query_count


@pytest.mark.parametrize("module", [speculator_module, dflash_module], ids=["dspark", "dflash"])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_draft_graph_manager_binds_speculator_and_update_stream(monkeypatch, module, enforce_eager):
    cls = module.AscendDSparkSpeculator if module is speculator_module else module.AscendDFlashSpeculator
    parent = DSparkSpeculator if module is speculator_module else dflash_module.DFlashSpeculator
    speculator = cls.__new__(cls)
    speculator.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)
    speculator.update_stream = object()
    modes = []

    def init_manager(self, mode):
        modes.append(mode)
        self.query_cudagraph_manager = SimpleNamespace()

    monkeypatch.setattr(parent, "init_cudagraph_manager", init_manager)
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    assert modes == [CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode.FULL_DECODE_ONLY]
    assert speculator.query_cudagraph_manager.speculator is speculator
    assert speculator.query_cudagraph_manager.update_stream is speculator.update_stream


@pytest.mark.parametrize("needs_capture", [False, True])
def test_dflash_graph_params_use_actual_captured_token_sizes(monkeypatch, needs_capture):
    def initialize(self, *args):
        sizes = (21, 14, 21) if needs_capture else ()
        self._capture_descs = {CUDAGraphMode.FULL: [SimpleNamespace(num_tokens=size) for size in sizes]}

    monkeypatch.setattr(DFlashCudaGraphManager, "__init__", initialize)
    monkeypatch.setattr(DFlashCudaGraphManager, "needs_capture", lambda self: needs_capture)
    set_params = MagicMock()
    monkeypatch.setattr(aclgraph_module, "set_draft_graph_params", set_params)

    manager = DFlashAclGraphManager(object(), torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, 7)

    assert manager.capture_sizes == ([14, 21] if needs_capture else [])
    if needs_capture:
        set_params.assert_called_once_with([14, 21])
    else:
        set_params.assert_not_called()


@pytest.mark.parametrize("valid_graph", [False, True])
def test_dflash_updatable_replay_resolves_current_metadata_before_replay(monkeypatch, valid_graph):
    events = []
    metadata, tasks, result, stream = object(), object(), object(), object()
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=14, num_reqs=2)
    graph = MagicMock(spec=UpdatableGraph)
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        attn_backends={"draft.0": _BackendA},
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([8])),
        build_draft_attn_metadatas=MagicMock(return_value=[metadata]),
    )
    manager.graphs = {desc: graph if valid_graph else object()}
    manager.update_stream = SimpleNamespace(wait_stream=lambda current: events.append(("wait", current)))
    monkeypatch.setattr(aclgraph_module, "use_updatable_graph", lambda backend: True)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    context_source = MagicMock(return_value=metadata)
    monkeypatch.setattr(aclgraph_module, "ContextSource", context_source)

    def replay(self, batch_desc):
        assert batch_desc is desc
        events.append(("replay", desc))
        return result

    def resolve_tasks(source):
        events.append(("resolve", source))
        return tasks

    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", replay)
    if not valid_graph:
        with pytest.raises(AssertionError):
            manager.run_fullgraph(desc)
        assert events == []
    else:
        graph.resolve_tasks.side_effect = resolve_tasks
        graph.update.side_effect = lambda update_stream, resolved: events.append(("update", resolved))
        assert manager.run_fullgraph(desc) is result
        assert events == [("resolve", metadata), ("wait", stream), ("replay", desc), ("update", tasks)]
        context_source.assert_called_once_with(metadata)
        graph.update.assert_called_once_with(manager.update_stream, tasks)
    manager.speculator.build_draft_attn_metadatas.assert_called_once_with(
        2, manager.speculator.input_batch.seq_lens_cpu_upper_bound
    )


@pytest.mark.parametrize("architecture", ["GQA", "MLA"])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE])
@pytest.mark.parametrize("query_count", [1, 3, 8])
def test_draft_builder_receives_consistent_request_boundaries(monkeypatch, architecture, graph_mode, query_count):
    capacity = 6
    uniform = torch.arange(capacity + 1, dtype=torch.int32) * query_count
    device_uniform = uniform.clone()
    inputs = SimpleNamespace(
        positions=torch.arange(capacity * query_count),
        query_start_loc=torch.zeros(capacity + 1, dtype=torch.int32),
        seq_lens=torch.zeros(capacity, dtype=torch.int32),
    )
    tables = torch.arange(capacity * 2).reshape(capacity, 2)
    slots = torch.full((1, capacity * query_count), -1, dtype=torch.int32)
    speculator = _speculator(
        attn_architecture=architecture,
        num_query_per_req=query_count,
        arange_np=np.arange(capacity + 1, dtype=np.int32),
        _graph_query_start_loc_cpu=uniform,
        _graph_query_start_loc_gpu=device_uniform,
        input_buffers=inputs,
        block_tables=SimpleNamespace(input_block_tables=[tables], slot_mappings=slots),
        dcp_size=1,
        max_model_len=1024,
        draft_max_seq_len=1024,
        attn_groups=[],
        kv_cache_config=object(),
        draft_is_prefilling=torch.zeros(capacity, dtype=torch.bool),
        vllm_config=SimpleNamespace(parallel_config=object()),
    )
    speculator._prepare_draft_dcp_metadata_inputs = MagicMock(
        return_value=(None, torch.zeros(capacity, dtype=torch.bool))
    )
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    captured = {}

    def build_metadata(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"draft.0": kwargs}

    monkeypatch.setattr(attn_utils_module, "build_attn_metadata", build_metadata)
    desc = BatchExecutionDescriptor(cg_mode=graph_mode, num_tokens=capacity * query_count, num_reqs=capacity)
    # Capture-sized input followed by smaller replay batches reuses the same buffers.
    for real_requests in (capacity, 4, 1):
        inputs.query_start_loc.copy_(uniform.clamp(max=real_requests * query_count))
        inputs.seq_lens.zero_()
        inputs.seq_lens[:real_requests] = 17 + query_count
        slots.fill_(-1)
        slots[:, : real_requests * query_count] = 42
        original_query = inputs.query_start_loc.clone()
        original_slots = slots.clone()
        speculator._build_uniform_attn_metadata(
            batch_desc=desc,
            num_reqs=real_requests,
            num_query_per_req=query_count,
            seq_lens_cpu_upper_bound=torch.full((real_requests,), 17, dtype=torch.int32),
            step=query_count,
        )
        expected = uniform if graph_mode == CUDAGraphMode.FULL else original_query
        torch.testing.assert_close(captured["query_start_loc_cpu"], expected)
        torch.testing.assert_close(captured["query_start_loc_gpu"], expected)
        assert captured["num_tokens"] == expected[-1].item()
        assert captured["num_reqs"] == capacity
        assert captured["seq_lens"].tolist() == [17 + query_count] * real_requests + [0] * (capacity - real_requests)
        assert captured["seq_lens_cpu_upper_bound"].tolist() == captured["seq_lens"].tolist()
        assert captured["block_tables"][0].data_ptr() == tables.data_ptr()
        assert captured["block_tables"][0].shape[0] == capacity
        torch.testing.assert_close(inputs.query_start_loc, original_query)
        torch.testing.assert_close(slots, original_slots)
        if graph_mode == CUDAGraphMode.FULL:
            assert captured["query_start_loc_gpu"].data_ptr() == device_uniform.data_ptr()
