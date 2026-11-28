# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.config.attention import AttentionConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.aclgraph as aclgraph_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from vllm_ascend.attention.attention_v1 import AscendKimiK31GQADraftAttentionBackend
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class _BackendA:
    @staticmethod
    def get_impl_cls():
        return object


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

    speculator.set_attn(None, cache, SimpleNamespace(cp_size=1), None, None)

    assert list(speculator.attn_backends) == ["draft.2", "draft.0"]
    assert speculator._context_slot_mappings.dtype == torch.int32
    assert active_context == []


def test_set_attn_shares_buffers_with_dcp8_target_using_local_draft_slots(monkeypatch):
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=8, cp_kv_cache_interleave_size=1664),
        attention_config=AttentionConfig(use_non_causal=False),
        cache_config=SimpleNamespace(block_size=1664),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k31")),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=SimpleNamespace(architectures=["Qwen3DSparkModel"])
        ),
    )
    speculator = _speculator(vllm_config=target_config, requires_non_causal=False, draft_attn_layer_names={"draft.swa"})
    speculator._init_dcp()
    assert not speculator.use_dcp and speculator.dcp_manager is None
    tables = SimpleNamespace(
        cp_size=8,
        cp_rank=6,
        cp_interleave=1664,
        slot_mappings=torch.zeros((2, 8), dtype=torch.int32),
        input_block_tables=(object(), object()),
    )
    backend = AscendKimiK31GQADraftAttentionBackend

    def parent_set_attn(self, model_state, config, block_tables, *args):
        assert get_current_vllm_config().parallel_config.decode_context_parallel_size == 1
        assert (block_tables.cp_size, block_tables.cp_rank, block_tables.cp_interleave) == (1, 0, 1)
        assert block_tables.slot_mappings is tables.slot_mappings
        assert block_tables.input_block_tables is tables.input_block_tables
        self._context_slot_mappings = torch.zeros(8, dtype=torch.int64)
        self.attn_groups = [[], [SimpleNamespace(backend=backend)]]

    monkeypatch.setattr(speculator_module.DSparkSpeculator, "set_attn", parent_set_attn)
    monkeypatch.setattr(
        speculator_module,
        "get_layers_from_vllm_config",
        lambda cfg, kind, names: {name: SimpleNamespace(get_attn_backend=lambda: backend) for name in names},
    )
    prepare = MagicMock()
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.dflash.speculator.prepare_dflash_inputs_triton", prepare)
    monkeypatch.setattr(speculator_module.dflash_speculator, "prepare_dflash_inputs", None)
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["target.0", "draft.swa"])])
    with set_current_vllm_config(target_config):
        speculator.set_attn(None, cache, tables, None, None)
        assert get_current_vllm_config() is target_config
    assert (tables.cp_size, tables.cp_rank, tables.cp_interleave) == (8, 6, 1664)
    assert speculator.attn_architecture == "GQA"
    assert speculator._context_slot_mappings.dtype == torch.int32

    prepare_inputs = speculator_module.dflash_speculator.prepare_dflash_inputs
    prepare_inputs(*([None] * 16), object(), 128, 0, 1, 1, 0, 3, 3, 1, 8, 4096)
    assert prepare.call_args.args[17:21] == (128, 0, 1, 1)
    assert prepare.call_args.kwargs["kv_cache_block_size"] == 128


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
    assert metadata.actual_seq_lengths_q == ([8, 16, 24, 32] if graph_mode == CUDAGraphMode.FULL else [8, 16, 24, 24])


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


@pytest.mark.parametrize("query_count", [1, 7, 8])
@pytest.mark.parametrize("num_reqs_padded", [1, 2, 4, 16])
def test_update_draft_metadata_refreshes_all_padded_query_lengths(query_count, num_reqs_padded):
    speculator = _speculator(num_query_per_req=query_count)
    metadata = {
        name: SimpleNamespace(actual_seq_lengths_q=[query_count], seq_lens=object()) for name in ("draft.2", "draft.0")
    }
    seq_lens = {name: value.seq_lens for name, value in metadata.items()}

    assert speculator._update_draft_attn_metadata(metadata, num_reqs_padded) is metadata
    for name, value in metadata.items():
        assert value.actual_seq_lengths_q == [query_count * (i + 1) for i in range(num_reqs_padded)]
        assert value.seq_lens is seq_lens[name]


def test_kimi_draft_logs_capture_and_first_serving_replay(monkeypatch):
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=96, num_reqs=32, uniform_token_count=3)
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.is_kimi_gqa_draft = True
    manager._logged_serving_replay = False
    manager.graphs = {}
    manager.speculator = SimpleNamespace(
        model=torch.nn.Identity(),
        _serving_draft_batch=False,
        input_batch=SimpleNamespace(num_reqs=5, seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=lambda *args: [{}],
        attn_backends={"draft.0": AscendKimiK31GQADraftAttentionBackend},
    )
    log = MagicMock()
    monkeypatch.setattr(aclgraph_module.logger, "info", log)
    monkeypatch.setattr(aclgraph_module, "communicator_switch", nullcontext)
    monkeypatch.setattr(DFlashCudaGraphManager, "capture", lambda *args: None)
    manager.capture(None, None, None, [], None, 4096)
    log.assert_not_called()
    monkeypatch.setattr(DFlashCudaGraphManager, "capture", lambda *args: manager.graphs.update({desc: object()}))
    manager.capture(None, None, None, [], None, 4096)
    log.assert_called_once_with("Kimi K3.1 DSpark draft graph stage=capture captured_token_sizes=%s", [96])
    log.reset_mock()

    monkeypatch.setattr(aclgraph_module, "use_updatable_graph", lambda backend: True)
    replay = MagicMock(side_effect=["dummy", RuntimeError("update failed"), "first", "second"])
    monkeypatch.setattr(manager, "_updatable_graph_replay", replay)
    assert manager.run_fullgraph(desc) == "dummy"
    manager.speculator._serving_draft_batch = True
    with pytest.raises(RuntimeError, match="update failed"):
        manager.run_fullgraph(desc)
    log.assert_not_called()
    assert manager.run_fullgraph(desc) == "first"
    assert manager.run_fullgraph(desc) == "second"
    log.assert_called_once_with(
        "Kimi K3.1 DSpark draft graph stage=replay mode=FULL requests=%d padded_requests=%d tokens=%d", 5, 32, 96
    )


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
    manager.is_kimi_gqa_draft = False
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


@pytest.mark.parametrize(
    "query_count,max_reqs,buckets,expected_mode",
    [
        (7, 16, list(range(16, 129, 16)), CUDAGraphMode.FULL),
        (8, 16, list(range(16, 129, 16)), CUDAGraphMode.FULL),
        (3, 32, [128], CUDAGraphMode.NONE),
        (4, 32, [128], CUDAGraphMode.FULL),
    ],
)
def test_dispatcher_pads_uniform_draft_descriptors(query_count, max_reqs, buckets, expected_mode):
    manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=buckets,
        max_cudagraph_capture_size=128,
    )
    manager.vllm_config = SimpleNamespace(speculative_config=None)
    manager.max_num_reqs = max_reqs
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

    for num_reqs in (1, 2, 3, 4, 8, max_reqs):
        desc = manager.dispatch(num_reqs, num_reqs * query_count, query_count, 0)
        assert desc.cg_mode == expected_mode
        assert desc.num_reqs >= num_reqs
        assert desc.num_tokens == desc.num_reqs * query_count
        if expected_mode == CUDAGraphMode.FULL:
            assert desc.uniform_token_count == query_count


@pytest.mark.parametrize(
    "max_reqs,buckets,mode,eager,kimi,expected",
    [
        (32, [128], CUDAGraphMode.FULL_DECODE_ONLY, False, True, [96]),
        (40, [128], CUDAGraphMode.FULL_DECODE_ONLY, False, True, [120]),
        (43, [128], CUDAGraphMode.FULL_DECODE_ONLY, False, True, [126]),
        (32, [8, 16, 32, 64, 96, 128], CUDAGraphMode.FULL_DECODE_ONLY, False, True, [9, 18, 33, 66, 96]),
        (32, [2], CUDAGraphMode.FULL_DECODE_ONLY, False, True, []),
        (32, [], CUDAGraphMode.FULL_DECODE_ONLY, False, True, []),
        (32, [128], CUDAGraphMode.NONE, False, True, []),
        (32, [128], CUDAGraphMode.FULL_DECODE_ONLY, True, True, []),
        (32, [128], CUDAGraphMode.FULL_DECODE_ONLY, False, False, []),
    ],
)
def test_kimi_gqa_draft_capture_buckets(monkeypatch, max_reqs, buckets, mode, eager, kimi, expected):
    # Exercise the upstream speculator's backend capability check, Ascend's
    # manager construction and the production descriptor lookup without an NPU.
    monkeypatch.setattr(speculator_module.dflash_speculator, "DFlashCudaGraphManager", DFlashAclGraphManager)
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None)
    register = MagicMock()
    monkeypatch.setattr(aclgraph_module, "set_draft_graph_params", register)
    speculative = SimpleNamespace(
        method="dspark",
        enforce_eager=eager,
        uses_dynamic_speculative_decoding=lambda: False,
        draft_model_config=SimpleNamespace(architectures=["Qwen3DSparkModel"]),
    )
    target_sizes = buckets.copy()
    compilation = SimpleNamespace(cudagraph_capture_sizes=target_sizes, max_cudagraph_capture_size=128)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k31" if kimi else "qwen3")),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_reqs, max_num_batched_tokens=512),
        compilation_config=compilation,
        speculative_config=speculative,
    )
    spec = _speculator(
        vllm_config=config,
        speculative_config=speculative,
        num_query_per_req=3,
        max_num_reqs=max_reqs,
        device=torch.device("cpu"),
        update_stream=object(),
        attn_cg_support=SimpleNamespace(min_cg_support=AttentionCGSupport.UNIFORM_BATCH),
    )
    spec.init_cudagraph_manager(mode)
    manager = spec.query_cudagraph_manager
    assert manager.is_kimi_gqa_draft == kimi
    assert manager.capture_sizes == expected
    assert compilation.cudagraph_capture_sizes is target_sizes and target_sizes == buckets
    assert compilation.max_cudagraph_capture_size == 128
    if expected:
        register.assert_called_once_with(expected)
    else:
        register.assert_not_called()
    # Only capture_state and NPU graphs are mocked; candidates and dispatch are real.
    manager._graphs_captured = True
    for reqs in range(1, max_reqs + 1):
        desc = manager.dispatch(reqs, 3 * reqs, 3, 0)
        bucket = next((size for size in expected if size >= 3 * reqs), None)
        assert desc.cg_mode == (CUDAGraphMode.FULL if bucket else CUDAGraphMode.NONE)
        if bucket:
            assert (desc.num_tokens, desc.num_reqs) == (bucket, bucket // 3)
