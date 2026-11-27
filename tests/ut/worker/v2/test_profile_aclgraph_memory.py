# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.compilation import acl_graph
from vllm_ascend.worker import encoder_acl_graph
from vllm_ascend.worker.v2 import model_runner as runner_module
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.utils import C8KVBlockZeroer


class _DummyACLWrapper:
    def __init__(self, graph_pool):
        self.graph_pool = graph_pool
        self.concrete_aclgraph_entries = {}
        self.first_run_finished = False

    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()
        self.first_run_finished = False


def test_profile_cudagraph_memory_swaps_acl_pools_and_cleans_up(monkeypatch):
    original_pool = object()
    throwaway_pool = object()
    wrapper = _DummyACLWrapper(original_pool)
    acl_graph._acl_graph_wrappers.clear()
    acl_graph._acl_graph_wrappers.add(wrapper)

    acl_graph._graph_params = object()  # type: ignore[assignment]
    acl_graph._draft_graph_params = object()  # type: ignore[assignment]
    acl_graph._draft_graph_prefill_params = object()  # type: ignore[assignment]
    encoder_acl_graph._encoder_graph_params = object()  # type: ignore[assignment]

    delegate_zeroer = object()
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.cache_config = SimpleNamespace(enable_extensible_kv_cache=False)
    runner._c8_mxfp_v_scale_caches = [("scale", "cache", 1)]
    runner.kv_block_zeroer = C8KVBlockZeroer(delegate_zeroer, runner._c8_mxfp_v_scale_caches, lambda *_: None)
    runner.kvpp = object()
    runner.model_state = SimpleNamespace(kvpp_runtime=runner.kvpp, device_metadata=object())

    observed = {}

    def fake_parent_profile(self):
        observed["wrapper_pool"] = wrapper.graph_pool
        observed["platform_pool"] = runner_module.current_platform.graph_pool_handle()
        observed["graph_params_before_capture"] = acl_graph._graph_params
        wrapper.concrete_aclgraph_entries["desc"] = "captured_entry"
        wrapper.first_run_finished = True
        acl_graph._graph_params = object()  # type: ignore[assignment]
        encoder_acl_graph._encoder_graph_params = object()  # type: ignore[assignment]
        return 123456

    empty_cache = MagicMock()
    monkeypatch.setattr(runner_module.current_platform, "graph_pool_handle", lambda: throwaway_pool, raising=False)
    monkeypatch.setattr(runner_module, "torch_cuda_wrapper", nullcontext)
    monkeypatch.setattr(runner_module, "communicator_switch", nullcontext)
    monkeypatch.setattr(GPUModelRunner, "profile_cudagraph_memory", fake_parent_profile)
    monkeypatch.setattr(torch.npu, "empty_cache", empty_cache)

    estimate = runner.profile_cudagraph_memory()

    assert estimate == 123456
    assert observed["wrapper_pool"] is throwaway_pool
    assert observed["platform_pool"] is throwaway_pool
    assert observed["graph_params_before_capture"] is None
    assert wrapper.graph_pool is original_pool
    assert wrapper.concrete_aclgraph_entries == {}
    assert wrapper.first_run_finished is False
    assert acl_graph._graph_params is None
    assert acl_graph._draft_graph_params is None
    assert acl_graph._draft_graph_prefill_params is None
    assert encoder_acl_graph._encoder_graph_params is None
    assert runner._c8_mxfp_v_scale_caches == []
    assert runner.kv_block_zeroer is delegate_zeroer
    assert runner.kvpp is None
    assert runner.model_state.kvpp_runtime is None
    assert runner.model_state.device_metadata is None
    empty_cache.assert_called_once_with()


def test_profile_cudagraph_memory_restores_state_on_failure(monkeypatch):
    original_pool = object()
    throwaway_pool = object()
    wrapper = _DummyACLWrapper(original_pool)
    wrapper.concrete_aclgraph_entries["stale"] = "entry"
    wrapper.first_run_finished = True
    acl_graph._acl_graph_wrappers.clear()
    acl_graph._acl_graph_wrappers.add(wrapper)

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.cache_config = SimpleNamespace(enable_extensible_kv_cache=False)
    runner.model_state = SimpleNamespace(kvpp_runtime="kvpp", device_metadata="meta")

    def boom(self):
        acl_graph._graph_params = object()  # type: ignore[assignment]
        raise RuntimeError("capture failed")

    monkeypatch.setattr(runner_module.current_platform, "graph_pool_handle", lambda: throwaway_pool, raising=False)
    monkeypatch.setattr(runner_module, "torch_cuda_wrapper", nullcontext)
    monkeypatch.setattr(runner_module, "communicator_switch", nullcontext)
    monkeypatch.setattr(GPUModelRunner, "profile_cudagraph_memory", boom)

    with pytest.raises(RuntimeError, match="capture failed"):
        runner.profile_cudagraph_memory()

    assert wrapper.graph_pool is original_pool
    assert wrapper.concrete_aclgraph_entries == {}
    assert wrapper.first_run_finished is False
    assert acl_graph._graph_params is None


def test_initialize_kv_cache_profiling_skips_external_side_effects(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(enabled=True, use_fused_copy_sfa=False)
    )
    runner.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object(), indexer_kv_dtype=None),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    runner.compilation_config = runner.vllm_config.compilation_config
    runner.model_config = SimpleNamespace(architecture="Qwen3ForCausalLM")
    runner.speculator = None
    runner.is_last_pp_rank = True
    runner.max_num_reqs = 8
    runner.max_num_tokens = 32
    runner.device = torch.device("cpu")
    runner.pcp_manager = None
    runner.attn_groups = []
    runner.block_tables = SimpleNamespace(block_tables=[])
    runner.model = torch.nn.Module()
    runner.model_state = SimpleNamespace(copy_state_after_kv_load=lambda *_: None)
    runner.kv_connector = object()
    runner._kv_trace = MagicMock()
    monkeypatch.setattr(runner, "initialize_spec_cpu_length_sync", MagicMock())
    monkeypatch.setattr(runner, "_fill_c8_mxfp_v_scale_caches", MagicMock())

    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    observed_upstream = {}

    def fake_upstream(cfg, ctx, is_profiling=False):
        observed_upstream["num_blocks"] = cfg.num_blocks
        observed_upstream["is_profiling"] = is_profiling
        runner.kv_cache_config = cfg

    init_offload = MagicMock()
    bind_dspark = MagicMock()
    mamba_wrap = MagicMock()
    adapt_hisparse = MagicMock()
    overlap_mirror = MagicMock()

    monkeypatch.setattr(runner, "_initialize_kv_cache_upstream", fake_upstream)
    monkeypatch.setattr(runner_module, "configure_dspark_kv_transfer", lambda *_, **__: None)
    monkeypatch.setattr(runner_module, "get_resident_dspark_layer_names", lambda *_, **__: set())
    monkeypatch.setattr(runner_module, "get_layerwise_reuse_layout", lambda *_: None)
    monkeypatch.setattr(runner_module, "is_hisparse_ascendstore_enabled", lambda *_: False)
    monkeypatch.setattr(runner_module, "graph_manager_wrapper", lambda *_: nullcontext())
    monkeypatch.setattr(runner_module, "initialize_kernel_block_table_buffers", lambda *_: None)
    monkeypatch.setattr(runner_module, "is_circular_kv_cache_spec", lambda *_: True)
    monkeypatch.setattr(runner_module, "prepare_v41_source_rope", lambda *_: None)
    monkeypatch.setattr(runner_module, "uses_a5_packed_cache", lambda: False)
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        runner_module,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(requires_mamba_state_copy_after_load=True),
    )
    monkeypatch.setattr(runner_module.KVPPRuntime, "create_from_kv_cache", lambda **_: None)
    monkeypatch.setattr(runner_module, "init_sparse_kv_offload_manager", init_offload)
    monkeypatch.setattr(runner_module, "bind_dspark_context_receiver", bind_dspark)
    monkeypatch.setattr(runner_module, "MambaCopyKVConnector", mamba_wrap)
    monkeypatch.setattr(runner_module, "adapt_pcp_hisparse_connector", adapt_hisparse)
    monkeypatch.setattr(runner_module, "overlap_target_host_mirror", overlap_mirror)
    monkeypatch.setattr("vllm_ascend.ops.kda_state_copy_plan.initialize_kda_state_copy", lambda *_: None)

    runner.initialize_kv_cache(kv_cache_config, is_profiling=True)

    assert observed_upstream["is_profiling"] is True
    assert observed_upstream["num_blocks"] == 9
    init_offload.assert_not_called()
    bind_dspark.assert_not_called()
    mamba_wrap.assert_not_called()
    adapt_hisparse.assert_not_called()
    overlap_mirror.assert_not_called()
    runner._kv_trace.configure.assert_not_called()


def test_xlite_runner_skips_kv_registration_and_bypasses_wrapper_during_profiling():
    xlite_pkg = types.ModuleType("xlite")
    xlite_c = types.ModuleType("xlite._C")
    for cls_name in (
        "Model",
        "ModelConfig",
        "AttnDSA",
        "AttnMetaV2",
        "AttnMHA",
        "AttnMLA",
        "Runtime",
        "ScoringFuncSigmoid",
        "ScoringFuncSoftmax",
    ):
        setattr(xlite_c, cls_name, type(cls_name, (), {}))
    xlite_pkg._C = xlite_c  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"xlite": xlite_pkg, "xlite._C": xlite_c}):
        from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerV2

    runner = XliteModelRunnerV2.__new__(XliteModelRunnerV2)
    runner.fallback_model = torch.nn.Module()
    runner.runner_model = MagicMock()
    runner._model = runner.runner_model
    runner._runner_enabled = True
    runner.kv_caches = [torch.zeros((2, 4))]

    with patch.object(NPUModelRunner, "initialize_kv_cache") as parent_init:
        runner.initialize_kv_cache(SimpleNamespace(), is_profiling=True)
    parent_init.assert_called_once()
    assert parent_init.call_args.kwargs["is_profiling"] is True
    runner.runner_model.register_kv_caches.assert_not_called()

    models_during_profile = []

    def fake_profile() -> int:
        models_during_profile.append(runner.model)
        return 4096

    with patch.object(
        NPUModelRunner,
        "profile_cudagraph_memory",
        side_effect=fake_profile,
    ):
        assert runner.profile_cudagraph_memory() == 4096
    assert models_during_profile == [runner.fallback_model]
    assert runner.model is runner.runner_model
