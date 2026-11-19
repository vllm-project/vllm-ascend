from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.triton.eplb_map_flat import _EPLB_MAP_KERNEL as kernel
from vllm_ascend.patch.platform import patch_fused_moe as owner


def test_map_keys():
    if not hasattr(kernel, "get_warmup_keys"):
        pytest.skip("model registry warmup is unavailable on this branch")
    keys = kernel.get_warmup_keys(128, 8)
    assert len(set(keys)) == 6
    assert {(k.block, k.ids_dtype) for k in keys} == {
        (b, dt) for b in (64, 128, 1024) for dt in (torch.int32, torch.int64)
    }
    for key in keys:
        args = kernel.warmup_inputs(key)
        assert args["TOPK"] == 8 and args["num_logical_experts"] == 128
        assert args["topk_ids"].dtype == args["physical_ids"].dtype == key.ids_dtype
        assert args["routing_table"].dtype == torch.int32


def test_factory_registers_only_active_eplb():
    if not hasattr(kernel, "get_warmup_keys"):
        pytest.skip("model registry warmup is unavailable on this branch")
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    config = SimpleNamespace(use_v2_model_runner=True)
    registry = JitWarmupRegistry(config)
    router = Mock()
    factory = Mock(return_value=SimpleNamespace(router=router))
    ascend = SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None))
    with (
        patch.object(owner, "get_device_config", return_value=SimpleNamespace(use_flat_eplb_map=True)),
        patch.object(owner, "_original_FusedMoE", factory),
        patch.object(owner, "_adapt_eplb_router"),
        patch.object(owner, "get_ascend_config", return_value=ascend),
        patch.object(owner, "get_current_vllm_config", return_value=config),
    ):
        owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
        assert len(registry) == 0
        with registry.activate():
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=False)
            assert len(registry) == 0
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
            assert len(registry) == 1
            assert registry._registrations[kernel] == [((128, 8), {})]


def test_worker_warmup_compiles_each_variant_once():
    from vllm_ascend.ops.triton.eplb_map_worker_warmup import WorkerMapKernel

    jit = Mock()
    worker_kernel = WorkerMapKernel(jit, (64, 128, 1024), 1024)
    worker_kernel.register_warmup(128, 8)
    worker_kernel.register_warmup(128, 8)
    worker_kernel.warmup()
    worker_kernel.warmup()
    assert len(jit.warmup.call_args_list) == 6
    for call in jit.warmup.call_args_list:
        dtype, table_dtype, output_dtype, experts, tile, rows = call.args
        assert dtype == output_dtype and dtype in (torch.int32, torch.int64)
        assert table_dtype == torch.int32 and (experts, rows) == (128, 1024)
        assert call.kwargs == {"grid": (1,), "TOPK": 8, "BLOCK_SIZE": tile}


def test_factory_collects_worker_warmup_geometry():
    from vllm_ascend.ops.triton.eplb_map_worker_warmup import WorkerMapKernel

    worker_kernel = WorkerMapKernel(Mock(), (64, 128, 1024), 1024)
    config = SimpleNamespace(use_v2_model_runner=True)
    router = Mock()
    factory = Mock(return_value=SimpleNamespace(router=router))
    ascend = SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None))
    with (
        patch("vllm_ascend.ops.triton.eplb_map_flat._EPLB_MAP_KERNEL", worker_kernel),
        patch.object(owner, "get_device_config", return_value=SimpleNamespace(use_flat_eplb_map=True)),
        patch.object(owner, "_original_FusedMoE", factory),
        patch.object(owner, "_adapt_eplb_router"),
        patch.object(owner, "get_ascend_config", return_value=ascend),
        patch.object(owner, "get_current_vllm_config", return_value=config),
    ):
        owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=False)
        assert not worker_kernel.pending
        owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
        assert worker_kernel.pending == {(128, 8)}
