# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops.triton import eplb
from vllm_ascend.patch.platform import patch_fused_moe as owner


@pytest.fixture
def map_kernel(monkeypatch):
    wrapper = eplb.EplbMapKernel()
    definition = eplb.EplbMapKernel.kernel
    arg_names = list(inspect.signature(getattr(definition, "fn", definition)).parameters)
    jit_kernel = Mock(arg_names=arg_names)
    jit_kernel.__getitem__ = Mock(return_value=Mock())
    wrapper.kernel = jit_kernel
    monkeypatch.setattr(eplb, "_EPLB_MAP_KERNEL", wrapper)
    return wrapper


def test_factory_registers_eplb_map_in_active_window(map_kernel):
    config = SimpleNamespace(use_v2_model_runner=True)
    registry = make_registry(config)
    router = Mock()
    factory = Mock(return_value=SimpleNamespace(router=router))
    ascend = SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_path=None))
    with (
        patch.object(owner, "_original_FusedMoE", factory),
        patch.object(owner, "_adapt_eplb_router"),
        patch.object(owner, "get_ascend_config", return_value=ascend),
        patch.object(owner, "vllm_version_is", return_value=True),
    ):
        owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
        assert len(registry) == 0
        with registry.activate():
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=False)
            assert len(registry) == 0
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)
            owner._ascend_FusedMoE(128, 8, router=router, enable_eplb=True)

    assert registry._registrations[map_kernel] == [((128, 8), {})]
    map_kernel.kernel.warmup.assert_not_called()


def test_map_warmup_binds_all_variants_without_launching(map_kernel):
    keys = map_kernel.get_warmup_keys(128, 8)
    assert len(keys) == len(set(keys)) == 6
    assert {(key.block_size, key.ids_dtype) for key in keys} == {
        (tile, dtype) for tile in (64, 128, 1024) for dtype in (torch.int32, torch.int64)
    }
    for key in keys:
        map_kernel.compile(key)
        args = map_kernel.kernel.warmup.call_args.kwargs
        assert args["grid"] == (1,)
        assert args["TOPK"] == 8
        assert args["BLOCK_SIZE"] == key.block_size
        assert args["num_logical_experts"] == 128
        assert args["routing_table_rows"] == eplb.EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
        assert args["numel"] == key.block_size
        for ptr, dtype in (
            ("topk_ids_ptr", key.ids_dtype),
            ("routing_table_ptr", torch.int32),
            ("physical_ids_ptr", key.ids_dtype),
        ):
            assert isinstance(args[ptr], TritonWarmupTensor)
            assert args[ptr].dtype == dtype
            assert args[ptr].data_ptr() == 0
    map_kernel.kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens,tile", [(0, None), (1, 64), (256, 64), (257, 128), (1024, 128), (1025, 1024)])
def test_map_runtime_dispatch_is_covered_by_warmup(map_kernel, tokens, tile, dtype):
    topk = 8
    experts = 128
    rows = eplb.EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
    ids = torch.empty((tokens, topk), dtype=dtype)
    table = torch.empty((rows, experts), dtype=torch.int32)

    out = eplb.map_to_physical_triton(ids, table)
    if tile is None:
        assert out is ids
        map_kernel.kernel.__getitem__.assert_not_called()
        return

    numel = tokens * topk
    map_kernel.kernel.__getitem__.assert_called_once_with(((numel + tile - 1) // tile,))
    map_kernel.kernel.__getitem__.return_value.assert_called_once_with(
        topk_ids_ptr=ids,
        routing_table_ptr=table,
        physical_ids_ptr=out,
        num_logical_experts=experts,
        numel=numel,
        routing_table_rows=rows,
        TOPK=topk,
        BLOCK_SIZE=tile,
    )
    key = map_kernel.dispatch(num_logical_experts=experts, topk=topk, block_size=tile, ids_dtype=dtype)
    assert key in map_kernel.get_warmup_keys(experts, topk)
