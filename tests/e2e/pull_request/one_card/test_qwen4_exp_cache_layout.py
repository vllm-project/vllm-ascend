# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device validation of the actual four-row cache views and their consumers."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from tests.ut.models.qwen4_exp.test_cache_config import _config, _specs
from vllm_ascend.models.qwen4_exp.cache_config import (
    allocate_qwen4_exp_kv_cache_tensors,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
    reshape_qwen4_exp_kv_cache_tensors,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_store_cache_rows
from vllm_ascend.patch.worker.patch_mamba_utils import (
    _collect_mamba_copy_meta_torch,
    _do_mamba_copy_block_torch,
)


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_qwen_split_rows_npu_store_restore_and_graph(ssm_dtype):
    specs, model_config = _specs(2), _config(cycles=2)
    for name, spec in list(specs.items()):
        if name.endswith("linear_attn"):
            specs[name] = replace(spec, dtypes=(torch.bfloat16, ssm_dtype))
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    config = get_qwen4_exp_kv_cache_config(model_config, groups, 4 * get_qwen4_exp_pool_bytes_per_block(groups))
    backings = []

    def allocator(size, alignment):
        backing = torch.zeros(size, dtype=torch.int8, device="npu")
        backings.append(backing)
        return backing

    raw = allocate_qwen4_exp_kv_cache_tensors(config, allocator)
    caches = reshape_qwen4_exp_kv_cache_tensors(config, raw)
    assert len({backing.untyped_storage().data_ptr() for backing in backings}) == 2
    key, value = caches["model.layers.3.self_attn.attn"]
    slots = torch.tensor([0, 1, 128, 129], dtype=torch.int64, device="npu")
    updates = torch.arange(4 * 256, device="npu", dtype=torch.float32).reshape(4, 1, 256).bfloat16()
    qsa_store_cache_rows(key, slots, updates)
    qsa_store_cache_rows(value, slots, updates)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        qsa_store_cache_rows(key, slots, updates)
        qsa_store_cache_rows(value, slots, updates)
    key.zero_()
    value.zero_()
    graph.replay()
    torch.npu.synchronize()
    for cache in (key, value):
        torch.testing.assert_close(cache[:2, :2].cpu().reshape_as(updates.cpu()), updates.cpu())
        assert not torch.count_nonzero(cache[2:].cpu())
    assert not torch.count_nonzero(caches["model.layers.7.self_attn.attn"][0].cpu())

    # G1 owns block 2; restoring it into block 3 must use each component's
    # physical stride, with distinct copy functions for GDN and PLE.
    for marker, name in enumerate(groups[1].layer_names, 1):
        for state in caches[name]:
            state[2].fill_(marker)
    copy_funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func(),
        MambaAttentionBackendEnum.SHORT_CONV: MambaStateCopyFuncCalculator.short_conv_state_copy_func(),
    }
    buffers = SimpleNamespace(offset=0, sizes=SimpleNamespace(np=np.zeros(13, dtype=np.int32)))
    request = SimpleNamespace(block_ids=([0, 1, 2, 3],) * 3)
    context = {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
    _collect_mamba_copy_meta_torch(buffers, config, copy_funcs, [1], 2, 3, 0, request, context)
    _do_mamba_copy_block_torch(buffers)
    for marker, name in enumerate(groups[1].layer_names, 1):
        for state in caches[name]:
            assert torch.all(state[3].cpu() == marker)
    # The independent compressed-key backing is untouched by all state copies.
    assert not torch.count_nonzero(backings[1].cpu())
