# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device validation of standard page-strided cache views and their consumers."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from tests.ut.models.qwen4_exp.test_cache_config import _allocate, _config, _make_config, _specs
from vllm_ascend.models.qwen4_exp.qsa import _split_qsa_kv_cache
from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_store_cache_rows
from vllm_ascend.patch.worker.patch_mamba_utils import (
    _collect_mamba_copy_meta_torch,
    _do_mamba_copy_block_torch,
)


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_qwen_standard_pages_npu_store_restore_and_graph(ssm_dtype):
    specs, model_config = _specs(2), _config(cycles=2)
    for name, spec in list(specs.items()):
        if name.endswith("linear_attn"):
            specs[name] = replace(spec, dtypes=(torch.bfloat16, ssm_dtype))
    config = _make_config(specs, 4, model_config)
    groups = config.kv_cache_groups
    backings, raw, caches = _allocate(config, device="npu")
    assert len(backings) == 1
    key, value = _split_qsa_kv_cache(caches["model.layers.3.self_attn.attn"], 256)
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
    assert not torch.count_nonzero(caches["model.layers.7.self_attn.attn"].cpu())

    copy_funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func(),
        MambaAttentionBackendEnum.SHORT_CONV: MambaStateCopyFuncCalculator.short_conv_state_copy_func(),
    }
    for group_id in (2, 3):
        for marker, name in enumerate(groups[group_id].layer_names, 1):
            for state in caches[name]:
                state[2].fill_(marker)
        buffers = SimpleNamespace(offset=0, sizes=SimpleNamespace(np=np.zeros(13, dtype=np.int32)))
        request = SimpleNamespace(block_ids=([0, 1, 2, 3],) * len(groups))
        context = {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
        _collect_mamba_copy_meta_torch(buffers, config, copy_funcs, [group_id], 2, 3, 0, request, context)
        _do_mamba_copy_block_torch(buffers)
        for marker, name in enumerate(groups[group_id].layer_names, 1):
            for state in caches[name]:
                assert torch.all(state[3].cpu() == marker)
    torch.testing.assert_close(key[:2, :2].cpu().reshape_as(updates.cpu()), updates.cpu())
