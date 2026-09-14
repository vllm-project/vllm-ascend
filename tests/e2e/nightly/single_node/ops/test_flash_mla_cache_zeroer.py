# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Actual NPU zeroing of upstream packed, overlaid and virtually split pages."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import KVBlockZeroer, allocate_kv_cache

from vllm_ascend.device.device_config import is_950
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache


@pytest.mark.parametrize(("manager_size", "interleaved"), [(16, True), (32, False)])
def test_real_npu_zeroer_preserves_other_blocks_and_padding(monkeypatch, manager_size, interleaved):
    if not torch.npu.is_available() or not is_950():
        pytest.skip("Requires an A5 NPU and the RFC 16464 pinned runtime.")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASH_MLA", "1")
    device = torch.device("npu:0")
    blocks, offset, kernel_size = 3, 64, 16
    page_bytes = manager_size * 576 * 2
    pitch = 2 * page_bytes + 128 if interleaved else page_bytes
    layer_stride = page_bytes if interleaved else blocks * page_bytes
    size = offset + blocks * (2 * page_bytes + 128) + 64
    mla_names = ["model.layers.0.attn", "model.layers.1.attn"]
    mla = MLAAttentionSpec(block_size=manager_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    state = MambaSpec(
        block_size=32768,
        shapes=((6, 48), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_bytes,
    )
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(size, mla_names, layer_stride, pitch, offset),
            KVCacheTensor(size, ["model.layers.2.mamba"], page_bytes, pitch, offset),
        ],
        kv_cache_groups=[KVCacheGroupSpec(mla_names, mla), KVCacheGroupSpec(["model.layers.2.mamba"], state)],
    )
    layout = KVCacheLayout.BLNHC if interleaved else KVCacheLayout.LBNHC
    kernel_sizes = [kernel_size, state.block_size]
    caches = allocate_kv_cache(config, device, layout, kernel_sizes)
    contexts = {}
    for name in mla_names:
        layer = SimpleNamespace(_vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=False)))
        layer.bind_kv_cache = lambda cache, layer=layer: MLAAttention.bind_kv_cache(layer, cache)
        contexts[name] = layer
    mamba = SimpleNamespace(get_state_shape=lambda: state.shapes, get_state_dtype=lambda: state.dtypes)
    mamba.bind_kv_cache = lambda cache: MambaBase.bind_kv_cache(mamba, cache)
    contexts["model.layers.2.mamba"] = mamba
    bind_kv_cache(caches, contexts, [], kv_cache_groups=config.kv_cache_groups)
    groups = [SimpleNamespace(kv_cache_spec=mla, kv_cache_group_id=0, layer_names=mla_names)]
    zeroer = KVBlockZeroer(device, groups, kernel_sizes, contexts, blocks)
    raw = torch.empty(0, dtype=torch.int8, device=device).set_(caches[mla_names[0]].untyped_storage())
    raw.fill_(-91)
    zeroer.zero_block_ids([1])
    torch.npu.synchronize()
    actual = raw.cpu()
    expected = torch.full_like(actual, -91)
    for layer_index in range(len(mla_names)):
        start = offset + layer_index * layer_stride + pitch
        expected[start : start + page_bytes] = 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The state group overlays attention's first layer, so the selected global
    # manager block owns and clears both typed state payloads as well.
    conv, ssm = contexts["model.layers.2.mamba"].kv_cache
    assert torch.count_nonzero(conv[1]).item() == 0
    assert torch.count_nonzero(ssm[1]).item() == 0
    previous = actual.clone()
    zeroer.zero_block_ids([])
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), previous, rtol=0, atol=0)
