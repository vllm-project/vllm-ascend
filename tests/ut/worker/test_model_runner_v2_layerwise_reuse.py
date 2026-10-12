# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize("reuse_enabled", [False, True])
@pytest.mark.parametrize("zero_stride", [False, True])
@pytest.mark.parametrize("cache_kind", ["full", "sfa_c8", "indexer", "indexer_scale"])
def test_real_allocator_aliases_only_planned_slots(monkeypatch, reuse_enabled, zero_stride, cache_kind):
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float16)
    if cache_kind == "sfa_c8":
        spec = AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            cache_sparse_sfa_c8=True,
        )
    elif cache_kind.startswith("indexer"):
        spec = AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            scale_dim=1 if cache_kind == "indexer_scale" else 0,
        )
    names = ["model.layers.0.attn", "model.layers.1.attn"]
    size = 3 * spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=size if zero_stride else 2 * size,
                layers=names,
                layer_stride=0 if zero_stride else size,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(kv_transfer_config=None)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "is_deepseek_v41_cache", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: cache_kind == "sfa_c8")
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda _: False)
    monkeypatch.setattr(attn_utils, "get_layerwise_reuse_config", lambda _: {} if reuse_enabled else None)
    layer = SimpleNamespace(get_attn_backend=lambda: SimpleNamespace(is_sparse=lambda: False))
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: dict.fromkeys(names, layer))
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8, device=device)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    raw = attn_utils._allocate_kv_cache(config, shared_layers={}, device=torch.device("cpu"))
    shares = reuse_enabled and zero_stride
    assert len(allocations) == (1 if shares else 2)
    assert (raw[names[0]] is raw[names[1]]) == shares
    first = raw[names[0]] if isinstance(raw[names[0]], torch.Tensor) else raw[names[0]][0]
    second = raw[names[1]] if isinstance(raw[names[1]], torch.Tensor) else raw[names[1]][0]
    first.fill_(7)
    assert torch.all(second == (7 if shares else 0))
    assert config.kv_cache_tensors[0].layers == names
