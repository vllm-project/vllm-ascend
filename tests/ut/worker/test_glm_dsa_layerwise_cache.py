# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import plan_layerwise_cache_reuse
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.worker.v2 import attn_utils


def test_glm_dsa_reuses_mla_and_indexer_and_preserves_views(monkeypatch):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    text_config = SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78)
    transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "backend": "memcache",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": 4,
            "layerwise_independent_layers": [],
        },
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            supports_kv_cache_grouping=False,
            get_total_num_hidden_layers=lambda: text_config.num_hidden_layers,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=transfer_config,
    )
    main = AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=656, dtype=torch.float8_e4m3fn, cache_sparse_sfa_c8=True
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
    )
    main_entries = {layer: f"model.layers.{layer}.self_attn.attn" for layer in range(79)}
    indexer_entries = {layer: f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in (0, 1, *range(2, 79, 4))}
    specs = dict.fromkeys(main_entries.values(), main) | dict.fromkeys(indexer_entries.values(), indexer)
    draft_entries = {main_entries[78], indexer_entries[78]}
    for name in draft_entries:
        specs[name] = replace(specs[name], is_draft=True)

    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)
    assert len(groups) == 1 and len(groups[0].layer_names) == 101
    bytes_per_block = builder._get_pool_bytes_per_block(runtime, groups)
    assert bytes_per_block == 5 * (main.page_size_bytes + indexer.page_size_bytes)
    config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
    layout = plan_layerwise_cache_reuse(config, runtime)
    assert layout is not None
    assert layout.prefetch_layer_map[10] == (0, 6)
    assert layout.draft_entry_names == draft_entries
    assert layout.reused_cache_names == specs.keys() - draft_entries
    assert len(layout.kv_cache_tensors) == 10
    expected_bytes = config.num_blocks * bytes_per_block
    assert get_layerwise_reuse_backing_bytes(config) == expected_bytes
    worker_config = apply_layerwise_kv_cache_plan(config, transfer_config, layout)

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_: (512, 64))
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8, device=device)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    raw = attn_utils._allocate_kv_cache(worker_config, shared_layers={}, device=torch.device("cpu"))
    assert allocations == [expected_bytes]
    assert raw[main_entries[10]] is raw[main_entries[6]]
    assert raw[indexer_entries[10]] is raw[indexer_entries[0]]
    for name in draft_entries:
        assert all(name not in tensor.layers or tensor.layers == [name] for tensor in layout.kv_cache_tensors)

    def get_shape(blocks, tokens, heads, width, cache_dtype=None):
        return blocks, tokens, heads, width

    backend = SimpleNamespace(get_kv_cache_shape=get_shape)
    attn_groups = [
        SimpleNamespace(kv_cache_group_id=0, kv_cache_spec=specs[name], layer_names=[name], backend=backend)
        for name in specs
    ]
    caches = attn_utils._reshape_kv_cache_v2(attn_groups, raw, "fp8", [128], {}, worker_config)
    for name in indexer_entries.values():
        key, scale = caches[name]
        assert key.is_contiguous() and scale.is_contiguous()
        assert key.data_ptr() == raw[name].data_ptr()
        assert scale.data_ptr() == key.data_ptr() + key.numel() * key.element_size()


def test_indexer_reshape_checks_capacity_after_scale_alignment(monkeypatch):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    spec = AscendSFAIndexerCacheSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
    )
    config = KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec([layer_name], spec)])
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        layer_names=[layer_name],
        backend=SimpleNamespace(get_kv_cache_shape=lambda *shape: shape),
    )
    monkeypatch.setattr(attn_utils, "get_layerwise_reuse_layout", lambda _: object())
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: SimpleNamespace())
    # One K byte and four scale bytes fit tightly, but FP32 alignment moves the scale to offset four.
    raw = {layer_name: torch.empty(spec.page_size_bytes, dtype=torch.int8)}
    with pytest.raises(ValueError, match=r"requires 8 bytes for aligned K/scale views, got 5 bytes"):
        attn_utils._reshape_kv_cache_v2([group], raw, "fp8", [1], {}, config)
