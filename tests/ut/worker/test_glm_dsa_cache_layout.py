# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise the GLM-5.2/5.3 spec publisher, Core planner, and MRV2 allocation together."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.mla_v1 import AscendMLAAttention
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, register_ascend_kv_cache_specs
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize("attention_layout", ["contiguous", "block_interleaved"])
@pytest.mark.parametrize("pool_layout", [KVCacheLayout.LBNHC, KVCacheLayout.LBHNC])
def test_glm_planned_cache_bytes(monkeypatch, attention_layout, pool_layout):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    blocks, tokens = 3, 128
    latent, rope, index_width = 512, 64, 128
    text = SimpleNamespace(model_type="glm_moe_dsa", indexer_types=["full", "shared"])
    runtime = SimpleNamespace(
        additional_config={"sfa_cache_layout": attention_layout},
        model_config=SimpleNamespace(hf_config=text, hf_text_config=text, supports_kv_cache_grouping=False),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: pool_layout,
            cache_dtype="auto",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
        speculative_config=None,
    )
    # Two attention owners share top-k results. Only the full indexer owns a cache.
    first = "model.layers.0.self_attn.attn"
    second = "model.layers.1.self_attn.attn"
    indexer = "model.layers.0.self_attn.indexer.k_cache"
    source = MLAAttentionSpec(block_size=tokens, num_kv_heads=1, head_size=latent + rope, dtype=torch.bfloat16)
    monkeypatch.setattr(MLAAttention, "get_kv_cache_spec", lambda *_: source)
    layer = AscendMLAAttention.__new__(AscendMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace()
    layer.num_heads = 8
    layer.kv_lora_rank = latent
    layer.qk_rope_head_dim = rope
    layer.is_draft_layer = False
    layer.attn_backend = AscendSFABackend
    layer.kv_sharing_target_layer_name = None
    main_spec = layer.get_kv_cache_spec(runtime)
    assert main_spec.num_heads == 1
    index_spec = AscendSFAIndexerCacheSpec(
        block_size=tokens,
        num_kv_heads=1,
        head_size=index_width,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    specs = {first: replace(main_spec, is_index_group_leader=True), second: main_spec, indexer: index_spec}
    layers = {first: layer, second: layer, indexer: SimpleNamespace(get_attn_backend=lambda: AscendSFAIndexerBackend)}
    runtime.compilation_config = SimpleNamespace(static_forward_context=layers)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    ascend_config = SimpleNamespace(
        sfa_cache_layout=attention_layout,
        kvpp_config=SimpleNamespace(size=1),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(attn_utils, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)
    config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=blocks)
    caches = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), pool_layout, [tokens] * len(groups))

    first_nope, first_rope = caches[first]
    second_nope, second_rope = caches[second]
    keys, scales = caches[indexer]
    assert set(caches) == set(specs)
    # Address expectations come from GLM's wire geometry, independently of the
    # candidate views. A stray query-head factor would move the second owner
    # beyond the planner's reservation; a component-major indexer would move
    # scale[0] behind every K page.
    main_page = tokens * (latent + rope) * 2
    index_page = tokens * (index_width + 2)
    assert first_nope.untyped_storage().nbytes() == blocks * (2 * main_page + index_page)
    assert second_nope.data_ptr() - first_nope.data_ptr() == blocks * main_page
    assert keys.data_ptr() - second_nope.data_ptr() == blocks * main_page
    assert scales.data_ptr() - keys.data_ptr() == tokens * index_width
    assert keys.stride(0) == index_page
    assert scales.stride(0) * scales.element_size() == index_page
    for nope_cache, rope_cache in [(first_nope, first_rope), (second_nope, second_rope)]:
        assert nope_cache.shape == (blocks, tokens, 1, latent)
        assert rope_cache.shape == (blocks, tokens, 1, rope)
        if attention_layout == "contiguous":
            assert nope_cache.is_contiguous() and rope_cache.is_contiguous()
            assert rope_cache.data_ptr() - nope_cache.data_ptr() == blocks * tokens * latent * 2
        else:
            assert nope_cache.stride(0) * 2 == rope_cache.stride(0) * 2 == main_page
            assert rope_cache.data_ptr() - nope_cache.data_ptr() == tokens * latent * 2

    keys[1].fill_(7)
    scales[2].fill_(0.5)
    first_nope[1].fill_(2)
    second_rope[0].fill_(5)
    expected = torch.zeros(blocks * (2 * main_page + index_page), dtype=torch.uint8)
    nope_start = tokens * latent * 2 if attention_layout == "contiguous" else main_page
    rope_start = blocks * main_page + (blocks if attention_layout == "contiguous" else 1) * tokens * latent * 2
    expected[nope_start : nope_start + tokens * latent * 2].view(torch.bfloat16).fill_(2)
    expected[rope_start : rope_start + tokens * rope * 2].view(torch.bfloat16).fill_(5)
    index_start = 2 * blocks * main_page
    expected[index_start + index_page : index_start + index_page + tokens * index_width].fill_(7)
    expected[index_start + 2 * index_page + tokens * index_width :].view(torch.float16).fill_(0.5)
    assert torch.equal(torch.tensor(first_nope.untyped_storage(), dtype=torch.uint8), expected)


@pytest.mark.parametrize("c4", [False, True], ids=["int8-fp16", "mxfp4-e8m0"])
def test_replicated_indexer_places_scales_after_each_kernel_page(c4):
    # Two DCP replicas, each with two kernel pages per manager block.
    blocks, kernel_tokens, replicas = 3, 128, 2
    width, scale_dim, scale_dtype = (64, 4, torch.uint8) if c4 else (128, 1, torch.float16)
    spec = AscendSFAIndexerCacheSpec(
        block_size=2 * kernel_tokens,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.int8,
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
        cache_sparse_li_c4=c4,
        cache_sparse_li_c8=not c4,
        sfa_dcp_replicated_indexer_size=replicas,
    )
    # Include a nonzero storage offset, as after base-address alignment in PD.
    size = blocks * spec.page_size_bytes
    backing = torch.zeros(size + 64, dtype=torch.int8)
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=[KVCacheGroupSpec(["index"], spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=size,
                layers=["index"],
                offset=0,
                layer_stride=size,
                block_stride=spec.page_size_bytes,
            )
        ],
    )
    group = SimpleNamespace(
        backend=AscendSFAIndexerBackend,
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        kernel_block_size=kernel_tokens,
        layer_names=["index"],
    )
    key, scale = attn_utils.create_kv_cache_runtime_views(
        config,
        backing[64:],
        [group],
        [spec.block_size],
        {},
        contiguous_main_cache=True,
    )["index"]
    key_bytes = kernel_tokens * width
    scale_bytes = kernel_tokens * scale_dim * scale.element_size()
    page_bytes = key_bytes + scale_bytes
    expected = torch.zeros_like(backing)
    assert key.shape == (blocks * replicas * 2, kernel_tokens, 1, width)
    scale_suffix = (2, 2) if c4 else (1,)
    assert scale.shape == (*key.shape[:3], *scale_suffix)
    for page in range(key.shape[0]):
        start = 64 + page * page_bytes
        key[page].fill_(page + 1)
        scale[page].fill_(page + 2)
        expected[start : start + key_bytes].fill_(page + 1)
        expected[start + key_bytes : start + page_bytes].view(scale_dtype).fill_(page + 2)
    assert torch.equal(backing, expected)


@pytest.mark.parametrize("mode", ["kvpp", "sparse_offload", "layerwise_offload"])
def test_specialized_allocators_bypass_sfa_views(monkeypatch, mode):
    text = SimpleNamespace(model_type="glm_moe_dsa", indexer_types=["full", "shared"])
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text), cache_config=SimpleNamespace(cache_dtype="auto")
    )
    config = SimpleNamespace(host_group_ids=(), kv_cache_groups=[object()])
    ascend = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=2 if mode == "kvpp" else 1),
        sparse_kv_offload_config=SimpleNamespace(enabled=mode == "sparse_offload"),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: {})
    monkeypatch.setattr(attn_utils, "_get_cache_attention_groups", lambda *_: [])
    monkeypatch.setattr(
        attn_utils, "get_layerwise_reuse_layout", lambda _: object() if mode == "layerwise_offload" else None
    )
    existing = {"owner": object()}
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", lambda *args, **kwargs: existing)
    monkeypatch.setattr(attn_utils, "_reshape_kv_cache_v2", lambda **kwargs: kwargs["kv_cache_raw_tensors"])
    assert attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), KVCacheLayout.LBNHC, [128]) is existing
