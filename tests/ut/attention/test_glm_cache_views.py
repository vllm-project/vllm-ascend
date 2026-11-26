# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture(autouse=True)
def register_specs_and_small_blocks(monkeypatch):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    for backend in (AscendSFABackend, AscendSFAIndexerBackend, AscendAttentionBackend):
        sizes = backend.get_supported_kernel_block_sizes()
        monkeypatch.setattr(backend, "get_supported_kernel_block_sizes", lambda spec=None, sizes=sizes: [2, 4, *sizes])


def _make_glm_runtime(
    *,
    model_type: str = "glm_moe_dsa",
    cache_dtype: str = "auto",
    layout: KVCacheLayout = KVCacheLayout.LBNHC,
    connector: str | None = None,
    enable_kvpp: bool = False,
):
    text_config = SimpleNamespace(model_type=model_type, num_hidden_layers=4, index_topk=2048)
    return SimpleNamespace(
        additional_config={"enable_kvpp": enable_kvpp},
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            use_mla=True,
            supports_kv_cache_grouping=False,
            get_total_num_hidden_layers=lambda: text_config.num_hidden_layers,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
            get_resolved_kv_cache_layout=lambda: layout,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=2 if enable_kvpp else 1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=(
            SimpleNamespace(kv_connector=connector, kv_connector_extra_config={}) if connector else None
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _build_glm_plan(
    runtime,
    specs: dict[str, AscendMLAAttentionSpec | AscendSFAIndexerCacheSpec | FullAttentionSpec],
    *,
    num_blocks: int = 3,
) -> KVCacheConfig:
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)
    return builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=num_blocks)


def _allocate_unified(
    config: KVCacheConfig,
    runtime,
    *,
    extra_layers: dict[str, object] | None = None,
    kernel_block_sizes: list[int] | None = None,
):
    layers: dict[str, object] = {}
    for group in config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for name in group.layer_names:
            spec = group_spec.kv_cache_specs[name] if hasattr(group_spec, "kv_cache_specs") else group_spec
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                backend = AscendSFAIndexerBackend
            elif isinstance(spec, AscendMLAAttentionSpec):
                backend = AscendSFABackend
            elif isinstance(spec, FullAttentionSpec):
                backend = AscendAttentionBackend
            else:
                backend = AscendSFABackend
            layers[name] = SimpleNamespace(get_attn_backend=lambda backend=backend: backend, num_heads=1)
    if extra_layers:
        layers.update(extra_layers)
    runtime.compilation_config = SimpleNamespace(static_forward_context=layers)

    size = config.kv_cache_tensors[0].size
    raw = torch.full((size,), 33, dtype=torch.int8)
    sizes = kernel_block_sizes or [group.kv_cache_spec.block_size for group in config.kv_cache_groups]
    with (
        patch.object(attn_utils, "get_current_vllm_config", return_value=runtime),
        patch.object(attn_utils, "get_layers_from_vllm_config", return_value=layers),
        patch.object(attn_utils, "_allocate_int8_cache_tensor", return_value=raw) as allocator,
        patch.object(attn_utils, "_allocate_kv_cache", side_effect=AssertionError("Must not use legacy allocator")),
        patch.object(attn_utils, "_reshape_kv_cache_v2", side_effect=AssertionError("Must not use legacy reshape")),
    ):
        caches = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, sizes)
        allocator.assert_called_once_with(size, 2 * 1024 * 1024, torch.device("cpu"))
    return raw, caches


@pytest.mark.parametrize(
    "main_dtype,packed_flag,main_width,nope_dim,rope_dim",
    [
        (torch.bfloat16, None, 16, 12, 4),
        (torch.float16, None, 16, 12, 4),
        (torch.uint8, "cache_sparse_sfa_hif8", 12, 12, 0),
        (torch.int8, "cache_sparse_sfa_c8", 20, None, 4),
        (torch.uint8, "cache_sparse_sfa_turboquant", 18, None, 4),
    ],
)
@pytest.mark.parametrize(
    "idx_dtype,idx_width,scale_dim,scale_dtype,li_c4",
    [
        (torch.bfloat16, 8, 0, torch.int8, False),
        (torch.int8, 8, 1, torch.float16, False),
        (torch.float8_e4m3fn, 8, 1, torch.float32, False),
        (torch.uint8, 8, 4, torch.float8_e8m0fnu, True),
    ],
)
def test_glm52_53_unified_pool_supports_all_sfa_and_indexer_formats(
    main_dtype,
    packed_flag,
    main_width,
    nope_dim,
    rope_dim,
    idx_dtype,
    idx_width,
    scale_dim,
    scale_dtype,
    li_c4,
):
    runtime = _make_glm_runtime()
    main_kwargs = {packed_flag: True} if packed_flag else {}
    main_spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=main_width,
        num_query_heads=16,
        dtype=main_dtype,
        kv_lora_rank=nope_dim,
        qk_rope_head_dim=rope_dim,
        **main_kwargs,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=idx_width,
        dtype=idx_dtype,
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
        cache_sparse_li_c8=scale_dim > 0 and not li_c4,
        cache_sparse_li_c4=li_c4,
    )
    specs = {
        "model.layers.0.self_attn.attn": main_spec,
        "model.layers.0.self_attn.indexer.k_cache": indexer_spec,
        "model.layers.1.self_attn.attn": main_spec,
        "model.layers.1.self_attn.indexer.k_cache": indexer_spec,
        "model.mtp.layers.0.self_attn.attn": replace(main_spec, is_draft=True, non_causal_multi_token_decode=True),
    }
    config = _build_glm_plan(runtime, specs, num_blocks=3)
    raw, caches = _allocate_unified(config, runtime)

    # Every component across main, indexer, and MTP draft aliases the single backing buffer.
    assert {part.untyped_storage().data_ptr() for components in caches.values() for part in components} == {
        raw.untyped_storage().data_ptr()
    }

    main_names = (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
        "model.mtp.layers.0.self_attn.attn",
    )
    for main_name in main_names:
        parts = caches[main_name]
        assert isinstance(parts, tuple)
        assert all(part.is_contiguous() and part.dtype == main_dtype for part in parts)
        if main_spec.uses_packed_sfa_main_cache:
            (packed_view,) = parts
            assert packed_view.shape == (3, 4, 1, main_width)
        else:
            nope_view, rope_view = parts
            assert nope_view.shape == (3, 4, 1, nope_dim)
            assert rope_view.shape == (3, 4, 1, rope_dim)
            if rope_dim > 0:
                assert rope_view.data_ptr() == nope_view.data_ptr() + nope_view.nbytes

    for idx_name in ("model.layers.0.self_attn.indexer.k_cache", "model.layers.1.self_attn.indexer.k_cache"):
        idx_parts = caches[idx_name]
        assert isinstance(idx_parts, tuple)
        assert all(part.is_contiguous() for part in idx_parts)
        if scale_dim == 0:
            (key_view,) = idx_parts
            assert key_view.shape == (3, 4, 1, idx_width)
        else:
            key_view, scale_view = idx_parts
            assert key_view.shape == (3, 4, 1, idx_width)
            expected_scale_shape = (3, 4, 1, scale_dim // 2, 2) if li_c4 else (3, 4, 1, scale_dim)
            assert scale_view.shape == expected_scale_shape
            assert scale_view.dtype == scale_dtype
            assert scale_view.data_ptr() == key_view.data_ptr() + key_view.nbytes

    # Byte-oracle write isolation across every layer and component.
    snapshot = raw.clone()
    for name, parts in caches.items():
        for part in parts:
            if part.numel() == 0:
                continue
            part.view(torch.uint8).fill_(91)
            for other_name, other_parts in caches.items():
                for other_part in other_parts:
                    if other_part.data_ptr() != part.data_ptr() and other_part.numel() > 0:
                        assert bool((other_part.view(torch.uint8) == 33).all()), (
                            f"Write into {name} bled into {other_name}"
                        )
            raw.copy_(snapshot)


def test_split_main_and_contiguous_indexer_exact_byte_oracle():
    runtime = _make_glm_runtime()
    main = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        num_query_heads=16,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(size=1024, layers=["main.0"], offset=32, layer_stride=288, block_stride=96),
            KVCacheTensor(size=1024, layers=["indexer.0"], offset=352, layer_stride=120, block_stride=40),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["main.0"], kv_cache_spec=main),
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=indexer),
        ],
    )
    raw, caches = _allocate_unified(config, runtime)
    nope, rope = caches["main.0"]
    key, scale = caches["indexer.0"]

    # Main layer slice [32 : 320]: NoPE [32 : 224] (3*4*8*2 = 192B), RoPE [224 : 320] (3*4*4*2 = 96B).
    nope[2, 1].fill_(2.0)
    rope[1, 3].fill_(-1.0)
    # Indexer slice [352 : 472]: K plane [352 : 448] (3*4*8*1 = 96B), scale plane [448 : 472] (3*4*1*2 = 24B).
    key[1, 2].fill_(7)
    scale[1, 2].fill_(0.5)

    expected = torch.full_like(raw, 33)
    nope_start = 32 + (2 * 4 + 1) * 8 * 2
    expected[nope_start : nope_start + 16] = torch.full((8,), 2.0, dtype=torch.float16).view(torch.int8)
    rope_start = 224 + (1 * 4 + 3) * 4 * 2
    expected[rope_start : rope_start + 8] = torch.full((4,), -1.0, dtype=torch.float16).view(torch.int8)
    key_start = 352 + (1 * 4 + 2) * 8
    expected[key_start : key_start + 8] = 7
    scale_start = 448 + (1 * 4 + 2) * 2
    expected[scale_start : scale_start + 2] = torch.tensor([0.5], dtype=torch.float16).view(torch.int8)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("replicas", [1, 2])
def test_kernel_block_splitting_and_replicated_indexer_pages(replicas):
    runtime = _make_glm_runtime()
    main = AscendMLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=576,
        num_query_heads=16,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        sfa_dcp_replicated_indexer_size=replicas,
    )
    config = _build_glm_plan(runtime, {"main.0": main, "indexer.0": indexer}, num_blocks=3)
    _, caches = _allocate_unified(config, runtime)

    nope, rope = caches["main.0"]
    assert nope.shape == (6, 128, 1, 512)
    assert rope.shape == (6, 128, 1, 64)
    assert nope.is_contiguous() and rope.is_contiguous()

    key, scale = caches["indexer.0"]
    expected_pages = 3 * 2 * replicas
    assert key.shape == (expected_pages, 128, 1, 128)
    assert scale.shape == (expected_pages, 128, 1, 1)
    assert key.is_contiguous() and scale.is_contiguous()
    assert scale.data_ptr() == key.data_ptr() + key.nbytes


@pytest.mark.parametrize("head_size_v", [8, 12], ids=["symmetric_kv", "asymmetric_kv"])
@pytest.mark.parametrize("kernel_block_size", [4, 2], ids=["unsplit", "split_kernel_blocks"])
def test_glm_with_full_attention_draft_uses_unified_pool(head_size_v, kernel_block_size):
    runtime = _make_glm_runtime()
    main = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        num_query_heads=16,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    draft = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        head_size_v=head_size_v,
        dtype=torch.float16,
        is_draft=True,
    )
    specs = {
        "model.layers.0.self_attn.attn": main,
        "model.layers.0.self_attn.indexer.k_cache": indexer,
        "draft.layers.0.self_attn.attn": draft,
    }
    config = _build_glm_plan(runtime, specs, num_blocks=3)
    raw, caches = _allocate_unified(config, runtime, kernel_block_sizes=[kernel_block_size])

    assert {
        part.untyped_storage().data_ptr() for components in caches.values() for part in components
    } == {raw.untyped_storage().data_ptr()}

    draft_k, draft_v = caches["draft.layers.0.self_attn.attn"]
    expected_blocks = 3 * (4 // kernel_block_size)
    assert draft_k.shape == (expected_blocks, kernel_block_size, 2, 8)
    assert draft_v.shape == (expected_blocks, kernel_block_size, 2, head_size_v)
    assert draft_k.is_contiguous() and draft_v.is_contiguous()
    assert draft_v.data_ptr() == draft_k.data_ptr() + draft_k.nbytes

    snapshot = raw.clone()
    for name, parts in caches.items():
        for part in parts:
            part.view(torch.uint8).fill_(77)
            for other_name, other_parts in caches.items():
                for other_part in other_parts:
                    if other_part.data_ptr() != part.data_ptr():
                        assert bool((other_part.view(torch.uint8) == 33).all()), (
                            f"Write into {name} bled into {other_name}"
                        )
            raw.copy_(snapshot)


@pytest.mark.parametrize(
    "scenario",
    [
        "kimi_linear",
        "glm5_next_text",
        "store_on_host",
        "kvpp",
        "blhnc_non_dense_main",
        "unmigrated_draft_backend",
        "padded_overlay_draft",
    ],
)
def test_specialized_or_incompatible_layouts_route_to_legacy_allocator(monkeypatch, scenario):
    from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionBackend

    model_type = scenario if scenario in ("kimi_linear", "glm5_next_text") else "glm_moe_dsa"
    layout = KVCacheLayout.BLHNC if scenario == "blhnc_non_dense_main" else KVCacheLayout.LBNHC
    runtime = _make_glm_runtime(model_type=model_type, layout=layout, enable_kvpp=(scenario == "kvpp"))
    main = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        num_query_heads=16,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        store_on_host=(scenario == "store_on_host"),
    )
    specs: dict[str, AscendMLAAttentionSpec | AscendSFAIndexerCacheSpec | FullAttentionSpec] = {
        "main.0": main,
        "main.1": main,
    }
    if scenario == "unmigrated_draft_backend":
        specs["draft.0"] = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16)
    elif scenario == "padded_overlay_draft":
        specs["draft.0"] = FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
            page_size_padded=256,
        )
    config = _build_glm_plan(runtime, specs, num_blocks=3)
    layers = {
        name: SimpleNamespace(
            get_attn_backend=lambda name=name, spec=spec: (
                CacheOnlyAttentionBackend
                if scenario == "unmigrated_draft_backend" and name == "draft.0"
                else (AscendAttentionBackend if isinstance(spec, FullAttentionSpec) else AscendSFABackend)
            ),
            num_heads=1,
        )
        for name, spec in specs.items()
    }
    runtime.compilation_config = SimpleNamespace(static_forward_context=layers)
    legacy = MagicMock(return_value={})
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", legacy)
    monkeypatch.setattr(attn_utils, "_reshape_kv_cache_v2", lambda **kw: kw["kv_cache_raw_tensors"])

    sizes = [group.kv_cache_spec.block_size for group in config.kv_cache_groups]
    assert attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), layout, sizes) is legacy.return_value
    legacy.assert_called_once()


@pytest.mark.parametrize("model_type", ["deepseek_v41", "glm_moe_dsa"])
def test_unified_models_fail_fast_on_mismatched_descriptor_sizes(monkeypatch, model_type):
    """Unified models must never silently fall back to legacy allocation on invalid descriptors."""
    runtime = _make_glm_runtime(model_type=model_type)
    spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
    )
    layers = {name: SimpleNamespace(get_attn_backend=lambda: AscendSFABackend, num_heads=1) for name in ("l0", "l1")}
    runtime.compilation_config = SimpleNamespace(static_forward_context=layers)
    bad_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=128, layers=["l0"], offset=0, layer_stride=128, block_stride=64),
            KVCacheTensor(size=256, layers=["l1"], offset=0, layer_stride=256, block_stride=64),
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0", "l1"], kv_cache_spec=spec)],
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", MagicMock(side_effect=AssertionError("unexpected fallback")))
    with pytest.raises(ValueError, match="share one backing allocation"):
        attn_utils.allocate_kv_cache_main(bad_config, torch.device("cpu"), None, [4])


@pytest.mark.parametrize("offset,message", [(1, "aligned"), (900, "exceeds")])
def test_contiguous_component_alignment_and_bounds(offset, message):
    raw = torch.empty(1024, dtype=torch.int8)
    with pytest.raises(ValueError, match=message):
        attn_utils.create_contiguous_cache_component_views(raw[offset:], [(3, 4, 1, 16)], [torch.float16])


@pytest.mark.parametrize("layout,expected_pages", [(KVCacheLayout.LBHNC, (3, 1)), (KVCacheLayout.BLHNC, (1, 2))])
def test_main_cache_upstream_placement_uses_physical_heads(layout, expected_pages):
    spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
        num_query_heads=16,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
    )
    layer_stride, block_stride, *_ = compute_layout_strides(spec, 3, 2, layout)
    assert spec.num_heads == 1
    assert spec.num_query_heads == 16
    page_bytes = spec.block_size * spec.head_size * 2
    assert spec.page_size_bytes == page_bytes
    assert (layer_stride, block_stride) == tuple(pages * page_bytes for pages in expected_pages)


def _register_mooncake_worker(config, caches, monkeypatch, kvpp_size=1):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker

    worker = base_worker.MooncakeBaseConnectorWorker.__new__(base_worker.MooncakeBaseConnectorWorker)
    worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))
    worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=kvpp_size))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = config
    worker.engine_id, worker.te_rpc_port = "test", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 4, "127.0.0.1", 5000
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    worker.register_kv_caches(caches)
    return worker.xfer_handshake_metadata


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
def test_mooncake_exports_per_component_addresses_for_glm(connector, monkeypatch):
    runtime = _make_glm_runtime(connector=connector)
    specs = {
        "main.0": AscendMLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=12,
            num_query_heads=16,
            dtype=torch.float16,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
        ),
        "indexer.0": AscendSFAIndexerCacheSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        ),
    }
    config = _build_glm_plan(runtime, specs, num_blocks=3)
    _, caches = _allocate_unified(config, runtime)
    metadata = _register_mooncake_worker(config, caches, monkeypatch)
    assert set(metadata.layer_names) == set(caches)
    for index, name in enumerate(metadata.layer_names):
        components = caches[name]
        assert metadata.kv_caches_base_addr[index] == [part.data_ptr() for part in components]
        assert metadata.block_strides[index] == [part.stride(0) * part.element_size() for part in components]
        assert metadata.block_lens[index] == [part[0].numel() * part.element_size() for part in components]


@pytest.mark.parametrize("li_c4", [False, True], ids=["li_c8", "li_c4"])
@pytest.mark.parametrize("decode_blocks", [1, 3], ids=["single_block_decode", "multi_block_decode"])
def test_mooncake_asymmetric_kvpp_prefill_to_unified_decode(monkeypatch, li_c4, decode_blocks):
    from tests.ut.kv_offload.mooncake_v2.helpers import make_pp_metadata
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

    runtime = _make_glm_runtime(connector="MooncakePullConnector")
    spec = AscendSFAIndexerCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8 if li_c4 else torch.int8,
        scale_dim=4 if li_c4 else 1,
        scale_dtype=torch.float8_e8m0fnu if li_c4 else torch.float16,
        cache_sparse_li_c8=not li_c4,
        cache_sparse_li_c4=li_c4,
    )
    decode_config = _build_glm_plan(runtime, {"indexer.0": spec}, num_blocks=decode_blocks)
    raw, caches = _allocate_unified(decode_config, runtime)
    decode = _register_mooncake_worker(decode_config, caches, monkeypatch)
    key, scale = caches["indexer.0"]

    prefill_config = _build_glm_plan(runtime, {"indexer.0": spec}, num_blocks=3)
    prefill_key = torch.full((3, *key.shape[1:]), 7, dtype=key.dtype)
    prefill_scale = (
        torch.arange(3 * scale[0].numel() * scale.element_size(), dtype=torch.uint8)
        .view(scale.dtype)
        .reshape(3, *scale.shape[1:])
    )
    monkeypatch.setattr(base_worker, "map_kvpp_layers_to_owners", lambda *_: {"indexer.0": 0})
    prefill = _register_mooncake_worker(
        prefill_config, {"indexer.0": (prefill_key, prefill_scale)}, monkeypatch, kvpp_size=2
    )

    dst_block = decode_blocks - 1
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.kv_cache_specs = [spec]
    thread.layer_names = decode.layer_names
    thread.kv_caches_base_addr = decode.kv_caches_base_addr
    thread.block_lens = decode.block_lens
    thread.block_strides = decode.block_strides
    remote = make_pp_metadata(
        tp_base_addrs={0: prefill.kv_caches_base_addr},
        block_lens=prefill.block_lens,
        block_strides=prefill.block_strides,
    )
    local_addrs: list[int] = []
    remote_addrs: list[int] = []
    lengths: list[int] = []
    thread._append_spec_transfer_addresses(
        0, 0, 0, 1, 1, 1, {(0, 0): [("request", [dst_block], [1])]}, remote, local_addrs, remote_addrs, lengths
    )
    assert local_addrs == [key[dst_block].data_ptr(), scale[dst_block].data_ptr()]
    assert remote_addrs == [prefill_key[1].data_ptr(), prefill_scale[1].data_ptr()]
    assert lengths == [key[dst_block].nbytes, scale[dst_block].nbytes]
    for local, source, length in zip(local_addrs, remote_addrs, lengths, strict=True):
        ctypes.memmove(local, source, length)
    assert torch.equal(key[dst_block], prefill_key[1])
    assert torch.equal(scale[dst_block].view(torch.uint8), prefill_scale[1].view(torch.uint8))
    if dst_block > 0:
        assert torch.all(key[:dst_block] == 33)
        assert torch.all(scale[:dst_block].view(torch.uint8) == 33)


def test_ascend_attention_runtime_shapes_rejects_subclasses_and_hnd(monkeypatch):
    import vllm.envs as envs_vllm

    class _DerivedAttentionBackend(AscendAttentionBackend):
        pass

    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="does not implement unified runtime shapes"):
        _DerivedAttentionBackend.get_runtime_shapes(spec, 3, None)
    monkeypatch.setattr(envs_vllm, "VLLM_KV_CACHE_LAYOUT", "LBHNC")
    with pytest.raises(ValueError, match="HND layout"):
        AscendAttentionBackend.get_runtime_shapes(spec, 3, None)


def test_ascend_store_asymmetric_kvpp_prefill_matches_unified_decode_segments():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    runtime = _make_glm_runtime(connector="AscendStoreConnector")
    indexer = AscendSFAIndexerCacheSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
        sfa_dcp_replicated_indexer_size=4,
    )
    config = _build_glm_plan(runtime, {"indexer.0": indexer}, num_blocks=3)
    _, decode_caches = _allocate_unified(config, runtime, kernel_block_sizes=[4])
    decode_key, decode_scale = decode_caches["indexer.0"]

    # KVPP prefill allocates separate contiguous component buffers.
    prefill_key = torch.empty_like(decode_key)
    prefill_scale = torch.empty_like(decode_scale)

    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.num_blocks = 3
    for prefill_part, decode_part in ((prefill_key, decode_key), (prefill_scale, decode_scale)):
        prefill_segments = worker._get_cache_transfer_segments(prefill_part)
        decode_segments = worker._get_cache_transfer_segments(decode_part)
        assert len(prefill_segments) == len(decode_segments) == 1
        assert prefill_segments[0][1:] == decode_segments[0][1:]
