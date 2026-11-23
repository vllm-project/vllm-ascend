# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Byte-oracle CPU tests for Step5 V2 unified KV cache allocation and binding."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_ascend.worker.v2.attn_utils as attn_utils_v2
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import KV_CACHE_BUFFER_ALIGNMENT
from vllm_ascend.models.step5.cache_config import (
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_pool_bytes_per_block,
)
from vllm_ascend.models.step5.model import AscendStep5DSAAttention
from vllm_ascend.models.step5.sparse_attention import (
    AscendStep5CSATailRing,
    AscendStep5DSABackend,
    AscendStep5SparseSummaryCacheLayer,
)
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers

BLOCK = 128
MAIN_PAGE = 98304
SUMMARY_PAGE = 4096
# The CSA tail ring mirrors the main page class (96 KiB, capacity 96 rows).
RING_PAGE = MAIN_PAGE
RING_ROW_BYTES = 1024

L0 = "model.layers.0.self_attn"
L1 = "model.layers.1.self_attn"
L2 = "model.layers.2.self_attn"
RING = f"{L0}.csa_tail_ring"
SUMMARY0 = f"{L0}.sparse_summary_cache"
SUMMARY2 = f"{L2}.sparse_summary_cache"


def _specs():
    main = FullAttentionSpec(block_size=BLOCK, num_kv_heads=4, head_size=48, head_size_v=48, dtype=torch.bfloat16)
    sliding = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=48,
        head_size_v=48,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )
    summary = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=256, head_size_v=0, dtype=torch.uint8, tokens_per_state=8
    )
    ring = CircularBufferSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=384,
        head_size_v=0,
        dtype=torch.bfloat16,
        block_stride_alignment=RING_ROW_BYTES,
    )
    return {
        L0: main,
        L2: main,
        L1: sliding,
        SUMMARY0: summary,
        SUMMARY2: summary,
        RING: ring,
    }


def _orig_groups(vllm_config, kv_cache_spec):
    full = [name for name, spec in kv_cache_spec.items() if type(spec) is FullAttentionSpec]
    sliding = [name for name, spec in kv_cache_spec.items() if isinstance(spec, SlidingWindowSpec)]
    groups = [KVCacheGroupSpec(full, UniformTypeKVCacheSpecs.from_specs({n: kv_cache_spec[n] for n in full}))]
    groups.append(KVCacheGroupSpec(sliding, UniformTypeKVCacheSpecs.from_specs({n: kv_cache_spec[n] for n in sliding})))
    return groups


def _plan(num_blocks: int, layout: KVCacheLayout, with_aux: bool, model_type: str = "step5"):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=SimpleNamespace(model_type=model_type),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=8192,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_blocks,
            prefix_cache_retention_interval=None,
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: layout,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={},
        kv_transfer_config=None,
    )
    if not with_aux:
        # This worker only owns sliding-window layers from the global plan.
        groups = [KVCacheGroupSpec([L0, L1], _specs()[L1])]
        config = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(vllm_config, groups, num_blocks)
        return vllm_config, groups, config
    groups = get_step5_kv_cache_groups(vllm_config, _specs(), _orig_groups)
    config = get_step5_kv_cache_config_from_groups(
        vllm_config, groups, num_blocks * get_step5_pool_bytes_per_block(groups)
    )
    return vllm_config, groups, config


def _make_dense_attention() -> Attention:
    dense = Attention.__new__(Attention)
    dense.attn_backend = AscendAttentionBackend
    dense.kv_cache = torch.empty(0)
    return dense


@pytest.mark.parametrize(
    ("num_blocks", "kv_transfer", "layout", "with_aux", "model_type"),
    [
        (1, False, KVCacheLayout.LBNHC, True, "step5"),
        (3, False, KVCacheLayout.LBNHC, True, "step5"),
        (3, True, KVCacheLayout.LBHNC, True, "step5_vl"),
        (3, False, KVCacheLayout.LBNHC, False, "step5_mtp"),
        (3, False, KVCacheLayout.LBHNC, False, "step5"),
    ],
)
def test_step5_unified_allocator_byte_geometry_and_binding(
    monkeypatch, num_blocks, kv_transfer, layout, with_aux, model_type
):
    runtime, groups, config = _plan(num_blocks, layout, with_aux, model_type=model_type)
    runtime.kv_transfer_config = object() if kv_transfer else None
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils_v2, "get_shared_kv_cache_layers", lambda _: {})

    layers = {L1: _make_dense_attention()}
    if with_aux:
        dsa0 = AscendStep5DSAAttention.__new__(AscendStep5DSAAttention)
        dsa0.attn_backend = AscendStep5DSABackend
        dsa2 = AscendStep5DSAAttention.__new__(AscendStep5DSAAttention)
        dsa2.attn_backend = AscendStep5DSABackend
        ring = AscendStep5CSATailRing.__new__(AscendStep5CSATailRing)
        layers.update({L0: dsa0, L2: dsa2, RING: ring})
        for name in (SUMMARY0, SUMMARY2):
            summary = AscendStep5SparseSummaryCacheLayer.__new__(AscendStep5SparseSummaryCacheLayer)
            summary._target_impl = SimpleNamespace()
            layers[name] = summary
    else:
        layers[L0] = _make_dense_attention()
    monkeypatch.setattr(attn_utils_v2, "get_layers_from_vllm_config", lambda *_: layers)

    allocations = []
    allocate = attn_utils_v2._allocate_int8_cache_tensor

    def record_allocation(size, alignment, device):
        backing = allocate(size, alignment, device)
        allocations.append(backing)
        return backing

    def reject_legacy(*args, **kwargs):
        pytest.fail("Step5 must use the unified V2 allocator")

    monkeypatch.setattr(attn_utils_v2, "_allocate_int8_cache_tensor", record_allocation)
    monkeypatch.setattr(attn_utils_v2, "_allocate_kv_cache", reject_legacy)
    monkeypatch.setattr(attn_utils_v2, "_reshape_kv_cache_v2", reject_legacy)

    block_sizes = [BLOCK] * len(groups)
    caches = attn_utils_v2.allocate_kv_cache_main(config, torch.device("cpu"), layout, block_sizes)

    (backing,) = allocations
    assert backing.nbytes == config.kv_cache_tensors[0].size
    if kv_transfer:
        assert backing.data_ptr() % KV_CACHE_BUFFER_ALIGNMENT == 0

    assert caches.keys() == layers.keys()
    placements = {
        name: (descriptor.offset + idx * descriptor.layer_stride, descriptor.block_stride)
        for descriptor in config.kv_cache_tensors
        for idx, name in enumerate(descriptor.layers)
    }

    dense_shape = (num_blocks, 4, BLOCK, 48) if layout is KVCacheLayout.LBHNC else (num_blocks, BLOCK, 4, 48)
    dsa_shape = (num_blocks, BLOCK, 4, 48)
    dense_names = (L1,) if with_aux else (L0, L1)
    dsa_names = (L0, L2) if with_aux else ()

    for name in dense_names:
        k_view, v_view = caches[name]
        assert isinstance(caches[name], tuple) and len(caches[name]) == 2
        offset, block_stride = placements[name]
        assert block_stride == MAIN_PAGE
        for view, component_offset in ((k_view, 0), (v_view, MAIN_PAGE // 2)):
            assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            assert view.data_ptr() - backing.data_ptr() == offset + component_offset
            assert tuple(view.shape) == dense_shape
            assert view.dtype is torch.bfloat16
            assert view.stride(0) * view.element_size() == MAIN_PAGE

    for name in dsa_names:
        k_view, v_view = caches[name]
        assert isinstance(caches[name], tuple) and len(caches[name]) == 2
        offset, block_stride = placements[name]
        assert block_stride == MAIN_PAGE
        for view, component_offset in ((k_view, 0), (v_view, MAIN_PAGE // 2)):
            assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            assert view.data_ptr() - backing.data_ptr() == offset + component_offset
            assert tuple(view.shape) == dsa_shape
            assert view.dtype is torch.bfloat16
            assert view.stride(0) * view.element_size() == MAIN_PAGE

    if with_aux:
        ring_view = caches[RING]
        ring_offset, ring_stride = placements[RING]
        assert isinstance(ring_view, torch.Tensor)
        assert ring_view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        assert ring_view.data_ptr() - backing.data_ptr() == ring_offset
        assert tuple(ring_view.shape) == (num_blocks, 1, BLOCK, 384)
        assert ring_view.dtype is torch.bfloat16
        assert ring_view.stride(0) * ring_view.element_size() == ring_stride == RING_PAGE

        for name in (SUMMARY0, SUMMARY2):
            summary_view = caches[name]
            summary_offset, summary_stride = placements[name]
            assert isinstance(summary_view, torch.Tensor)
            assert summary_view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
            assert summary_view.data_ptr() - backing.data_ptr() == summary_offset
            assert tuple(summary_view.shape) == (num_blocks, 1, BLOCK // 8, 256)
            assert summary_view.dtype is torch.uint8
            assert summary_view.stride(0) * summary_view.element_size() == summary_stride == SUMMARY_PAGE

        # Byte write-through oracle: writing to a block's K, V, and summary lands at exact backing offsets.
        last_block = num_blocks - 1
        k0, v0 = caches[L0]
        k0[last_block].fill_(1.5)
        v0[last_block].fill_(-2.0)
        caches[SUMMARY0][last_block].fill_(7)
        l0_offset, _ = placements[L0]
        s0_offset, _ = placements[SUMMARY0]
        k_start = l0_offset + last_block * MAIN_PAGE
        v_start = k_start + MAIN_PAGE // 2
        s_start = s0_offset + last_block * SUMMARY_PAGE
        assert torch.all(backing[k_start : k_start + MAIN_PAGE // 2].view(torch.bfloat16) == 1.5)
        assert torch.all(backing[v_start : v_start + MAIN_PAGE // 2].view(torch.bfloat16) == -2.0)
        assert torch.all(backing[s_start : s_start + SUMMARY_PAGE].view(torch.uint8) == 7)

    # Tuple K/V views must not invoke tensor-only Attention.bind_kv_cache hooks.
    for name in dense_names:
        layers[name].bind_kv_cache = lambda _cache: pytest.fail("Dense Attention.bind_kv_cache must not be called")

    bind_kv_cache_to_layers(caches, layers, kv_cache_groups=groups)
    for name in (*dense_names, *dsa_names):
        assert layers[name].kv_cache is caches[name]
    if with_aux:
        assert layers[RING].rows.data_ptr() == caches[RING].data_ptr()
        assert layers[RING].capacity == RING_PAGE // RING_ROW_BYTES
        assert layers[RING].rows_per_block == RING_PAGE // RING_ROW_BYTES
        for name in (SUMMARY0, SUMMARY2):
            summary_cache = layers[name]._target_impl.summary_cache
            assert summary_cache is not None
            assert summary_cache.mean.data_ptr() == caches[name].data_ptr()
            assert tuple(summary_cache.mean.shape) == (num_blocks, BLOCK // 8, 256)


def test_unified_attention_splits_kernel_blocks_within_manager_pages(monkeypatch):
    """Verify backend runtime shapes split multi-kernel-block pages with exact byte strides."""
    num_blocks = 2
    manager_block_size = 256
    kernel_block_size = 128
    spec = SlidingWindowSpec(
        block_size=manager_block_size,
        num_kv_heads=2,
        head_size=32,
        head_size_v=32,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )
    groups = [KVCacheGroupSpec([L1], spec)]
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="step5"),
            hf_text_config=SimpleNamespace(model_type="step5"),
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=8192,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_blocks,
            prefix_cache_retention_interval=None,
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={},
        kv_transfer_config=None,
    )
    config = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks)
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils_v2, "get_shared_kv_cache_layers", lambda _: {})
    monkeypatch.setattr(attn_utils_v2, "get_layers_from_vllm_config", lambda *_: {L1: _make_dense_attention()})

    caches = attn_utils_v2.allocate_kv_cache_main(config, torch.device("cpu"), KVCacheLayout.LBNHC, [kernel_block_size])
    k_view, v_view = caches[L1]
    kernel_stride = spec.page_size_bytes // 2
    assert tuple(k_view.shape) == (num_blocks * 2, kernel_block_size, 2, 32)
    assert tuple(v_view.shape) == (num_blocks * 2, kernel_block_size, 2, 32)
    assert k_view.stride(0) * k_view.element_size() == kernel_stride
    assert v_view.stride(0) * v_view.element_size() == kernel_stride
    assert v_view.data_ptr() - k_view.data_ptr() == kernel_stride // 2
