# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare Step5 V2 cache allocation and view geometry on CPU."""

from types import SimpleNamespace

import pytest
import torch
from vllm import envs
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import KV_CACHE_BUFFER_ALIGNMENT
from vllm_ascend.models.step5.cache_config import (
    get_step5_kv_cache_config_from_groups,
    get_step5_kv_cache_groups,
    get_step5_pool_bytes_per_block,
)
from vllm_ascend.models.step5.sparse_attention import (
    AscendStep5CSATailRing,
    AscendStep5DSABackend,
    AscendStep5SparseSummaryBackend,
)

BLOCK = 128
MAIN_PAGE = 98304
SUMMARY_PAGE = 4096
# The CSA tail ring mirrors the main page class (96 KiB, capacity 96 rows).
RING_PAGE = MAIN_PAGE

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
        block_stride_alignment=1024,
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


def _plan(num_blocks, layout, with_aux):
    vllm_config = SimpleNamespace(
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
            get_resolved_kv_cache_layout=lambda: layout,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={},
        kv_transfer_config=None,
    )
    if not with_aux:
        # This worker only owns sliding-window layers from the global plan.
        groups = [KVCacheGroupSpec([L0, L1], _specs()[L1])]
        config = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(vllm_config, groups, num_blocks)
        return vllm_config, groups, config
    groups = get_step5_kv_cache_groups(vllm_config, _specs(), _orig_groups)
    config = get_step5_kv_cache_config_from_groups(
        vllm_config, groups, num_blocks * get_step5_pool_bytes_per_block(groups)
    )
    return vllm_config, groups, config


def _snapshot(caches, backing):
    return {
        name: (
            isinstance(cache, tuple),
            [
                (
                    tensor.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr(),
                    tensor.data_ptr() - backing.data_ptr(),
                    tuple(tensor.shape),
                    tensor.dtype,
                    tensor.stride(),
                )
                for tensor in (cache if isinstance(cache, tuple) else (cache,))
            ],
        )
        for name, cache in caches.items()
    }


@pytest.mark.parametrize(
    ("num_blocks", "kv_transfer", "layout", "with_aux"),
    [
        (1, False, KVCacheLayout.LBNHC, True),
        (3, False, KVCacheLayout.LBNHC, True),
        (3, True, KVCacheLayout.LBHNC, True),
        (3, False, KVCacheLayout.LBNHC, False),
    ],
)
def test_step5_shared_allocator_matches_legacy_views(monkeypatch, num_blocks, kv_transfer, layout, with_aux):
    runtime, groups, config = _plan(num_blocks, layout, with_aux)
    runtime.kv_transfer_config = object() if kv_transfer else None
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_LAYOUT", layout.name)
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils_v2, "get_shared_kv_cache_layers", lambda _: {})
    ring = AscendStep5CSATailRing.__new__(AscendStep5CSATailRing)
    backends = {
        L0: AscendStep5DSABackend,
        L2: AscendStep5DSABackend,
        L1: AscendAttentionBackend,
        RING: ring.get_attn_backend(),
        SUMMARY0: AscendStep5SparseSummaryBackend,
        SUMMARY2: AscendStep5SparseSummaryBackend,
    }
    if not with_aux:
        backends = {name: AscendAttentionBackend for name in (L0, L1)}
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda backend=backend: backend) for name, backend in backends.items()
    }
    monkeypatch.setattr(attn_utils_v2, "get_layers_from_vllm_config", lambda *_: layers)
    allocations = []
    allocate = attn_utils_v2._allocate_int8_cache_tensor

    def record_allocation(size, alignment, device):
        backing = allocate(size, alignment, device)
        allocations.append(backing)
        return backing

    monkeypatch.setattr(attn_utils_v2, "_allocate_int8_cache_tensor", record_allocation)
    device = torch.device("cpu")
    block_sizes = [BLOCK] * len(groups)
    raw = attn_utils_v2._allocate_kv_cache(config, {}, device)
    legacy = attn_utils_v2._reshape_kv_cache_v2(
        attn_utils_v2._get_cache_attention_groups(config, layers, {}),
        raw,
        runtime.cache_config.cache_dtype,
        block_sizes,
        {},
        config,
    )
    (legacy_backing,) = allocations
    allocations.clear()

    def reject_legacy(*args, **kwargs):
        pytest.fail("Step5 should use the shared V2 allocator")

    monkeypatch.setattr(attn_utils_v2, "_allocate_kv_cache", reject_legacy)
    monkeypatch.setattr(attn_utils_v2, "_reshape_kv_cache_v2", reject_legacy)
    actual = attn_utils_v2.allocate_kv_cache_main(config, device, layout, block_sizes)
    (backing,) = allocations
    assert backing.nbytes == legacy_backing.nbytes == config.kv_cache_tensors[0].size
    assert backing.untyped_storage().nbytes() == legacy_backing.untyped_storage().nbytes()
    if kv_transfer:
        assert backing.data_ptr() % KV_CACHE_BUFFER_ALIGNMENT == 0
        assert legacy_backing.data_ptr() % KV_CACHE_BUFFER_ALIGNMENT == 0

    assert actual.keys() == legacy.keys() == backends.keys()
    attention_names = (L0, L1, L2) if with_aux else (L0, L1)
    assert all(isinstance(actual[name], tuple) and len(actual[name]) == 2 for name in attention_names)
    if with_aux:
        assert all(isinstance(actual[name], torch.Tensor) for name in (RING, SUMMARY0, SUMMARY2))
    views = _snapshot(actual, backing)
    assert views == _snapshot(legacy, legacy_backing)
    assert all(shares_backing for _, components in views.values() for shares_backing, *_ in components)
