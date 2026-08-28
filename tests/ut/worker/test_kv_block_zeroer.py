# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.worker.utils import AscendKVBlockZeroer
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _group(spec, names, group_id=0):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=names, kv_cache_group_id=group_id)


def _init_zeroer(groups, caches, kernel_sizes, excluded=None):
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    zeroer.init_meta(
        groups,
        kernel_sizes,
        "auto",
        set() if excluded is None else excluded,
        {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()},
    )
    return zeroer


def test_mla_backend_accepts_upstream_cache_dtype_keyword():
    assert AscendMLABackend.get_kv_cache_block_dim(128, 1, 576, cache_dtype_str="auto") == 0
    assert AscendMLABackend.get_kv_cache_shape(2, 128, 1, 576, cache_dtype_str="auto") == (2, 128, 1, 576)


@pytest.mark.parametrize("spec_cls", [AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec])
@pytest.mark.parametrize("needs_zeroing", [False, True])
def test_ascend_attention_managers_record_new_blocks_for_zeroing(spec_cls, needs_zeroing):
    register_ascend_kv_cache_specs()
    spec = spec_cls(block_size=384, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    manager = get_manager_for_kv_cache_spec(
        spec,
        max_in_flight_tokens=512,
        max_model_len=4096,
        block_pool=BlockPool(16, enable_caching=False, hash_block_size=384),
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=384,
        needs_kv_cache_zeroing=needs_zeroing,
    )
    blocks = manager.allocate_new_blocks("request", 385, 385)
    assert len(blocks) == 2
    assert manager.take_new_block_ids() == ([block.block_id for block in blocks] if needs_zeroing else [])
    assert manager.take_new_block_ids() == []


def test_mixed_mla_gqa_pages_skip_empty_shared_and_mamba_views():
    mla_spec = AscendMLAAttentionSpec(block_size=384, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    gqa_spec = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    mamba_spec = MambaSpec(block_size=384, shapes=((2, 4),), dtypes=(torch.float32,))
    k = torch.empty(6, 128, 1, 512, dtype=torch.bfloat16)
    rope = torch.empty(6, 128, 1, 64, dtype=torch.bfloat16)
    gqa_k = torch.empty(2, 128, 2, 64, dtype=torch.int8)
    gqa_v = torch.empty(2, 128, 2, 64, dtype=torch.bfloat16)
    groups = [
        _group(mla_spec, ["mla", "shared", "excluded", "no_rope"]),
        _group(gqa_spec, ["gqa"], 1),
        _group(mamba_spec, ["mamba"], 2),
        _group(gqa_spec, ["unallocated"], 3),
    ]
    zeroer = _init_zeroer(
        groups,
        {
            "mla": (k, rope),
            "shared": (k, rope),
            "no_rope": (k, torch.empty(6, 128, 1, 0)),
            "gqa": (gqa_k, gqa_v),
        },
        [[128], [128], [384]],
        excluded={"excluded"},
    )
    addrs, sizes, strides, max_chunks, chunk_size, num_segments = zeroer._meta
    assert addrs.tolist() == [cache.data_ptr() for cache in (k, rope, gqa_k, gqa_v)]
    assert sizes.tolist() == [384 * 512 // 2, 384 * 64 // 2, 128 * 2 * 64 // 4, 128 * 2 * 64 // 2]
    assert torch.equal(sizes, strides)
    assert num_segments == 4
    assert max_chunks * chunk_size == max(sizes.tolist())


@pytest.mark.parametrize("ratio", [1, 3])
def test_strided_views_keep_payload_separate_from_scheduler_stride(ratio):
    spec = FullAttentionSpec(block_size=4 * ratio, num_kv_heads=1, head_size=2, dtype=torch.float32)
    raw = torch.empty(2 * ratio * 24 + 4, dtype=torch.float32)
    # Distinct components share storage, with a leading offset and guard holes.
    k = torch.as_strided(raw, (2 * ratio, 4, 1, 2), (24, 2, 2, 1), storage_offset=4)
    v = torch.as_strided(raw, (2 * ratio, 4, 1, 2), (24, 2, 2, 1), storage_offset=12)
    zeroer = _init_zeroer([_group(spec, ["attn"])], {"attn": (k, v)}, [[4]])
    addrs, sizes, strides, _, _, num_segments = zeroer._meta
    assert addrs.tolist() == [cache.data_ptr() + i * 24 * 4 for cache in (k, v) for i in range(ratio)]
    assert sizes.tolist() == [8] * (2 * ratio)
    assert strides.tolist() == [24 * ratio] * (2 * ratio)
    assert num_segments == 2 * ratio


def test_compressed_cache_uses_physical_block_size():
    spec = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, compress_ratio=4)
    cache = torch.empty(2, 32, 1, 8, dtype=torch.bfloat16)
    zeroer = _init_zeroer([_group(spec, ["attn"])], {"attn": (cache,)}, [[128]])
    assert zeroer._meta[1].tolist() == [32 * 8 // 2]


def test_zeroer_empty_cache_is_noop():
    zeroer = _init_zeroer([], {}, [])
    assert zeroer._meta is None
    with patch("vllm_ascend.worker.utils._zero_kv_blocks_kernel") as kernel:
        zeroer.zero_block_ids([0])
        zeroer.zero_block_ids([])
        kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "cache",
    [torch.empty(2, 1, 1, 1, dtype=torch.int8), torch.empty(2, 4, 2, 2).transpose(1, 2)],
    ids=["unaligned-payload", "noncontiguous-inner-dimensions"],
)
def test_zeroer_rejects_unsupported_component_layout(cache):
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32)
    with pytest.raises(AssertionError):
        _init_zeroer([_group(spec, ["attn"])], {"attn": (cache,)}, [[4]])


def test_zeroer_reuses_block_id_buffers_and_dispatches_all_segments():
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32)
    k, v = torch.empty(3, 4, 1, 2), torch.empty(3, 4, 1, 2)
    zeroer = _init_zeroer([_group(spec, ["attn"])], {"attn": (k, v)}, [[4]])
    ids = zeroer._ids_gpu
    with (
        patch("vllm_ascend.worker.utils._zero_kv_blocks_kernel") as kernel,
        patch("vllm_ascend.worker.utils.get_vectorcore_num", return_value=8),
    ):
        zeroer.zero_block_ids([])
        kernel.__getitem__.assert_not_called()
        zeroer.warmup(3)
        zeroer.zero_block_ids([1, 2])
        assert zeroer._ids_gpu is ids
        assert ids[:2].tolist() == [1, 2]
        args, kwargs = kernel.__getitem__.return_value.call_args
        assert args[4] == 2
        assert kwargs["N_SEGS"] == 2
        assert kwargs["MAX_CHUNKS"] == 1


def test_mrv2_initializes_ascend_zeroer_on_the_upstream_attribute():
    group = object()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        attn_groups=[[group]],
        kernel_block_sizes=[128],
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with patch("vllm_ascend.worker.v2.model_runner.AscendKVBlockZeroer") as zeroer_cls:
        NPUModelRunner._init_kv_zero_meta(runner)
    assert runner.kv_block_zeroer is zeroer_cls.return_value
    kwargs = runner.kv_block_zeroer.init_meta.call_args.kwargs
    assert list(kwargs["attn_groups_iter"]) == [group]
    assert kwargs["kernel_block_sizes"] == [[128]]
    assert kwargs["static_forward_context"] is runner.compilation_config.static_forward_context
