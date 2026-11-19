# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, compute_layout_strides
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.worker.v2 import attn_utils


def make_config():
    main = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=12,
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
    return KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(size=1024, layers=["main.0", "main.1"], offset=32, layer_stride=96, block_stride=320),
            KVCacheTensor(size=1024, layers=["indexer.0"], offset=224, layer_stride=40, block_stride=320),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["main.0", "main.1"], kv_cache_spec=main),
            KVCacheGroupSpec(layer_names=["indexer.0"], kv_cache_spec=indexer),
        ],
    )


def allocate(config=None):
    config = config or make_config()
    raw = torch.full((1024,), 33, dtype=torch.int8)
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda backend=backend: backend)
        for group, backend in zip(config.kv_cache_groups, (AscendSFABackend, AscendSFAIndexerBackend))
        for name in group.layer_names
    }
    with (
        patch.object(attn_utils, "get_current_vllm_config"),
        patch.object(attn_utils, "get_shared_kv_cache_layers", return_value={}),
        patch.object(attn_utils, "get_layers_from_vllm_config", return_value=layers),
        patch.object(attn_utils, "_allocate_int8_cache_tensor", return_value=raw) as allocator,
    ):
        caches = attn_utils.unified_allocate_kv_cache(config, torch.device("cpu"), None, [4, 4])
        allocator.assert_called_once_with(1024, 2 * 1024 * 1024, torch.device("cpu"))
    return raw, caches


def test_component_views_preserve_page_format_and_storage():
    raw, caches = allocate()
    nope, rope = caches["main.0"]
    key, scale = caches["indexer.0"]
    assert type(caches["main.0"]) is tuple
    assert nope.shape == (3, 4, 1, 8)
    assert rope.shape == (3, 4, 1, 4)
    assert nope.stride() == (160, 8, 8, 1)
    assert key.stride() == (320, 8, 8, 1)
    assert scale.stride() == (160, 1, 1, 1)
    assert rope.data_ptr() - nope.data_ptr() == 64
    assert scale.data_ptr() - key.data_ptr() == 32
    assert caches["main.1"][0].data_ptr() - nope.data_ptr() == 96
    for components in caches.values():
        for component in components:
            assert component.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key[1, 2].fill_(7)
    scale[1, 2].fill_(0.5)
    expected = torch.full_like(raw, 33)
    expected[560:568] = 7
    expected[580:582] = torch.tensor([0.5], dtype=torch.float16).view(torch.int8)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("flag", ["cache_sparse_sfa_c8", "cache_sparse_sfa_turboquant"])
def test_packed_main_runtime_shape(flag):
    spec = replace(make_config().kv_cache_groups[0].kv_cache_spec, dtype=torch.int8, head_size=386, **{flag: True})
    assert AscendSFABackend.get_runtime_shapes(spec, 3, None) == ((3, 1, 4, 386),)
    assert AscendSFABackend.get_runtime_dtypes(spec) == (torch.int8,)


def test_c4_scale_axes_are_backend_owned():
    spec = replace(make_config().kv_cache_groups[1].kv_cache_spec, scale_dim=4, cache_sparse_li_c4=True)
    assert AscendSFAIndexerBackend.get_runtime_shapes(spec, 3, None)[1] == (3, 1, 4, 2, 4)


@pytest.mark.parametrize("offset,stride,message", [(1, 320, "aligned"), (0, 319, "aligned"), (900, 320, "exceeds")])
def test_component_alignment_and_bounds(offset, stride, message):
    raw = torch.empty(1024, dtype=torch.int8)
    with pytest.raises(ValueError, match=message):
        attn_utils.create_cache_component_views(raw[offset:], [(3, 4, 1, 8)], [torch.float16], stride)


@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize(
    "dtype,width,scale_dim,scale_dtype,c4",
    [
        (torch.bfloat16, 128, 0, torch.int8, False),
        (torch.int8, 128, 1, torch.float16, False),
        (torch.float8_e4m3fn, 128, 1, torch.float32, False),
        (torch.int8, 64, 4, torch.uint8, True),
    ],
)
def test_upstream_indexer_placement_includes_scales_and_replicas(dtype, width, scale_dim, scale_dtype, c4, replicas):
    spec = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=width,
        dtype=dtype,
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
        cache_sparse_li_c8=scale_dim > 0,
        cache_sparse_li_c4=c4,
        sfa_dcp_replicated_indexer_size=replicas,
    )
    strides = compute_layout_strides(spec, 3, 2, KVCacheLayout.LBHNC)
    page_bytes = (
        128
        * replicas
        * (
            width * torch.empty((), dtype=dtype).element_size()
            + scale_dim * torch.empty((), dtype=scale_dtype).element_size()
        )
    )
    assert strides[1] == page_bytes
    assert strides[0] == 3 * page_bytes
    assert spec.page_size_bytes == page_bytes
    # The final layer's final page must fit the upstream backing exactly.
    assert strides[0] + 2 * strides[1] + spec.page_size_bytes == 6 * page_bytes
