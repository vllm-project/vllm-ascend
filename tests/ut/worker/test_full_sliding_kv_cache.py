# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.attn_utils import (
    allocate_shared_attention_kv_cache,
    can_share_full_and_sliding_kv_cache,
)


def _config():
    layout = SimpleNamespace(is_layer_compact=True, is_block_compact=True)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="gemma4_text"), use_mla=False),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout, cache_dtype="auto"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        kv_transfer_config=None,
    )


def _cache_config(sliding_groups=5, dtype=torch.bfloat16):
    # Different head dimensions and block sizes, but equal physical pages.
    full = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=8, dtype=dtype)
    sliding = SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=dtype, sliding_window=8)
    assert full.page_size_bytes == sliding.page_size_bytes
    num_blocks = 8
    layer_size = num_blocks * full.page_size_bytes
    groups = []
    descriptors = []
    for group_index, spec in enumerate([full, *[sliding] * sliding_groups]):
        names = [f"model.layers.{2 * group_index + index}.self_attn.attn" for index in range(2)]
        groups.append(KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec))
        # Split the descriptor to exercise a nonzero offset as well as aliasing.
        for index, name in enumerate(names):
            descriptors.append(
                KVCacheTensor(
                    size=2 * layer_size,
                    layers=[name],
                    offset=index * layer_size,
                    layer_stride=layer_size,
                    block_stride=spec.page_size_bytes,
                )
            )
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=descriptors, kv_cache_groups=groups)


def _specs(cache_config):
    return {name: group.kv_cache_spec for group in cache_config.kv_cache_groups for name in group.layer_names}


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(model_config=SimpleNamespace()),
        SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace())),
    ],
)
def test_partial_model_configs_keep_existing_path(config):
    assert not can_share_full_and_sliding_kv_cache(config, _specs(_cache_config()).values())


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("sliding_groups", [4, 5])
@pytest.mark.parametrize("supports_shared", [False, True])
@pytest.mark.parametrize("model_type", ["gemma4_text", "other_full_sliding_model", "step3p5", "step3p7"])
def test_budget_tracks_physical_shared_allocation(monkeypatch, sliding_groups, supports_shared, model_type, use_v2):
    config = _config()
    config.model_config.hf_text_config.model_type = model_type
    cache = _cache_config(sliding_groups)
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.vllm_config = config
    worker.use_v2_model_runner = use_v2
    worker.model_config = config.model_config
    worker.get_kv_cache_spec = lambda: _specs(cache)
    worker.model_runner = SimpleNamespace(
        supports_standardized_shared_kv_backing=supports_shared, use_sparse=False, use_compress=False
    )
    monkeypatch.setattr(worker_module, "get_kv_cache_groups", lambda *_: cache.kv_cache_groups)
    # Divisible by both five and six; the unsupported runner must retain scaling.
    available = 60 * cache.kv_cache_tensors[0].size
    shares_pool = supports_shared and (model_type in ("step3p5", "step3p7") or (use_v2 and model_type == "gemma4_text"))
    expected = available if shares_pool else available // (sliding_groups + 1)
    assert worker._scale_kv_cache_memory_for_multi_group(available) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_block_size", [1, 2])
def test_v2_shares_storage_and_keeps_distinct_blocks_isolated(monkeypatch, dtype, kernel_block_size):
    config = _config()
    cache = _cache_config(dtype=dtype)
    backend = SimpleNamespace(
        get_kv_cache_shape=lambda blocks, block_size, heads, head_size, *_: (2, blocks, block_size, heads, head_size)
    )
    groups = [
        SimpleNamespace(
            kv_cache_group_id=index,
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
            backend=backend,
        )
        for index, group in enumerate(cache.kv_cache_groups)
    ]
    kernel_sizes = [kernel_block_size] * len(groups)
    allocate = Mock(side_effect=lambda size: torch.zeros(size, dtype=torch.int8))
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils.KVPPConfig, "from_vllm_config", lambda _: SimpleNamespace(size=1))
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", lambda size, *_: allocate(size))
    raw = attn_utils._allocate_kv_cache(cache, {}, torch.device("cpu"))
    reshaped = attn_utils._reshape_kv_cache_v2(
        groups, raw, "auto", kernel_sizes, {"shared_layer": groups[0].layer_names[0]}, cache
    )
    assert reshaped["shared_layer"] is reshaped[groups[0].layer_names[0]]
    allocate.assert_called_once_with(cache.kv_cache_tensors[0].size)
    assert len({tensor.untyped_storage().data_ptr() for pair in raw.values() for tensor in pair}) == 1
    specs = _specs(cache)
    views = {
        name: tuple(
            tensor.view(cache.num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
            for tensor in reshaped[name]
        )
        for name, spec in specs.items()
    }
    # Each group owns a distinct block ID in the common pool. All six groups
    # can write both K and V without modifying another group's live block.
    for group_id, group in enumerate(cache.kv_cache_groups):
        for slot, name in enumerate(group.layer_names):
            k_cache, v_cache = views[name]
            assert k_cache.is_contiguous() and v_cache.is_contiguous()
            k_cache[group_id].fill_(10 * group_id + slot + 1)
            v_cache[group_id].fill_(-10 * group_id - slot - 1)
    for group_id, group in enumerate(cache.kv_cache_groups):
        for slot, name in enumerate(group.layer_names):
            k_cache, v_cache = views[name]
            assert torch.all(k_cache[group_id] == 10 * group_id + slot + 1)
            assert torch.all(v_cache[group_id] == -10 * group_id - slot - 1)
            assert torch.count_nonzero(k_cache[-1]) == 0
            assert torch.count_nonzero(v_cache[-1]) == 0
    first, second = cache.kv_cache_groups[0].layer_names
    alias = cache.kv_cache_groups[1].layer_names[0]
    assert raw[first][0].data_ptr() == raw[alias][0].data_ptr()
    assert raw[first][0].data_ptr() != raw[second][0].data_ptr()


def test_uniform_group_wrappers_share_the_same_pool():
    cache = _cache_config()
    for group in cache.kv_cache_groups:
        group.kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=group.kv_cache_spec.block_size,
            kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names},
        )
    allocate = Mock(side_effect=lambda size: torch.zeros(size, dtype=torch.int8))
    raw = allocate_shared_attention_kv_cache(cache, allocate)
    allocate.assert_called_once_with(cache.kv_cache_tensors[0].size)
    assert len({tensor.untyped_storage().data_ptr() for pair in raw.values() for tensor in pair}) == 1


def test_group_descriptors_with_multiple_layers():
    cache = _cache_config()
    cache.kv_cache_tensors = [
        replace(descriptor, layers=group.layer_names)
        for descriptor, group in zip(cache.kv_cache_tensors[::2], cache.kv_cache_groups)
    ]
    raw = allocate_shared_attention_kv_cache(cache, lambda size: torch.zeros(size, dtype=torch.int8))
    first, second = cache.kv_cache_groups[0].layer_names
    assert raw[second][0].data_ptr() - raw[first][0].data_ptr() == cache.kv_cache_tensors[0].layer_stride
    assert raw[first][0].data_ptr() == raw[cache.kv_cache_groups[-1].layer_names[0]][0].data_ptr()


@pytest.mark.parametrize(
    "unsupported",
    [
        "empty",
        "full_only",
        "transfer",
        "dcp",
        "pcp",
        "layout",
        "padding",
        "quantized",
        "unequal_kv",
        "other_model",
        "custom_cache_dtype",
    ],
)
def test_unsupported_layouts_keep_private_allocation(unsupported):
    config = _config()
    specs = list(_specs(_cache_config()).values())
    if unsupported == "empty":
        specs = []
    elif unsupported == "full_only":
        specs = specs[:2]
    elif unsupported == "transfer":
        config.kv_transfer_config = object()
    elif unsupported in ("dcp", "pcp"):
        field = "decode_context_parallel_size" if unsupported == "dcp" else "prefill_context_parallel_size"
        setattr(config.parallel_config, field, 2)
    elif unsupported == "layout":
        config.cache_config.get_resolved_kv_cache_layout().is_layer_compact = False
    elif unsupported == "padding":
        specs = [replace(spec, page_size_padded=spec.page_size_bytes * 2) for spec in specs]
    elif unsupported == "quantized":
        specs = [replace(spec, kv_quant_mode=KVQuantMode.FP8_PER_TENSOR) for spec in specs]
    elif unsupported == "other_model":
        config.model_config.hf_text_config.model_type = "other_full_sliding_model"
    elif unsupported == "custom_cache_dtype":
        config.cache_config.cache_dtype = "mxfp8"
    else:
        specs[0] = replace(specs[0], head_size_v=4)
    assert not can_share_full_and_sliding_kv_cache(config, specs)


@pytest.mark.parametrize("invalid", ["size", "layer_stride", "block_stride", "zero"])
def test_invalid_descriptors_fail_before_allocating(invalid):
    cache = _cache_config()
    descriptor = cache.kv_cache_tensors[0]
    if invalid == "size":
        descriptor.size += 1
    elif invalid in ("layer_stride", "block_stride"):
        setattr(descriptor, invalid, 1)
    else:
        cache.num_blocks = 0
    allocate = Mock()
    with pytest.raises(ValueError, match="Shared attention|Invalid shared attention"):
        allocate_shared_attention_kv_cache(cache, allocate)
    allocate.assert_not_called()
