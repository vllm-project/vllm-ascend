# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.paired_sfa_cache import paired_sfa_cache_config, paired_sfa_cache_groups
from vllm_ascend.worker.v2 import attn_utils

NUM_BLOCKS = 4
BLOCK_SIZE = 128
ALIGNMENT = 2 * 1024 * 1024


@pytest.fixture
def vllm_config(monkeypatch):
    config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            use_mla=True,
            hf_config=SimpleNamespace(model_type="kimi_k31"),
            hf_text_config=SimpleNamespace(model_type="kimi_linear", index_topk=2048),
            max_model_len=4096,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=NUM_BLOCKS, prefix_cache_retention_interval=None, cache_dtype="int8"
        ),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    return config


def make_paired_cache(config, scale_dtype, with_draft=True, local_layers=None, dcp_size=1):
    config.parallel_config.decode_context_parallel_size = dcp_size
    specs = {}
    for layer in (3, 7):
        specs[f"model.layers.{layer}.attn"] = AscendMLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=656, dtype=torch.int8, cache_sparse_sfa_c8=True
        )
        specs[f"model.layers.{layer}.indexer.k_cache"] = AscendSFAIndexerCacheSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=scale_dtype,
            cache_sparse_li_c8=True,
            sfa_dcp_replicated_indexer_size=dcp_size,
        )
    for layer in range(3):
        specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((3, 32), (8, 8)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=BLOCK_SIZE * 656,
        )
    if with_draft:
        for layer, dtype in ((8, torch.bfloat16), (9, torch.int8)):
            specs[f"model.layers.{layer}.attn"] = AscendMLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=576,
                dtype=dtype,
                is_draft=True,
                non_causal_multi_token_decode=True,
            )
    groups = paired_sfa_cache_groups(config, specs)
    if local_layers is not None:
        groups = [
            replace(
                group, layer_names=[name for name in group.layer_names if extract_layer_index(name) in local_layers]
            )
            for group in groups
        ]
    local_specs = {name: specs[name] for group in groups for name in group.layer_names}
    return paired_sfa_cache_config(config, groups, 0), local_specs


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("with_draft", "local_layers", "num_main_slots"),
    [
        (False, None, 2),
        (True, None, 2),
        (True, (3, 7), 2),
        (True, (0, 1, 2), 2),
        (True, (8, 9), 0),
        (True, (0, 1, 2, 3), 2),
    ],
    ids=["target-only", "dspark", "pp-attention", "pp-recurrent", "pp-draft", "pp-mixed"],
)
@pytest.mark.parametrize("kv_transfer", [False, True])
def test_paired_cache_allocates_one_backing(
    monkeypatch, vllm_config, scale_dtype, with_draft, local_layers, num_main_slots, kv_transfer
):
    if kv_transfer:
        vllm_config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    plan, specs = make_paired_cache(vllm_config, scale_dtype, with_draft=with_draft, local_layers=local_layers)
    allocate = Mock(wraps=torch.zeros)
    monkeypatch.setattr(attn_utils.torch, "zeros", allocate)

    caches = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))

    main_layer_bytes = NUM_BLOCKS * BLOCK_SIZE * 656
    indexer_layer_bytes = NUM_BLOCKS * BLOCK_SIZE * (128 + get_dtype_size(scale_dtype))
    offsets = {
        "model.layers.3.attn": 0,
        "model.layers.7.attn": main_layer_bytes,
        "model.layers.0.linear_attn": 0,
        "model.layers.1.linear_attn": 0,
        "model.layers.2.linear_attn": main_layer_bytes,
    }
    offset = num_main_slots * main_layer_bytes
    for layer in (3, 7):
        name = f"model.layers.{layer}.indexer.k_cache"
        if name in specs:
            offsets[name] = offset
            offset += indexer_layer_bytes
    for layer, dtype_bytes in ((8, 2), (9, 1)):
        name = f"model.layers.{layer}.attn"
        if name in specs:
            offsets[name] = offset
            offset += NUM_BLOCKS * BLOCK_SIZE * 576 * dtype_bytes
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {offset}
    allocate.assert_called_once_with(
        offset + (ALIGNMENT if kv_transfer else 0), dtype=torch.int8, device=torch.device("cpu")
    )
    assert caches.keys() == specs.keys()
    first_name = next(iter(caches))
    base_ptr = caches[first_name].data_ptr() - offsets[first_name]
    if kv_transfer:
        assert base_ptr % ALIGNMENT == 0
    storage_ptrs = {cache.untyped_storage().data_ptr() for cache in caches.values()}
    assert len(storage_ptrs) == 1
    for name, cache in caches.items():
        assert cache.data_ptr() == base_ptr + offsets[name]
        assert cache.numel() == plan.num_blocks * specs[name].page_size_bytes

    first = caches[first_name]
    backing = first.as_strided((offset,), (1,), first.storage_offset() - offsets[first_name])
    for index, (name, cache) in enumerate(caches.items(), start=1):
        before = backing.clone()
        cache.fill_(index)
        start, end = offsets[name], offsets[name] + cache.numel()
        assert torch.all(backing[start:end] == index)
        torch.testing.assert_close(backing[:start], before[:start])
        torch.testing.assert_close(backing[end:], before[end:])


@pytest.mark.parametrize("draft_layer", [8, 9])
def test_draft_fia_reshape_uses_draft_cache_dtype_in_packed_target_group(monkeypatch, vllm_config, draft_layer):
    plan, specs = make_paired_cache(vllm_config, torch.float16)
    caches = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    name = f"model.layers.{draft_layer}.attn"
    spec = specs[name]
    backend = SimpleNamespace(
        get_kv_cache_shape=Mock(side_effect=lambda blocks, block, heads, size, dtype: (blocks, block, heads, size))
    )
    monkeypatch.setattr(attn_utils, "_uses_single_raw_mla_cache", lambda *_args: False)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_args: (512, 64))

    reshaped = attn_utils._reshape_kv_cache_v2(
        attn_groups=[AttentionGroup(backend, [name], spec, 0)],
        kv_cache_raw_tensors=caches,
        cache_dtype="int8",
        kernel_block_sizes=[BLOCK_SIZE],
        shared_kv_cache_layers={},
        kv_cache_config=plan,
    )

    backend.get_kv_cache_shape.assert_called_once_with(NUM_BLOCKS, BLOCK_SIZE, 1, 576, "auto")
    k_cache, v_cache = reshaped[name]
    assert k_cache.shape == (NUM_BLOCKS, BLOCK_SIZE, 1, 512)
    assert v_cache.shape == (NUM_BLOCKS, BLOCK_SIZE, 1, 64)
    assert k_cache.dtype == v_cache.dtype == spec.dtype


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("kernel_block_size", [64, 128])
@pytest.mark.parametrize("dcp_size", [1, 8])
def test_paired_indexer_and_mamba_writes_stay_in_their_pages(vllm_config, scale_dtype, kernel_block_size, dcp_size):
    plan, specs = make_paired_cache(vllm_config, scale_dtype, dcp_size=dcp_size)
    caches = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    indexer_name = "model.layers.7.indexer.k_cache"
    indexer_spec = specs[indexer_name]
    backend = SimpleNamespace(get_kv_cache_shape=lambda blocks, block, heads, size: (blocks, block, heads, size))
    reshaped = attn_utils._reshape_kv_cache_v2(
        attn_groups=[AttentionGroup(backend, [indexer_name], indexer_spec, 0)],
        kv_cache_raw_tensors=caches,
        cache_dtype="int8",
        kernel_block_sizes=[kernel_block_size],
        shared_kv_cache_layers={},
        kv_cache_config=plan,
    )
    k_cache, scale_cache = reshaped[indexer_name]
    backing_size = plan.kv_cache_tensors[0].size
    first = caches["model.layers.3.attn"]
    backing = first.as_strided((backing_size,), (1,), first.storage_offset())
    backing.fill_(42)
    before = backing.clone()
    kernel_parts = BLOCK_SIZE // kernel_block_size * dcp_size

    k_cache[kernel_parts : 2 * kernel_parts].fill_(1)
    scale_cache[kernel_parts : 2 * kernel_parts].fill_(2)

    expected_page = torch.cat(
        (
            torch.full((kernel_block_size * 128,), 1, dtype=torch.int8),
            torch.full((kernel_block_size,), 2, dtype=scale_dtype).view(torch.int8),
        )
    ).repeat(kernel_parts)
    raw_indexer = caches[indexer_name]
    page_bytes = indexer_spec.page_size_bytes
    torch.testing.assert_close(raw_indexer[page_bytes : 2 * page_bytes], expected_page)
    start = raw_indexer.data_ptr() - backing.data_ptr() + page_bytes
    end = start + page_bytes
    torch.testing.assert_close(backing[:start], before[:start])
    torch.testing.assert_close(backing[end:], before[end:])

    mamba_name = "model.layers.2.linear_attn"
    conv, ssm = attn_utils._reshape_mamba_kv_cache(caches[mamba_name], specs[mamba_name])
    before = backing.clone()
    conv[2].fill_(0.5)
    ssm[2].fill_(0.25)
    main_page_bytes = BLOCK_SIZE * 656
    start = caches[mamba_name].data_ptr() - backing.data_ptr() + 2 * main_page_bytes
    end = start + conv[2].numel() * conv.element_size() + ssm[2].numel() * ssm.element_size()
    torch.testing.assert_close(backing[:start], before[:start])
    torch.testing.assert_close(backing[end:], before[end:])


@pytest.mark.parametrize("model_type", ["deepseek_v32", "glm_moe_dsa"])
@pytest.mark.parametrize("num_heads", [2, 64])
def test_generic_sfa_mixed_group_keeps_private_caches(monkeypatch, vllm_config, model_type, num_heads):
    """K3.1 paired dispatch must not consume generic SFA descriptor strides."""
    vllm_config.model_config.hf_config.model_type = model_type
    vllm_config.model_config.hf_text_config.model_type = model_type
    main_spec = AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_heads=num_heads,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = AscendSFAIndexerCacheSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    # DeepSeek constructs each indexer before its MLA attention layer.
    indexer_names = [f"model.layers.{layer}.indexer.k_cache" for layer in (0, 1)]
    main_names = [f"model.layers.{layer}.attn" for layer in (0, 1)]
    specs: dict[str, AscendMLAAttentionSpec | AscendSFAIndexerCacheSpec] = {}
    for indexer_name, main_name in zip(indexer_names, main_names):
        specs[indexer_name] = indexer_spec
        specs[main_name] = main_spec
    backing_size = NUM_BLOCKS * sum(spec.page_size_bytes for spec in specs.values())
    descriptors = []
    offset = 0
    for spec, names in ((indexer_spec, indexer_names), (main_spec, main_names)):
        # The generic planner derives descriptors from the logical head shape.
        # Ascend's query-head field must not change its one-latent-head storage.
        layer_stride, block_stride, *_ = compute_layout_strides(spec, NUM_BLOCKS, len(names), KVCacheLayout.LBNHC)
        descriptors.append(
            KVCacheTensor(
                size=backing_size,
                layers=names,
                offset=offset,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        )
        offset += len(names) * layer_stride
    assert descriptors[1].offset + descriptors[1].layer_stride >= backing_size
    plan = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=descriptors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=list(specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=BLOCK_SIZE, kv_cache_specs=specs),
            )
        ],
    )
    monkeypatch.setattr(attn_utils, "_uses_single_raw_mla_cache", lambda *_args: False)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_args: (512, 64))

    raw_caches = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    for name, spec in specs.items():
        assert isinstance(raw_caches[name], tuple)
        assert sum(part.numel() for part in raw_caches[name]) == NUM_BLOCKS * spec.page_size_bytes
    backend = SimpleNamespace(get_kv_cache_shape=lambda blocks, block, heads, size, *args: (blocks, block, heads, size))
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            AttentionGroup(backend, indexer_names, indexer_spec, 0),
            AttentionGroup(backend, main_names, main_spec, 0),
        ],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[BLOCK_SIZE],
        shared_kv_cache_layers={},
        kv_cache_config=plan,
    )
    for name in indexer_names:
        assert caches[name][0].shape == (NUM_BLOCKS, BLOCK_SIZE, 1, 128)
    for name in main_names:
        assert caches[name][0].shape == (NUM_BLOCKS, BLOCK_SIZE, 1, 512)
        assert caches[name][1].shape == (NUM_BLOCKS, BLOCK_SIZE, 1, 64)
    for name, parts in caches.items():
        for part in parts:
            part.fill_(7)
        for other_name, other_parts in caches.items():
            if other_name != name:
                assert all(torch.count_nonzero(part) == 0 for part in other_parts)
        for part in parts:
            part.zero_()
