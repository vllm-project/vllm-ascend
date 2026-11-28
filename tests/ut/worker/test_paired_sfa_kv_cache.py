# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_interface import (
    AscendGQADraftFullAttentionSpec,
    AscendGQADraftSlidingWindowSpec,
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.models.kimi_k3.paired_sfa_cache import (
    paired_sfa_cache_config,
    paired_sfa_cache_groups,
    supports_k31_draft_separate_pool,
)
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.utils import AscendV2KVBlockZeroer

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


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    ("spec", "draft_spec_type"),
    [
        (
            SlidingWindowSpec(
                block_size=BLOCK_SIZE, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, sliding_window=2048
            ),
            AscendGQADraftSlidingWindowSpec,
        ),
        (
            FullAttentionSpec(block_size=1664, num_kv_heads=2, head_size=64, dtype=torch.bfloat16),
            AscendGQADraftFullAttentionSpec,
        ),
    ],
    ids=["packed-swa", "split-full"],
)
def test_gqa_dspark_uses_private_contiguous_kv_planes(vllm_config, monkeypatch, layout, spec, draft_spec_type):
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2", get_from_extra_config={}.get)
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.cache_config.cache_dtype = "auto"
    vllm_config.cache_config.block_size = 768
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    _, specs = make_paired_cache(vllm_config, torch.float16, with_draft=False)
    target_specs = dict(specs)
    # Native GQA specs arrive untagged in the v0.30 runner.
    draft_name = "draft.model.layers.93.self_attn.attn"
    specs[draft_name] = spec
    marked = attn_utils.mark_kimi_k31_gqa_draft_specs(vllm_config, specs, [draft_name])
    assert specs == {**target_specs, draft_name: spec}
    assert isinstance(marked[draft_name], draft_spec_type)
    assert marked[draft_name].sliding_window == spec.sliding_window

    groups = paired_sfa_cache_groups(vllm_config, marked)
    assert groups[-1].layer_names == [draft_name]
    plan = paired_sfa_cache_config(vllm_config, groups, 0)
    raw_caches = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    reshaped = attn_utils._reshape_kv_cache_v2(
        attn_groups=[AttentionGroup(AscendAttentionBackend, [draft_name], marked[draft_name], len(groups) - 1)],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="int8",
        kernel_block_sizes=[BLOCK_SIZE] * len(groups),
        shared_kv_cache_layers={},
        kv_cache_config=plan,
    )

    key, value = reshaped[draft_name]
    assert key.is_contiguous() and value.is_contiguous()
    manager_block_size = 768 if draft_spec_type is AscendGQADraftSlidingWindowSpec else spec.block_size
    assert marked[draft_name].block_size == manager_block_size
    num_kernel_blocks = NUM_BLOCKS * (manager_block_size // BLOCK_SIZE)
    shape = (num_kernel_blocks, 2, BLOCK_SIZE, 64) if layout == "HND" else (num_kernel_blocks, BLOCK_SIZE, 2, 64)
    assert key.shape == value.shape == shape
    assert key.dtype == value.dtype == torch.bfloat16
    assert key.data_ptr() == raw_caches[draft_name].data_ptr()
    assert value.data_ptr() - key.data_ptr() == key.nbytes
    key[1].fill_(1)
    value[2].fill_(2)
    assert torch.count_nonzero(key) == key[1].numel()
    assert torch.count_nonzero(value) == value[2].numel()
    assert all(torch.count_nonzero(raw_caches[name]) == 0 for name in target_specs)


def test_kimi_k31_gqa_draft_spec_rejects_context_parallelism(vllm_config):
    vllm_config.parallel_config.prefill_context_parallel_size = 2
    vllm_config.parallel_config.decode_context_parallel_size = 8
    spec = SlidingWindowSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, sliding_window=2048
    )
    with pytest.raises(ValueError, match="requires PCP=1"):
        attn_utils.mark_kimi_k31_gqa_draft_specs(vllm_config, {"draft": spec}, ["draft"])


@pytest.mark.parametrize(
    ("connector", "cache_dtype", "override", "expected"),
    [
        (None, "auto", None, 128),
        ("OtherConnector", "auto", None, 128),
        ("MooncakeConnectorV2", "int8", None, 128),
        ("MooncakePullConnector", "auto", 384, 384),
        ("MooncakeConnectorV2", "auto", 0, None),
        ("MooncakeConnectorV2", "auto", 129, None),
        ("MooncakeConnectorV2", "auto", "768", None),
    ],
)
def test_gqa_dspark_swa_page_selection(vllm_config, connector, cache_dtype, override, expected):
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    options = {"dspark_swa_block_size": override}
    vllm_config.kv_transfer_config = (
        SimpleNamespace(kv_connector=connector, get_from_extra_config=options.get) if connector else None
    )
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.cache_config.cache_dtype = cache_dtype
    vllm_config.cache_config.block_size = 768
    spec = SlidingWindowSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, sliding_window=2048)
    if expected is None:
        with pytest.raises(ValueError, match="positive multiple of 128"):
            attn_utils.mark_kimi_k31_gqa_draft_specs(vllm_config, {"draft": spec}, ["draft"])
    else:
        marked = attn_utils.mark_kimi_k31_gqa_draft_specs(vllm_config, {"draft": spec}, ["draft"])
        assert marked["draft"].block_size == expected


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


def make_bf16_swa_cache(config, monkeypatch, page, *, private=False, split_budget=None):
    config.model_config.dtype = torch.bfloat16
    config.cache_config.cache_dtype = "auto"
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2")
    if split_budget is not None:
        config.use_v2_model_runner = True
        config.max_in_flight_tokens = 128
        config.cache_config.num_gpu_blocks_override = None
        config.kv_transfer_config.kv_connector_extra_config = {"k31_draft_separate_pool": True}
    config.parallel_config.cp_kv_cache_interleave_size = BLOCK_SIZE
    main = [f"model.layers.{4 * i + 3}.attn" for i in range(10)]
    indexers = [name.removesuffix("attn") + "indexer.k_cache" for name in main]
    drafts = [f"draft.model.layers.{93 + i}.attn" for i in range(5)]
    specs = {}
    for names, spec in (
        (main, AscendMLAAttentionSpec(block_size=page, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)),
        (indexers, AscendSFAIndexerCacheSpec(block_size=page, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)),
        (
            drafts,
            AscendGQADraftSlidingWindowSpec(
                block_size=page, num_kv_heads=2, head_size=64, dtype=torch.bfloat16, sliding_window=2048
            ),
        ),
    ):
        specs.update(dict.fromkeys(names, spec))
    groups = paired_sfa_cache_groups(config, specs)
    if private:
        groups[1] = replace(groups[1], kv_cache_spec=replace(groups[1].kv_cache_spec, can_overlay_indexer=False))
    plan = paired_sfa_cache_config(config, groups, split_budget or 0)
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    attn_groups = [
        AttentionGroup(AscendSFABackend, main, specs[main[0]], 0),
        AttentionGroup(AscendSFAIndexerBackend, indexers, specs[indexers[0]], 0),
        AttentionGroup(AscendAttentionBackend, drafts, groups[1].kv_cache_spec, 1),
    ]
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_: (512, 64))
    caches = attn_utils._reshape_kv_cache_v2(attn_groups, raw, "auto", [BLOCK_SIZE] * len(groups), {}, plan)
    return plan, attn_groups, caches, indexers, drafts


def test_dcp_sfa_reads_original_kernel_pages_from_mrv2(monkeypatch, vllm_config):
    from vllm_ascend.device.device_op import _sfa_paired_pages

    plan, groups, caches, _, _ = make_bf16_swa_cache(vllm_config, monkeypatch, page=896)
    for name in groups[0].layer_names:
        k, rope = caches[name]
        pages = _sfa_paired_pages(k, rope)
        assert pages is not None and pages.is_contiguous()
        assert pages.shape == (plan.num_blocks * 7, BLOCK_SIZE, 1, 576)
        assert pages.data_ptr() == k.data_ptr()
        assert rope.data_ptr() == pages.data_ptr() + BLOCK_SIZE * 512 * k.element_size()


@pytest.mark.skipif(not supports_k31_draft_separate_pool(), reason="Installed vLLM has a single KV block pool")
def test_worker_zero_and_cow_use_independent_pool_ids(monkeypatch, vllm_config):
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    plan, groups, caches, _, _ = make_bf16_swa_cache(vllm_config, monkeypatch, 768, split_budget=92 * 2**20)
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_cache_config = plan
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.attn_groups = [[g for g in groups if g.kv_cache_group_id == i] for i in range(2)]
    runner.kernel_block_sizes = [BLOCK_SIZE] * 2
    runner.compilation_config = SimpleNamespace(
        static_forward_context={name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
    )
    runner._pool_cache_views = {
        pid: (plan.num_blocks_per_group[pid], [caches[name] for name in group.layer_names])
        for pid, group in enumerate(plan.kv_cache_groups)
    }
    runner._init_kv_zero_meta()
    first = next(iter(caches.values()))[0]
    backing = first.view(torch.uint8).as_strided((plan.kv_cache_tensors[0].size,), (1,), first.storage_offset() * 2)

    def cpu_zero(addrs, strides, pages, ids, BLOCK_SIZE):
        for address, stride, size in zip(addrs.tolist(), strides.tolist(), pages.tolist(), strict=True):
            for block in ids.tolist():
                start = address - backing.data_ptr() + 4 * block * stride
                assert 0 <= start < start + size * 4 <= backing.numel()
                backing[start : start + size * 4].zero_()

    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=cpu_zero)
    monkeypatch.setattr("vllm.v1.worker.utils._zero_kv_blocks_kernel", kernel)

    def update_tables_and_zero_primary(self, output):
        assert not output.kv_cache_block_copies
        self.kv_block_zeroer.zero_block_ids(output.new_block_ids_to_zero or [])

    monkeypatch.setattr(GPUModelRunner, "update_requests", update_tables_and_zero_primary)
    # Same numerical IDs have unrelated contents and operations in the two pools.
    expected = {}
    for pid, (capacity, views) in runner._pool_cache_views.items():
        for layer, entry in enumerate(views):
            for i, tensor in enumerate(entry):
                blocks = tensor.unflatten(0, (capacity, -1))
                for block_id in range(capacity):
                    blocks[block_id].fill_(1 + 20 * pid + 2 * layer + i + block_id)
                before = blocks.clone()
                if pid == 0:
                    before[1].zero_()
                    before[3].copy_(before[2])
                else:
                    before[2].zero_()
                    before[3].copy_(before[1])
                expected[pid, layer, i] = before
    # A partially upgraded scheduler can send only legacy pool-0 operations.
    # Reject before the base runner changes block tables or clears live pages.
    with pytest.raises(ValueError, match="pool-aware SchedulerOutput"):
        runner.update_requests(SimpleNamespace(new_block_ids_to_zero=[1], kv_cache_block_copies=None))

    output = SchedulerOutput.make_empty()
    output.new_block_ids_to_zero = [1]
    output.new_block_ids_to_zero_by_pool = {1: [2]}
    output.kv_cache_block_copies = [KVCacheBlockCopy(2, 3)]
    output.kv_cache_block_copies_by_pool = {1: [KVCacheBlockCopy(1, 3)]}
    runner.update_requests(output)
    for pid, (capacity, views) in runner._pool_cache_views.items():
        for layer, entry in enumerate(views):
            for i, tensor in enumerate(entry):
                assert torch.equal(tensor.unflatten(0, (capacity, -1)), expected[pid, layer, i])


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("page", [768, 896])
def test_bf16_swa_indexer_alias_recycling_and_cow(monkeypatch, vllm_config, layout, page):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    plan, attn_groups, caches, indexers, drafts = make_bf16_swa_cache(vllm_config, monkeypatch, page)
    pool = BlockPool(plan.num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE)
    target, draft, spare = pool.get_new_blocks(3)
    assert (target.block_id, draft.block_id, spare.block_id) == (1, 2, 3)
    planes = [plane for name in drafts for plane in caches[name]]
    assert len({p.data_ptr() for p in planes}) == 10
    parts = page // BLOCK_SIZE
    shape = (2, BLOCK_SIZE, 64) if layout == "HND" else (BLOCK_SIZE, 2, 64)
    expected = []
    for i, (plane, index_name) in enumerate(zip(planes, indexers, strict=True)):
        (indexer,) = caches[index_name]
        assert plane.shape == (plan.num_blocks * parts, *shape)
        assert plane.is_contiguous() and indexer.is_contiguous()
        assert plane.data_ptr() == indexer.data_ptr() and plane.nbytes == indexer.nbytes
        # Interleave target-owned ID 1, draft-owned ID 2 and untouched guard IDs.
        index_bytes = indexer.view(torch.uint8).view(plan.num_blocks, -1)
        offset = torch.arange(index_bytes.shape[1], dtype=torch.int32)
        pattern = ((offset ^ (offset >> 8) ^ (offset >> 16)) + i).to(torch.uint8)
        index_bytes.fill_(43)
        index_bytes[1].copy_(pattern)
        plane[2 * parts : 3 * parts].view(torch.uint8).flatten().copy_(255 - pattern)
        reference = torch.full_like(index_bytes, 43)
        reference[1].copy_(pattern)
        reference[2].copy_(255 - pattern)
        assert torch.equal(index_bytes, reference)
        expected.append(reference)

    # Copy overlapping sources and destinations once, even though both
    # interpretations of each plane were supplied to the worker.
    copy_kv_cache_blocks_inplace(caches.values(), plan.num_blocks, [KVCacheBlockCopy(1, 2), KVCacheBlockCopy(2, 3)])
    for plane, reference in zip(planes, expected, strict=True):
        reference[[2, 3]] = reference[[1, 2]]
        assert torch.equal(plane.view(torch.uint8).view_as(reference), reference)

    # Replay the real zeroer's generated segment table using CPU tensors.
    first = caches[attn_groups[0].layer_names[0]][0]
    backing = first.view(torch.uint8).as_strided((plan.kv_cache_tensors[0].size,), (1,), first.storage_offset() * 2)

    def cpu_zero(addrs, strides, pages, block_ids, BLOCK_SIZE):
        for address, stride, size in zip(addrs.tolist(), strides.tolist(), pages.tolist(), strict=True):
            for block_id in block_ids.tolist():
                start = address - backing.data_ptr() + 4 * stride * block_id
                assert 0 <= start < start + 4 * size <= backing.numel()
                backing[start : start + 4 * size].zero_()

    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=cpu_zero)
    monkeypatch.setattr("vllm.v1.worker.utils._zero_kv_blocks_kernel", kernel)
    zeroer = AscendV2KVBlockZeroer(
        torch.device("cpu"),
        attn_groups,
        [BLOCK_SIZE] * len(plan.kv_cache_groups),
        {name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()},
        plan.num_blocks,
    )
    pool.free_blocks([draft])
    (new_draft,) = pool.get_new_blocks(1)
    assert new_draft.block_id == 2
    zeroer.zero_block_ids([new_draft.block_id])
    for plane, index_name, reference in zip(planes, indexers, expected, strict=True):
        reference[2].zero_()
        observed = plane.view(torch.uint8).view_as(reference)
        assert torch.equal(observed, reference)
        # The released target page becomes draft-owned. Its neighbors are live.
        plane[2 * parts : 3 * parts].view(torch.uint8).fill_(197)
        reference[2].fill_(197)
        assert torch.equal(caches[index_name][0].view(torch.uint8).view_as(reference), reference)


@pytest.mark.parametrize("layout", ["overlaid", "private-producer", "split"])
def test_mooncake_pulls_target_and_draft_into_disjoint_ids(monkeypatch, vllm_config, layout):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import ReqMeta
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakeSchedulerSendingThread
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    monkeypatch.setattr(torch.npu, "set_device", lambda *_: None)
    vllm_config.model_config.is_deepseek_mla = True
    vllm_config.speculative_config = None
    engine = Mock()
    monkeypatch.setattr(base_worker, "global_te", engine)
    page = 768

    def register(engine_id, private=False):
        budget = (75 if engine_id == "P" else 92) * 2**20 if layout == "split" else None
        plan, _, caches, indexers, drafts = make_bf16_swa_cache(
            vllm_config, monkeypatch, page, private=private, split_budget=budget
        )
        worker = base_worker.MooncakeBaseConnectorWorker.__new__(base_worker.MooncakeBaseConnectorWorker)
        worker.vllm_config = vllm_config
        worker.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
        worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
        worker.kv_cache_config = plan
        worker.engine_id, worker.te_rpc_port = engine_id, 9000
        worker.block_size, worker.side_channel_host, worker.handshake_port = page, "127.0.0.1", 6000
        worker.register_kv_caches(caches)
        ([base], [length]) = engine.register_buffer.call_args.args
        assert length == plan.kv_cache_tensors[0].size
        return worker, caches, indexers, drafts, (base, length)

    p, producer, indexers, drafts, p_range = register("P", layout == "private-producer")
    d, consumer, _, _, d_range = register("D")
    sender = MooncakeSchedulerSendingThread(
        "127.0.0.1", 6000, "P", {0: p.transfer_metadata}, 1, 1, 1, 1, False, Event()
    )
    receiver = MooncakePullRecvingThread(
        engine,
        vllm_config,
        d.kv_cache_config,
        d.kv_cache_specs,
        d.layer_name_to_group_index,
        d.layer_name_to_spec_index,
        d.transfer_metadata,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
        dp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_rank=0,
        dcp_size=1,
        device=torch.device("cpu"),
        ready_event=Event(),
    )
    remote = receiver.decoder.decode(sender.encoded_metadata)
    receiver._validate_remote_metadata(remote, "P")
    split = bool(d.transfer_metadata.block_pool_ids)
    assert split == (layout == "split" and supports_k31_draft_separate_pool())
    if split:
        # Pools may have different capacities on each host. Mixing local-ID
        # and global-ID layouts must fail before a transfer is submitted.
        pp = remote.metadata_by_pp_rank[0]
        mixed = replace(remote, metadata_by_pp_rank={0: replace(pp, block_pool_ids=(), num_blocks_per_group=())})
        with pytest.raises(ValueError, match="block pool layout mismatch"):
            receiver._build_remote_transfer_layout(mixed)
    receiver.remote_metadata["P"] = remote
    receiver.remote_tp_rank_groups["P"], receiver.remote_layer_index_pairs["P"] = (
        receiver._build_remote_transfer_layout(remote)
    )

    def cpu_read(session_id, local, peer, lengths):
        for dst, src, size in zip(local, peer, lengths, strict=True):
            assert d_range[0] <= dst < dst + size <= sum(d_range)
            assert p_range[0] <= src < src + size <= sum(p_range)
            ctypes.memmove(dst, src, size)
        return 0

    engine.batch_transfer_sync_read.side_effect = cpu_read
    # P target/draft IDs 1/2, D target/draft IDs 3/1. Transfer two kernel
    # blocks of a six-block manager page, including one partial final block.
    if split:
        assert p.transfer_metadata.block_pool_ids == d.transfer_metadata.block_pool_ids == (0, 1)
        assert p.transfer_metadata.num_blocks_per_group != d.transfer_metadata.num_blocks_per_group
        assert len(set(d.transfer_metadata.num_blocks_per_group)) == 2
    dst_draft = 3 if split else 1
    src_draft = 1 if split else 2
    local_ids, remote_ids = ([3], [dst_draft]), ([1], [src_draft])
    req = ReqMeta(local_ids, 194, 193, 0, remote_ids, "127.0.0.1", 6000, "P", "req-p", 193, local_ids)
    parts = page // BLOCK_SIZE
    for i, name in enumerate(producer):
        src_id = src_draft if name in drafts else 1
        for component, tensor in enumerate(producer[name]):
            block = tensor[src_id * parts : (src_id + 1) * parts]
            offsets = torch.arange(block.numel(), dtype=torch.int32).reshape(block.shape)
            block.copy_(
                ((offsets ^ (offsets >> 8) ^ (offsets >> 16)) + 13 * i + component).to(torch.int16).view(torch.bfloat16)
            )
    try:
        assert receiver._handle_requests("P", "127.0.0.1", 6000, {"req": req}) == set()
    finally:
        receiver.executor.shutdown(wait=True)
    engine.batch_transfer_sync_read.assert_called_once()

    # Check the entire D plane, including both live owners, unused ID 2,
    # guard ID 0 and the untransferred suffix of each allocated manager page.
    for i, index_name in enumerate(indexers):
        draft_name = drafts[i // 2]
        result = consumer[index_name][0].view(torch.uint8)
        expected = torch.zeros_like(result)
        expected[3 * parts : 3 * parts + 2].copy_(producer[index_name][0][parts : parts + 2].view(torch.uint8))
        draft_part = producer[draft_name][i % 2][src_draft * parts : src_draft * parts + 2].view(torch.uint8)
        if split:
            observed_draft = consumer[draft_name][i % 2].view(torch.uint8)
            expected_draft = torch.zeros_like(observed_draft)
            expected_draft[dst_draft * parts : dst_draft * parts + 2].copy_(draft_part)
            assert torch.equal(observed_draft, expected_draft)
        else:
            expected[parts : parts + 2].copy_(draft_part.reshape_as(expected[parts : parts + 2]))
        assert torch.equal(result, expected)
