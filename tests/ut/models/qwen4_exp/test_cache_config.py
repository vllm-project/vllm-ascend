# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.models.qwen4_exp.cache_config import (
    QWEN_CACHE_GROUP_SIZE,
    get_qwen4_exp_kv_cache_groups,
    is_qwen4_exp_cache,
)
from vllm_ascend.models.qwen4_exp.gbsa import _as_gbsa_page_pool
from vllm_ascend.models.qwen4_exp.qsa import _split_qsa_kv_cache
from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_project_kv_cache_groups_to_worker
from vllm_ascend.worker.v1.model_runner import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

_LAYER_TYPES = tuple(["linear_attention"] * 3 + ["qwen_sparse_attention"]) * 12


def _config(num_blocks=None, cycles=12):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=list(_LAYER_TYPES[: cycles * 4]), model_type="qwen4_exp"),
            max_model_len=8192,
            original_max_model_len=8192,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_blocks,
            prefix_cache_retention_interval=0,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
            enable_prefix_caching=True,
            block_size=128,
            mamba_cache_mode="align",
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
        kv_transfer_config=None,
        num_prefill_lookahead_tokens=0,
    )


def _specs(cycles: int = 12) -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=130,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        model_version="qwen4_exp",
    )
    gdn = MambaSpec(
        block_size=128,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        shapes=((1, 1280), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    ple = MambaSpec(
        block_size=128,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        shapes=((10240,),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        tp_replicated=True,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index, layer_type in enumerate(_LAYER_TYPES[: cycles * 4]):
        if layer_type == "linear_attention":
            specs[f"model.layers.{index}.linear_attn"] = gdn
        else:
            source = f"model.layers.{index}.self_attn"
            specs[f"{source}.attn"] = main
            specs[f"{source}.indexer.compressed_key_cache"] = compressed
            specs[f"{source}.indexer.raw_key_cache"] = raw
    specs["model.ple"] = ple
    return specs


def _make_config(specs=None, num_blocks=8, model_config=None):
    model_config = model_config or _config()
    specs = _specs() if specs is None else specs
    groups = get_qwen4_exp_kv_cache_groups(model_config, specs)
    pool_bytes = max(len(g.layer_names) * g.kv_cache_spec.page_size_bytes for g in groups)
    return kv_cache_utils.get_kv_cache_config_from_groups(model_config, groups, num_blocks * pool_bytes)


def _allocate(config, *, aligned=False, device="cpu"):
    """Exercise the real standard runner allocation and reshape paths."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device(device)
    runner.vllm_config = _config()
    runner.model_config = runner.vllm_config.model_config
    runner.compilation_config = runner.vllm_config.compilation_config
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1), xlite_graph_config=SimpleNamespace(enabled=False)
    )
    runner.dcp_size = 1
    runner.use_sparse = True  # QSA must not fall back to per-layer allocation.
    runner.use_compress = False
    runner.sparse_kv_offload_enabled = False
    runner.runner_only_attn_layers = set()
    runner.kv_cache_config = config
    runner._is_c8_mxfp_kv_cache = lambda spec: False
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                layer_names=g.layer_names, kv_cache_spec=g.kv_cache_spec, kv_cache_group_id=i, backend=SimpleNamespace()
            )
            for i, g in enumerate(config.kv_cache_groups)
        ]
    )
    backings = []

    def allocator(size, alignment):
        # A nonzero storage offset is deliberate, even if torch's allocator
        # happens to return an already aligned address.
        storage = torch.zeros(size + (alignment + 16 if aligned else 0), dtype=torch.int8, device=device)
        offset = 16 if aligned else 0
        backing = storage[offset : offset + size]
        backings.append(backing)
        return backing

    runner._allocate_int8_cache_tensor = allocator
    if aligned:
        runner.vllm_config.kv_transfer_config = SimpleNamespace()
    raw = runner._allocate_kv_cache_tensors(config)
    return backings, raw, runner._reshape_kv_cache_tensors(config, raw)


def _views(cache):
    return cache if isinstance(cache, (tuple, list)) else (cache,)


def test_seven_groups_preserve_semantics_without_fake_layers():
    specs = _specs()
    original = copy.deepcopy(specs)
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    assert [len(g.layer_names) for g in groups] == [12, 12, 12, 12, 12, 1, 12]
    assert QWEN_CACHE_GROUP_SIZE == 12
    assert sum(len(g.layer_names) for g in groups) == len(specs) == 73
    assert set(n for g in groups for n in g.layer_names) == set(specs)
    assert len({g.kv_cache_spec.page_size_bytes for g in groups}) == 1
    assert all(type(g) is KVCacheGroupSpec for g in groups)
    for name, spec in specs.items():
        assert replace(spec, page_size_padded=original[name].page_size_padded) == original[name]
    assert groups[5].layer_names == ["model.ple"]
    assert groups[5].kv_cache_spec.tp_replicated
    assert not groups[2].kv_cache_spec.tp_replicated
    assert not groups[6].enable_kv_transfer and not groups[6].kv_cache_spec.prefix_cacheable
    gdn = sorted(
        (n for n in specs if n.endswith("linear_attn")), key=lambda n: int(n.split(".layers.")[1].split(".")[0])
    )
    assert [g.layer_names for g in groups[2:5]] == [gdn[i::3] for i in range(3)]


@pytest.mark.parametrize("blocks", [3, 9])
def test_standard_descriptor_geometry_and_single_backing(blocks):
    config = _make_config(num_blocks=blocks)
    page = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    assert len(config.kv_cache_tensors) == 7
    for d in config.kv_cache_tensors:
        assert (d.offset, d.block_stride, d.layer_stride, d.size) == (0, page, blocks * page, 12 * blocks * page)
    backings, raw, caches = _allocate(config, aligned=True)
    assert len(backings) == 1 and backings[0].numel() == 12 * blocks * page
    assert all(value is backings[0] for value in raw.values())
    assert backings[0].storage_offset() > 0
    for d in config.kv_cache_tensors:
        for i, name in enumerate(d.layers):
            first = _views(caches[name])[0]
            assert first.data_ptr() == backings[0].data_ptr() + d.offset + i * d.layer_stride
            for view in _views(caches[name]):
                assert view.stride(0) * view.element_size() == page
    key, value = _split_qsa_kv_cache(caches["model.layers.3.self_attn.attn"], 256)
    assert key.shape == (blocks, 128, 1, 256)
    assert key.stride(1) == 256 and value.data_ptr() - key.data_ptr() == 65536
    _as_gbsa_page_pool(key)
    conv, ssm = caches["model.layers.0.linear_attn"]
    assert ssm.data_ptr() - conv.data_ptr() == conv[0].numel() * conv.element_size()


def test_all_groups_with_distinct_blocks_are_isolated():
    config = _make_config(num_blocks=8)
    backings, _, caches = _allocate(config)
    for group_id, g in enumerate(config.kv_cache_groups):
        for layer_id, name in enumerate(g.layer_names):
            for view in _views(caches[name]):
                view[group_id].fill_(1 + layer_id + group_id * 12)
    for group_id, g in enumerate(config.kv_cache_groups):
        for layer_id, name in enumerate(g.layer_names):
            for view in _views(caches[name]):
                assert torch.all(view[group_id] == 1 + layer_id + group_id * 12)
    assert len(backings) == 1


@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
def test_real_tp8_pages_are_derived_from_specs(ssm_dtype):
    specs = _specs()
    for n, s in list(specs.items()):
        if isinstance(s, MambaSpec):
            specs[n] = replace(
                s,
                block_size=768,
                shapes=((3, 1280), (6, 128, 128)) if len(s.shapes) == 2 else ((9, 10240),),
                dtypes=(torch.bfloat16, ssm_dtype) if len(s.shapes) == 2 else s.dtypes,
            )
        elif not isinstance(s, CircularBufferSpec):
            specs[n] = replace(s, block_size=768)
    config = _make_config(specs, num_blocks=2)
    assert all(g.kv_cache_spec.page_size_bytes == 768 * 1024 for g in config.kv_cache_groups)
    assert config.kv_cache_tensors[0].size == 12 * 2 * 768 * 1024


def test_draft_buckets_remain_separate():
    specs = _specs()
    for suffix in (".attn", ".indexer.compressed_key_cache", ".indexer.raw_key_cache"):
        specs["model.mtp.layers.48.self_attn" + suffix] = specs["model.layers.3.self_attn" + suffix]
    config = _make_config(specs)
    assert len(config.kv_cache_groups) == 10
    assert [len(g.layer_names) for g in config.kv_cache_groups[-3:]] == [1, 1, 1]
    assert all(g.is_eagle_group for g in config.kv_cache_groups[-3:])
    assert not config.kv_cache_groups[-1].enable_kv_transfer


def test_pp_projects_local_layers_and_keeps_metadata():
    specs = _specs()
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    local = {n: s for n, s in specs.items() if ".layers.47." in n}
    projected = _ascend_project_kv_cache_groups_to_worker(groups, local)
    assert len(projected) == 7
    assert sum(len(g.layer_names) for g in projected) == 3
    assert not projected[-1].enable_kv_transfer
    config = kv_cache_utils.get_kv_cache_config_from_groups(
        _config(), projected, 3 * groups[0].kv_cache_spec.page_size_bytes
    )
    assert len(config.kv_cache_tensors) == 3
    assert all(d.size == 3 * groups[0].kv_cache_spec.page_size_bytes for d in config.kv_cache_tensors)
    assert len(_allocate(config)[0]) == 1


@pytest.mark.parametrize("override", [None, 256])
def test_standard_multiworker_replans_with_local_group_width(override):
    specs = _specs()
    local = {n: s for n, s in specs.items() if ".layers.47." in n}
    config = _config(num_blocks=override)
    plans = kv_cache_utils.get_kv_cache_configs(config, [specs, local], [10**9, 10**9])
    assert plans[0].num_blocks == plans[1].num_blocks
    assert plans[0].num_blocks == (
        override or 10**9 // (12 * plans[0].kv_cache_groups[0].kv_cache_spec.page_size_bytes)
    )
    assert plans[0].kv_cache_tensors[0].size == 12 * plans[1].kv_cache_tensors[0].size
    assert not plans[1].kv_cache_groups[-1].enable_kv_transfer


def test_identification_and_order_independence():
    specs = _specs()
    assert is_qwen4_exp_cache(specs)
    a = get_qwen4_exp_kv_cache_groups(_config(), specs)
    b = get_qwen4_exp_kv_cache_groups(_config(), dict(reversed(list(_specs().items()))))
    assert a == b and is_qwen4_exp_cache(a)


@pytest.mark.parametrize("attribute,value", [("offset", -1), ("block_stride", 16), ("layer_stride", 0), ("size", 1)])
def test_invalid_standard_descriptors_are_rejected(attribute, value):
    config = _make_config(num_blocks=3)
    config.kv_cache_tensors[0] = replace(config.kv_cache_tensors[0], **{attribute: value})
    with pytest.raises(ValueError, match="LBNHC|backing"):
        _allocate(config)


def test_pp_state_only_rank_allocates_one_local_slot():
    specs = _specs()
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    local = {"model.ple": specs["model.ple"]}
    projected = _ascend_project_kv_cache_groups_to_worker(groups, local)
    config = kv_cache_utils.get_kv_cache_config_from_groups(
        _config(), projected, 3 * specs["model.ple"].page_size_bytes
    )
    assert len(config.kv_cache_tensors) == 1
    backings, _, caches = _allocate(config, aligned=True)
    assert len(backings) == 1 and set(caches) == {"model.ple"}


def test_management_semantics_split_buckets_without_changing_specs():
    specs = _specs()
    name = "model.layers.0.linear_attn"
    specs[name] = replace(specs[name], num_prefill_checkpoint_blocks=2, tp_replicated=True)
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    group = next(g for g in groups if name in g.layer_names)
    assert group.layer_names == [name]
    assert group.kv_cache_spec.num_prefill_checkpoint_blocks == 2
    assert group.kv_cache_spec.tp_replicated


def test_pp_keeps_empty_draft_group_transfer_metadata():
    specs = _specs()
    specs["model.mtp.layers.48.self_attn.attn"] = specs["model.layers.3.self_attn.attn"]
    groups = get_qwen4_exp_kv_cache_groups(_config(), specs)
    projected = _ascend_project_kv_cache_groups_to_worker(groups, {"model.ple": specs["model.ple"]})
    assert projected[-1].is_eagle_group and not projected[-1].layer_names
    assert not projected[6].enable_kv_transfer


@pytest.mark.parametrize("transfer", [False, True])
def test_shared_worker_budget_reserves_alignment_once(transfer):
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = _config()
    worker.vllm_config.kv_transfer_config = SimpleNamespace() if transfer else None
    worker.model_config = worker.vllm_config.model_config
    worker.model_runner = SimpleNamespace(supports_page_strided_shared_kv_backing=True, use_sparse=True)
    worker.get_kv_cache_spec = _specs
    budget = 1_000_000_000
    assert worker._scale_kv_cache_memory_for_multi_group(budget) == budget - (2 * 1024 * 1024 if transfer else 0)
