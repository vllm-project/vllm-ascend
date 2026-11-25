# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
    LayerwiseCachePlanner,
    get_spec_reuse_rules,
    plan_layerwise_cache_reuse,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    allocate_cache_backing,
    apply_layerwise_kv_cache_plan,
    get_cache_views,
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.models.qwen4_exp.cache_config import (
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_pool_bytes_per_block,
    make_slab_view,
)
from vllm_ascend.models.qwen4_exp.cache_reuse import QWEN4_EXP_SPEC_REUSE_RULES


def _groups():
    sources = [f"model.layers.{layer}.self_attn" for layer in range(3, 48, 4)]
    main = FullAttentionSpec(block_size=256, num_kv_heads=1, head_size=64, head_size_v=64, dtype=torch.bfloat16)
    raw = CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=32, head_size_v=0, dtype=torch.bfloat16, page_size_padded=65536
    )
    compressed = MLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, tokens_per_state=4, page_size_padded=65536
    )
    state = MambaSpec(
        block_size=256,
        shapes=((3, 64), (8, 32, 32)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
        page_size_padded=65536,
    )
    ple = MambaSpec(
        block_size=256, shapes=((3, 64),), dtypes=(torch.bfloat16,), mamba_cache_mode="align", tp_replicated=True
    )
    composite = {f"{source}.attn": main for source in sources}
    composite.update({f"{source}.indexer.compressed_key_cache": compressed for source in sources})
    return [
        KVCacheGroupSpec([f"{source}.indexer.raw_key_cache" for source in sources], raw),
        KVCacheGroupSpec(list(composite), UniformTypeKVCacheSpecs.from_specs(composite)),
        *[
            KVCacheGroupSpec([f"model.layers.{layer}.linear_attn" for layer in layers], state)
            for layers in (
                tuple(layer for layer in range(start, start + 16) if layer % 4 != 3) for start in (0, 16, 32)
            )
        ],
        KVCacheGroupSpec(["model.layers.1.ple"], ple),
    ]


def _runtime(buffers=2, independent=None, model_type="qwen4_exp_text"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen4_exp"),
            hf_text_config=SimpleNamespace(model_type=model_type),
            get_num_layers=lambda _: 48,
            get_total_num_hidden_layers=lambda: 48,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        speculative_config=None,
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": buffers,
                "layerwise_independent_layers": [0] if independent is None else independent,
            },
        ),
    )


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_model_rules_resolve_through_common_registry(model_type):
    assert get_spec_reuse_rules(_runtime(model_type=model_type)) is QWEN4_EXP_SPEC_REUSE_RULES


def test_rules_keep_ple_raw_and_hidden_resident():
    groups = _groups()
    config = get_qwen4_exp_kv_cache_config(_runtime(), groups, get_qwen4_exp_pool_bytes_per_block(groups))
    specs = get_layerwise_kv_cache_specs(config)
    for name, spec in specs.items():
        expected = name.endswith((".attn", ".compressed_key_cache", ".linear_attn"))
        assert LayerwiseCachePlanner.can_reuse_spec(spec, QWEN4_EXP_SPEC_REUSE_RULES) == expected
    hidden = HiddenStateCacheSpec(block_size=256, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    assert not LayerwiseCachePlanner.can_reuse_spec(hidden, QWEN4_EXP_SPEC_REUSE_RULES)
    with pytest.raises(TypeError):
        QWEN4_EXP_SPEC_REUSE_RULES[FullAttentionSpec] = False


@pytest.mark.parametrize("buffers", [2, 4, 8])
def test_common_planner_allocates_native_qwen_views_and_accounts_once(buffers):
    groups = _groups()
    runtime = _runtime(buffers)
    cost = get_qwen4_exp_pool_bytes_per_block(groups)
    probe = get_qwen4_exp_kv_cache_config(runtime, groups, cost)
    native = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * cost)
    specs = get_layerwise_kv_cache_specs(native)
    original_windows = get_cache_views(native, specs)
    native_layout = build_qwen4_kv_cache_layout(groups, native.num_blocks)

    probe_layout = plan_layerwise_cache_reuse(probe, runtime)
    layout = plan_layerwise_cache_reuse(native, runtime)
    assert probe_layout is not None
    assert layout is not None
    assert "model.layers.4.linear_attn" in layout.reused_cache_names
    worker_config = apply_layerwise_kv_cache_plan(native, runtime.kv_transfer_config, layout)

    assert len({tensor.size for tensor in worker_config.kv_cache_tensors}) == 1
    assert worker_config.kv_cache_tensors[0].size == 3 * probe_layout.kv_cache_tensors[0].size < 3 * cost
    assert all(tensor.layer_stride == 0 for tensor in worker_config.kv_cache_tensors)
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_cache_backing(worker_config, specs, allocate, layout.cache_view_sizes)
    assert allocations == [worker_config.kv_cache_tensors[0].size]
    views = get_cache_views(worker_config, specs, layout.cache_view_sizes)
    for name, (_, start, end) in views.items():
        _, old_start, old_end = original_windows[name]
        assert end - start == old_end - old_start
        assert raw[name].numel() == end - start
        assert raw[name].data_ptr() - start == raw[next(iter(raw))].data_ptr() - views[next(iter(raw))][1]

    # The same owner-relative component offsets used by MRV2 must remain valid.
    name = "model.layers.4.linear_attn"
    spec = specs[name]
    descriptor = next(tensor for tensor in worker_config.kv_cache_tensors if name in tensor.layers)
    conv, ssm = [
        make_slab_view(
            raw[name],
            dtype=dtype,
            num_blocks=native.num_blocks,
            item_shape=shape,
            storage_offset=native_layout.region(region).offset - native_layout.region("r1").offset,
            block_stride=descriptor.block_stride,
        )
        for shape, dtype, region in zip(spec.shapes, spec.dtypes, ("r1", "r2"), strict=True)
    ]
    conv.fill_(7)
    ssm.fill_(11)
    assert torch.all(conv == 7) and torch.all(ssm == 11)
    assert ssm.dtype == torch.float32
    if buffers == 2:
        # Pack rotation changes phase at the next GDN group. Main KV and
        # compressed indexer now require two distinct predecessor completions.
        assert layout.prefetch_layer_map[19] == (11, 15)

    assert get_cache_views(native, specs) == original_windows
    assert plan_layerwise_cache_reuse(native, runtime) == layout


@pytest.mark.parametrize("independent,buffers", [("all", 2), ([0], 48)])
def test_no_rotation_preserves_native_qwen_descriptors(independent, buffers):
    groups = _groups()
    runtime = _runtime(buffers, independent)
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    original = deepcopy(config.kv_cache_tensors)
    assert plan_layerwise_cache_reuse(config, runtime) is None
    assert config.kv_cache_tensors == original


@pytest.mark.parametrize("unsupported", [{"mamba_cache_mode": "all"}, {"num_prefill_checkpoint_blocks": 1}])
def test_common_planner_rejects_unsupported_qwen_state(unsupported):
    groups = _groups()
    runtime = _runtime()
    groups[2] = replace(groups[2], kv_cache_spec=replace(groups[2].kv_cache_spec, **unsupported))
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    with pytest.raises(NotImplementedError, match="unsupported state semantics"):
        plan_layerwise_cache_reuse(config, runtime)


def test_draft_state_stays_resident():
    groups = _groups()
    draft_name = "model.mtp.0.linear_attn"
    groups.append(KVCacheGroupSpec([draft_name], groups[2].kv_cache_spec))
    runtime = _runtime()
    cost = get_qwen4_exp_pool_bytes_per_block(groups)
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * cost)
    layout = plan_layerwise_cache_reuse(config, runtime)
    assert layout is not None
    assert draft_name not in layout.reused_cache_names
    assert layout.draft_entry_names == {draft_name}


def test_unregistered_model_keeps_native_placement():
    groups = _groups()
    runtime = _runtime(model_type="unregistered")
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 3 * get_qwen4_exp_pool_bytes_per_block(groups))
    original = deepcopy(config.kv_cache_tensors)
    assert plan_layerwise_cache_reuse(config, runtime) is None
    assert config.kv_cache_tensors == original
