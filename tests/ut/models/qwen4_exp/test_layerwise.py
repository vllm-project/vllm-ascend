# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.layerwise_cache_config import get_model_summary
from vllm_ascend.core.layerwise_cache_plan import validate_physical_plan
from vllm_ascend.core.layerwise_cache_planner import materialize_layerwise_plan, plan_layerwise_cache
from vllm_ascend.models.layerwise import normalize_model_layerwise_capacities, rebuild_model_layerwise_plan
from vllm_ascend.models.qwen4_exp.layerwise import Qwen4LayerwiseAdapter

BUDGET = 12556015104
PREFIX = "language_model.model.layers."


def sample_groups():
    main = FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=256, head_size_v=256, dtype=torch.bfloat16)
    raw = CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=140, head_size_v=0, dtype=torch.bfloat16, page_size_padded=786432
    )
    compressed = MLAAttentionSpec(
        block_size=768, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, tokens_per_state=4, page_size_padded=786432
    )
    gdn = MambaSpec(
        block_size=768,
        shapes=((3, 1280), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=786432,
        mamba_cache_mode="align",
    )
    ple = MambaSpec(
        block_size=768,
        shapes=((3072, 30),),
        dtypes=(torch.bfloat16,),
        page_size_padded=786432,
        mamba_cache_mode="align",
        tp_replicated=True,
    )

    def group(specs):
        return KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(block_size=next(iter(specs.values())).block_size, kv_cache_specs=specs)
        )

    sources = [f"{PREFIX}{layer}.self_attn" for layer in range(3, 48, 4)]
    states = [f"{PREFIX}{layer}.linear_attn" for layer in range(48) if layer % 4 != 3]
    return [
        group({f"{source}.indexer.raw_key_cache": raw for source in sources}),
        group(
            {
                **{f"{source}.attn": main for source in sources},
                **{f"{source}.indexer.compressed_key_cache": compressed for source in sources},
            }
        ),
        *[group({name: gdn for name in states[start : start + 12]}) for start in (0, 12, 24)],
        group({f"{PREFIX}1.ple": ple}),
    ]


def make_plan(n=None):
    adapter = Qwen4LayerwiseAdapter()
    groups = sample_groups()
    policy = adapter.policy(groups, {"layerwise_num_shared_buffers": 2, "layerwise_independent_layers": [0, 1]})
    plan = (
        plan_layerwise_cache(adapter, groups, policy, BUDGET)
        if n is None
        else materialize_layerwise_plan(
            adapter, groups, policy, n, adapter.build_layout(groups, policy, n).required_bytes
        )
    )
    return adapter, groups, plan


def test_worker_profiling_preserves_physical_budget_before_group_planning():
    from vllm_ascend.worker.worker import NPUWorker

    groups = sample_groups()
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_config = SimpleNamespace(get_num_layers=lambda _: 48)
    worker.parallel_config = SimpleNamespace()
    worker.vllm_config = runtime_config()
    worker.get_kv_cache_spec = Mock(return_value=specs)
    with (
        patch("vllm_ascend.worker.worker.build_layerwise_reuse_layout", side_effect=AssertionError("legacy planner")),
        patch("vllm_ascend.worker.worker.get_kv_cache_groups", return_value=groups),
    ):
        assert worker._get_layerwise_kv_cache_memory_info(specs, {}) == (48, 48, 1.0)
        assert worker._scale_kv_cache_memory_for_multi_group(BUDGET) == BUDGET


def test_exact_capacity_and_descriptor_contract():
    adapter, groups, plan = make_plan()
    assert plan.num_blocks == 3648
    assert plan.required_bytes == BUDGET
    assert len(plan.backings) == 2
    assert plan.backings[0].size_bytes == 11883615744
    assert plan.backings[0].content_bytes == 11883257856
    assert plan.backings[1].size_bytes == 672399360
    before = copy.deepcopy(groups)
    config = adapter.export(KVCacheConfig(plan.num_blocks, [], groups), plan)
    assert groups == before and len(groups) == 6
    assert len(config.kv_cache_tensors) == 7
    assert [len(t.layers) for t in config.kv_cache_tensors] == [12, 12, 12, 12, 12, 12, 1]
    assert rebuild_model_layerwise_plan(config) == plan
    copied = copy.deepcopy(config)
    assert rebuild_model_layerwise_plan(copied) == plan
    assert get_model_summary(copied) == get_model_summary(config)


def test_rotation_offsets_and_resident_regions():
    _, _, plan = make_plan(2)
    slab = 786432 * 2
    for ordinal, layer in enumerate(range(3, 48, 4)):
        assert plan.view(f"{PREFIX}{layer}.self_attn.attn").offset_bytes == (2 + ordinal % 2) * slab
        assert (
            plan.view(f"{PREFIX}{layer}.self_attn.indexer.compressed_key_cache").offset_bytes
            == 4 * slab + ordinal % 2 * 49152 * 2
        )
        assert (
            plan.view(f"{PREFIX}{layer}.self_attn.indexer.raw_key_cache").offset_bytes
            == 4 * slab + 2 * 49152 * 2 + ordinal * 1120 * 2
        )
    states = [layer for layer in range(2, 48) if layer % 4 != 3]
    for ordinal, layer in enumerate(states):
        assert plan.view(f"{PREFIX}{layer}.linear_attn", "conv").offset_bytes == (2 + ordinal % 2) * slab
        assert plan.view(f"{PREFIX}{layer}.linear_attn", "ssm").offset_bytes == (2 + ordinal % 2) * slab + 7680
    assert plan.view(f"{PREFIX}0.linear_attn").offset_bytes == 0
    assert plan.view(f"{PREFIX}1.linear_attn").offset_bytes == slab


def test_cpu_views_preserve_kernel_interleave_and_resident_contents():
    adapter, groups, plan = make_plan(2)
    backings = {b.backing_id: torch.zeros(b.size_bytes, dtype=torch.int8) for b in plan.backings}
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    caches = {
        name: adapter.reshape(plan, name, spec, backings[plan.view(name).backing_id], 128)
        for name, spec in specs.items()
    }
    raw = caches[f"{PREFIX}3.self_attn.indexer.raw_key_cache"]
    raw.fill_(7)
    resident = caches[f"{PREFIX}0.linear_attn"]
    resident[0].fill_(8)
    resident[1].fill_(9)
    key, value = caches[f"{PREFIX}3.self_attn.attn"]
    key.fill_(2)
    value.fill_(3)
    assert value.data_ptr() - key.data_ptr() == 128 * 256 * 2
    assert key.stride(0) * key.element_size() == 2 * 128 * 256 * 2
    assert torch.all(raw == 7) and torch.all(resident[0] == 8) and torch.all(resident[1] == 9)
    assert caches[f"{PREFIX}11.self_attn.attn"][0].data_ptr() == key.data_ptr()
    assert caches[f"{PREFIX}1.ple"][0].is_contiguous()


def test_corrupt_alias_or_capacity_fails_before_allocation():
    adapter, groups, plan = make_plan(2)
    with pytest.raises(ValueError, match="Unauthorized alias"):
        validate_physical_plan(replace(plan, aliases=()))
    view = replace(plan.views[0], offset_bytes=1)
    with pytest.raises(ValueError, match="strided view"):
        validate_physical_plan(replace(plan, views=(view, *plan.views[1:])))
    config = adapter.export(KVCacheConfig(plan.num_blocks, [], groups), plan)
    config.num_blocks = 3
    with pytest.raises(ValueError):
        rebuild_model_layerwise_plan(config)


def test_normalization_rebuilds_every_offset_at_fixed_budget():
    adapter, groups, plan = make_plan(3)
    _, _, small = make_plan(2)
    configs = [adapter.export(KVCacheConfig(p.num_blocks, [], groups), p) for p in (plan, small)]
    normalized = normalize_model_layerwise_capacities(configs)
    assert [c.num_blocks for c in normalized] == [2, 2]
    rebuilt = [rebuild_model_layerwise_plan(c) for c in normalized]
    assert rebuilt[0].required_bytes == plan.required_bytes
    assert rebuilt[0].views == rebuilt[1].views


def test_scheduler_projection_retains_logical_summary():
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    adapter, groups, plan = make_plan(2)
    config = adapter.export(KVCacheConfig(plan.num_blocks, [], groups), plan)
    scheduler = generate_scheduler_kv_cache_config([config])
    assert get_model_summary(scheduler) == get_model_summary(config)
    assert get_model_summary(scheduler).has_reuse
    assert isinstance(config.kv_cache_groups[1].kv_cache_spec, UniformTypeKVCacheSpecs)


def runtime_config(independent=(0, 1)):
    return SimpleNamespace(
        additional_config={},
        use_v2_model_runner=True,
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": 2,
                "layerwise_independent_layers": list(independent),
            },
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=0),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1, world_size=1
        ),
        model_config=SimpleNamespace(
            get_num_layers=lambda _: 48,
            hf_text_config=SimpleNamespace(),
            hf_config=SimpleNamespace(),
            enforce_eager=True,
        ),
        speculative_config=None,
    )


def test_configuration_entry_and_worker_apply_are_idempotent():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
        apply_layerwise_kv_cache_plan,
    )
    from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config

    vc = runtime_config()
    config = get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    assert config.num_blocks == 3648
    plan = rebuild_model_layerwise_plan(config, vc)
    before = copy.deepcopy(config)
    for _ in range(2):
        apply_layerwise_kv_cache_plan(config, vc, model_plan=plan)
        assert config == before
    with pytest.raises(ValueError, match="matching physical plan"):
        apply_layerwise_kv_cache_plan(config, vc)
    vc.kv_transfer_config.kv_connector_extra_config["layerwise_num_shared_buffers"] = 3
    with pytest.raises(ValueError, match="policy changed"):
        rebuild_model_layerwise_plan(config, vc)


def test_all_independent_and_disabled_keep_original_descriptors():
    from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config

    vc = runtime_config(range(48))
    inactive = get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    assert not get_model_summary(inactive).has_reuse
    assert rebuild_model_layerwise_plan(inactive, vc) is None
    changed = runtime_config()
    with pytest.raises(ValueError, match="policy changed"):
        rebuild_model_layerwise_plan(inactive, changed)
    vc.kv_transfer_config = None
    disabled = get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    assert inactive.num_blocks == disabled.num_blocks == 1228
    assert inactive.kv_cache_tensors == disabled.kv_cache_tensors


def test_worker_allocator_uses_unique_backings(monkeypatch):
    from vllm_ascend.worker.v2 import attn_utils

    adapter, groups, plan = make_plan(2)
    config = adapter.export(KVCacheConfig(2, [], groups), plan)
    vc = runtime_config()
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vc)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    allocations = []

    def allocate(size, alignment, device):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.int8)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    raw = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    assert allocations == [b.size_bytes for b in plan.backings]
    assert len({t.data_ptr() for t in raw.values()}) == 2


def test_worker_kernel_selection_and_reshape_keep_qsa_virtual_pages(monkeypatch):
    from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend, QSAIndexerBackend
    from vllm_ascend.worker.utils import prepare_ascend_kernel_block_sizes
    from vllm_ascend.worker.v2 import attn_utils

    adapter, groups, plan = make_plan(2)
    config = adapter.export(KVCacheConfig(2, [], groups), plan)
    attention_groups = [
        [
            SimpleNamespace(
                backend=AscendQSABackend if name.endswith(".attn") else QSAIndexerBackend,
                layer_names=[name],
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
            )
            for name, spec in group.kv_cache_spec.kv_cache_specs.items()
        ]
        for group_id, group in enumerate(groups)
    ]
    # Exercise the real selection seam: non-affine descriptors must not make
    # upstream's block-outermost packing heuristic keep 768-token QSA pages.
    kernel_sizes = prepare_ascend_kernel_block_sizes(config, attention_groups)
    assert kernel_sizes[1] == 128
    assert kernel_sizes[2:] == [768] * 4
    with pytest.raises(ValueError, match="require their model layout summary"):
        prepare_ascend_kernel_block_sizes(KVCacheConfig(2, config.kv_cache_tensors, groups), attention_groups)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", runtime_config)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda _: False)
    backings = {b.backing_id: torch.zeros(b.size_bytes, dtype=torch.int8) for b in plan.backings}
    raw = {view.owner: backings[view.backing_id] for view in plan.views}
    views = attn_utils._reshape_kv_cache_v2(
        [group for subgroups in attention_groups for group in subgroups], raw, "auto", kernel_sizes, {}, config
    )
    key, value = views[f"{PREFIX}3.self_attn.attn"]
    compressed = views[f"{PREFIX}3.self_attn.indexer.compressed_key_cache"]
    assert key.shape == value.shape == (12, 128, 1, 256)
    assert compressed.shape == (12, 32, 1, 128)
    assert key.stride(0) * key.element_size() == 2 * 128 * 256 * 2
    assert value.data_ptr() - key.data_ptr() == 128 * 256 * 2


def _receive_config_from_worker_queue(handle, pipe):
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    reader = MessageQueue.create_from_handle(handle, 0)
    try:
        reader.wait_until_ready()
        config = reader.dequeue(timeout=10)
        pipe.send(rebuild_model_layerwise_plan(config).signature)
    finally:
        reader.shutdown()
        pipe.close()


@pytest.mark.skipif(sys.platform == "win32", reason="vLLM shared-memory worker queue requires Linux")
def test_real_worker_message_queue_preserves_model_contract():
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    adapter, groups, plan = make_plan(2)
    config = adapter.export(KVCacheConfig(2, [], groups), plan)
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    queue = MessageQueue(1, 1, max_chunk_bytes=1024 * 1024, max_chunks=2)
    process = ctx.Process(target=_receive_config_from_worker_queue, args=(queue.export_handle(), child))
    process.start()
    try:
        queue.wait_until_ready()
        queue.enqueue(config, timeout=10)
        assert parent.poll(10)
        assert parent.recv() == plan.signature
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        queue.shutdown()
        parent.close()
        child.close()


def test_admission_and_both_pool_cost_hooks_match_physical_plan():
    from vllm_ascend.models.layerwise import get_model_layerwise_pool_bytes
    from vllm_ascend.patch.platform.patch_kv_cache_utils import (
        _ascend_max_memory_usage_bytes_from_groups,
        _ascend_pool_bytes_per_block,
        _has_ascend_multi_allocation_kv_cache_layout,
    )

    groups = sample_groups()
    vc = runtime_config()
    vc.model_config.max_model_len = 8192
    vc.cache_config.mamba_cache_mode = "align"
    vc.max_in_flight_tokens = 1536
    assert get_model_layerwise_pool_bytes(vc, groups) == 3441792
    assert _ascend_pool_bytes_per_block(vc, groups) == 3441792
    assert _has_ascend_multi_allocation_kv_cache_layout(groups)
    blocks = sum(
        (g.kv_cache_spec.max_memory_usage_bytes(vc) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )
    assert _ascend_max_memory_usage_bytes_from_groups(vc, groups) == 3441792 * blocks


def test_rank_normalization_hook_preserves_budgets(monkeypatch):
    import vllm_ascend.patch.platform.patch_kv_cache_utils as patch

    adapter, groups, plan = make_plan(2)
    configs = [adapter.export(KVCacheConfig(2, [], groups), plan) for _ in range(2)]
    budgets = [plan.required_bytes + 345, plan.required_bytes + 678]
    monkeypatch.setattr(patch, "_orig_get_kv_cache_configs", lambda *args, **kwargs: configs)
    normalized = patch._ascend_get_kv_cache_configs(runtime_config(), [{}, {}], budgets)
    assert [c.num_blocks for c in normalized] == [2, 2]
    assert [rebuild_model_layerwise_plan(c).required_bytes for c in normalized] == budgets


def test_unknown_spec_missing_component_and_over_budget_override():
    from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config

    vc = runtime_config()
    vc.cache_config.num_gpu_blocks_override = 3649
    with pytest.raises(ValueError, match="needs .* available"):
        get_qwen4_exp_kv_cache_config(vc, sample_groups(), BUDGET)
    groups = sample_groups()
    groups[1].layer_names.pop()
    with pytest.raises(ValueError, match="one-to-one"):
        Qwen4LayerwiseAdapter().policy(groups, {})


def test_unsupported_consumer_only_and_kvpp_rejected_before_allocation():
    from vllm_ascend.models.layerwise import get_model_layerwise_context

    vc = runtime_config()
    vc.kv_transfer_config.kv_role = "kv_consumer"
    with pytest.raises(ValueError, match="requires saving"):
        get_model_layerwise_context(vc, sample_groups())
    vc.kv_transfer_config.kv_role = "kv_both"
    vc.additional_config = {"enable_kvpp": True}
    vc.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="KVPP"):
        get_model_layerwise_context(vc, sample_groups())
    vc.additional_config = {}
    vc.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce-eager"):
        get_model_layerwise_context(vc, sample_groups())


def test_hidden_owner_is_private_and_excluding_compressed_keeps_linked_main():
    adapter, groups, base = make_plan(2)
    name = "language_model.model.cache_only_layers.0"
    hidden = HiddenStateCacheSpec(block_size=768, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    groups.append(KVCacheGroupSpec([name], hidden))
    main = PREFIX + "3.self_attn.attn"
    compressed = PREFIX + "3.self_attn.indexer.compressed_key_cache"
    policy = adapter.policy(groups, runtime_config().kv_transfer_config.kv_connector_extra_config, (compressed,))
    plan = adapter.build_layout(groups, policy, 2)
    validate_physical_plan(plan)
    owners = {owner.name: owner for owner in plan.owners}
    assert owners[name].resident and owners[name].physical_layer == 0
    assert owners[main].resident and owners[compressed].resident
    hidden_backing = next(b for b in plan.backings if b.backing_id == plan.view(name).backing_id)
    assert hidden_backing.size_bytes == 2 * hidden.page_size_bytes
    assert plan.required_bytes == base.required_bytes + 2 * (hidden.page_size_bytes + 786432 + 49152)
    assert all(dep.owner not in (name, main, compressed) for dep in plan.dependencies)


def test_summary_version_and_missing_contract_are_rejected():
    adapter, groups, plan = make_plan(2)
    config = adapter.export(KVCacheConfig(2, [], groups), plan)
    summary = config.layerwise_summary
    config.layerwise_summary = replace(summary, version=summary.version + 1)
    with pytest.raises(ValueError, match="version mismatch"):
        rebuild_model_layerwise_plan(config)
    config.layerwise_summary = None
    with pytest.raises(ValueError, match="Non-affine"):
        rebuild_model_layerwise_plan(config)
