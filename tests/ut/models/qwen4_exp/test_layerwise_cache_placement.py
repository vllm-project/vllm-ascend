# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.layerwise_kv_cache_plan import (
    LayerwiseKVCacheConfig,
    build_layerwise_cache_tensors,
    plan_cache_placement,
)
from vllm_ascend.models.qwen4_exp import cache_config
from vllm_ascend.models.qwen4_exp.cache_config import (
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_pool_bytes_per_block,
)
from vllm_ascend.worker.v2 import attn_utils


def _qwen_groups():
    def spec(cls, block_size, width, **kwargs):
        return cls(block_size=block_size, head_size=width, num_kv_heads=1, dtype=torch.bfloat16, **kwargs)

    def state(shapes):
        return MambaSpec(block_size=128, shapes=shapes, dtypes=(torch.bfloat16,) * len(shapes))

    main = spec(FullAttentionSpec, 256, 64, head_size_v=64)
    compressed = spec(MLAAttentionSpec, 256, 32, tokens_per_state=4)
    raw = spec(CircularBufferSpec, 8, 32, head_size_v=0)
    sources = [f"model.layers.{index}.self_attn" for index in range(3, 48, 4)]
    composite = {f"{s}.attn": main for s in sources}
    composite.update({f"{s}.indexer.compressed_key_cache": compressed for s in sources})
    recurrent = [f"model.layers.{i}.linear_attn" for i in range(48) if i % 4 != 3]
    return [
        KVCacheGroupSpec(list(composite), UniformTypeKVCacheSpecs.from_specs(composite)),
        KVCacheGroupSpec([f"{s}.indexer.raw_key_cache" for s in sources], raw),
        *(KVCacheGroupSpec(recurrent[i : i + 12], state(((64, 4), (128, 128)))) for i in range(0, 36, 12)),
        KVCacheGroupSpec(["model.ple"], state(((128, 5),))),
    ]


def _runtime(independent, buffers):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_independent_layers": independent,
                "layerwise_num_shared_buffers": buffers,
                "layerwise_prefetch_layers": 3,
            },
        ),
        cache_config=SimpleNamespace(prefix_cache_retention_interval=None, num_gpu_blocks_override=None),
    )


@pytest.mark.parametrize("independent, resident_state_pages", [([], 0), ([0], 1)])
def test_qwen_capacity_aliases_and_overwrite_order(independent, resident_state_pages, monkeypatch):
    groups = _qwen_groups()
    runtime = _runtime(independent, 2)
    cost = (2 + resident_state_pages) * 65536 + 12 * (512 + 4096) + 1280
    assert get_qwen4_exp_pool_bytes_per_block(groups, runtime) == cost
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 7 * cost + 123)
    assert config.num_blocks == 7
    assert config.kv_cache_groups == groups
    assert config.kv_cache_tensors[0].size + config.kv_cache_tensors[-1].size == 7 * cost
    plan = config.placement
    assert plan.prefetch_layers == 3
    addresses = {
        name: tensor.offset + i * tensor.layer_stride
        for tensor in config.kv_cache_tensors
        for i, name in enumerate(tensor.layers)
    }
    assert len(addresses) == sum(len(group.layer_names) for group in groups)
    main = [f"model.layers.{i}.self_attn.attn" for i in range(3, 48, 4)]
    assert len(config.kv_cache_tensors) == 5 + resident_state_pages
    main_tensors = [t for t in config.kv_cache_tensors if main[0] in t.layers or main[1] in t.layers]
    for buffer, tensor in enumerate(main_tensors):
        expected = main[buffer::2] + [
            name
            for group in groups[2:5]
            for name in [n for n in group.layer_names if n not in plan.resident_names][buffer::2]
        ]
        assert tensor.layers == expected
    assert all(t.layer_stride == 0 and t.block_stride == 65536 for t in main_tensors)
    assert sum(len(t.layers) for t in config.kv_cache_tensors) == len(addresses)
    assert len({addresses[name] for name in main}) == 2
    assert plan.predecessors[11] == (3,)
    for group in groups[2:5]:
        active = [name for name in group.layer_names if name not in plan.resident_names]
        for ordinal, name in enumerate(active):
            assert addresses[name] == addresses[main[ordinal % 2]]
            assert plan.buffer_by_name[name] == ordinal % 2
    if independent:
        assert addresses["model.layers.0.linear_attn"] not in {addresses[name] for name in main}
        assert plan.predecessors[4] == (1,)
        assert plan.predecessors[5] == (2,)
        assert plan.predecessors[6] == (4,)
    resident = [n for n in addresses if ".indexer." in n]
    assert len({addresses[n] for n in resident}) == 24
    assert set(resident).issubset(plan.resident_names)
    assert not set(resident).intersection(plan.rotating_names)

    runtime.additional_config = {}
    runtime.model_config = SimpleNamespace(hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace())
    runtime.cache_config.cache_dtype = "auto"
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    backend = SimpleNamespace(full_cls_name=lambda: "qwen-cache-test")
    layers = {name: SimpleNamespace(get_attn_backend=lambda: backend, num_heads=1) for name in addresses}
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    caches = attn_utils.allocate_kv_cache_main(config, torch.device("cpu"), None, [128] * len(groups))
    first, later = (caches[n][0] for n in (main[0], main[2]))
    compressed = caches[main[0].removesuffix(".attn") + ".indexer.compressed_key_cache"]
    compressed.fill_(5)
    first[1].fill_(7)
    saved = first[1].clone()
    later[1].fill_(11)
    assert first.data_ptr() == later.data_ptr()
    assert torch.all(first[1] == 11) and torch.all(saved == 7) and torch.all(compressed == 5)
    gdn_state = caches[groups[3].layer_names[0]]
    assert first.data_ptr() == gdn_state[0].data_ptr()
    # QSA kernel pages 0 and 1 belong to manager block 0. A live GDN
    # state in manager block 1 uses another page in the very same slot.
    gdn_state[0][1].fill_(13)
    assert torch.all(first[1] == 11) and torch.all(gdn_state[0][1] == 13)
    assert gdn_state[1].stride(0) * gdn_state[1].element_size() == 65536


@pytest.mark.parametrize("independent,buffers", [("all", 2), ([], 12), ([], 20)])
def test_no_temporal_alias_preserves_every_descriptor(independent, buffers):
    groups = _qwen_groups()
    baseline = _runtime([], 12)
    baseline.kv_transfer_config = None
    cost = get_qwen4_exp_pool_bytes_per_block(groups)
    original = get_qwen4_exp_kv_cache_config(baseline, groups, 3 * cost)
    planned = get_qwen4_exp_kv_cache_config(_runtime(independent, buffers), groups, 3 * cost)
    assert not planned.placement.rotating_names
    assert original.kv_cache_tensors == planned.kv_cache_tensors


def test_planner_uses_only_strides_groups_and_policy():
    # Two groups share one stride bucket with arbitrary model/cache names.
    groups = [
        KVCacheGroupSpec([f"decoder.layers.{i}.{role}" for i in ids], None)
        for role, ids in (("state", [0, 2, 4, 6]), ("key", [1, 3, 5, 7]))
    ]
    config = KVCacheConfig(3, [KVCacheTensor(384, g.layer_names, 96, 32) for g in groups], groups)
    original = deepcopy(config.kv_cache_tensors)
    layers = {n: int(n.split(".")[2]) for g in groups for n in g.layer_names}
    plan = plan_cache_placement(config, {"state": True, "key": True}, layers, {2}, set(), 2, 1)
    assert plan.buffer_by_name["decoder.layers.3.key"] == 1
    assert plan.buffer_by_name["decoder.layers.4.state"] == 1
    assert plan.predecessors == {6: (0,), 5: (1,), 7: (3,)}
    assert plan.resident_names == {"decoder.layers.2.state"}
    assert config.kv_cache_tensors == original


def test_resident_slots_overlay_groups_and_stay_separate_from_reuse():
    groups = _qwen_groups()
    runtime = _runtime([0, 3, 16, 32], 2)
    # One resident main slot serves all four groups. The compressed cache of
    # layer 3 still counts toward the twelve resident compressed pages.
    cost = 3 * 65536 + 12 * (512 + 4096) + 1280
    assert get_qwen4_exp_pool_bytes_per_block(groups, runtime) == cost
    config = get_qwen4_exp_kv_cache_config(runtime, groups, 5 * cost)
    assert config.num_blocks == 5
    bucket = next(b for b in config.placement.buckets if b.block_stride == 65536)
    assert len(bucket.resident_slots) == 1
    assert set(bucket.resident_slots[0]) == {
        "model.layers.0.linear_attn",
        "model.layers.3.self_attn.attn",
        "model.layers.16.linear_attn",
        "model.layers.32.linear_attn",
    }
    assert len(bucket.reuse_slots) == 2
    addresses = {
        name: tensor.offset + i * tensor.layer_stride
        for tensor in config.kv_cache_tensors
        for i, name in enumerate(tensor.layers)
    }
    resident_address = addresses[bucket.resident_slots[0][0]]
    assert all(addresses[name] == resident_address for name in bucket.resident_slots[0])
    assert all(addresses[name] != resident_address for slot in bucket.reuse_slots for name in slot)
    assert config.placement.predecessors[15] == (7,)


def test_compressed_cache_gets_its_own_stride_bucket(monkeypatch):
    monkeypatch.setitem(cache_config.REUSE_CONFIG, "self_attn.indexer.compressed_key_cache", True)
    runtime = _runtime([0], 2)
    cost = 3 * 65536 + 12 * 512 + 2 * 4096 + 1280
    assert get_qwen4_exp_pool_bytes_per_block(_qwen_groups(), runtime) == cost
    config = get_qwen4_exp_kv_cache_config(runtime, _qwen_groups(), 8 * cost)
    assert config.num_blocks == 8
    assert len(config.kv_cache_tensors) == 7
    main = next(b for b in config.placement.buckets if b.block_stride == 65536)
    compressed = next(b for b in config.placement.buckets if b.block_stride == 4096)
    assert not compressed.resident_slots
    assert len(compressed.reuse_slots) == len(main.reuse_slots) == 2
    assert config.placement.predecessors[11] == (3,)


@pytest.mark.parametrize("split_descriptors", [False, True])
def test_stride_buckets_span_descriptors_and_cache_roles(split_descriptors):
    # Covers a mixed main/indexer group and a separate recurrent group, as
    # well as multiple same-stride components belonging to the same group.
    main = [f"decoder.layers.{i}.{role}" for i in (3, 7, 11) for role in ("main", "indexer")]
    states = [f"decoder.layers.{i}.state" for i in (0, 1, 2, 4)]
    groups = [KVCacheGroupSpec(main, None), KVCacheGroupSpec(states, None)]
    tensors = [KVCacheTensor(4096, main, 256, 64), KVCacheTensor(4096, states, 256, 64)]
    if split_descriptors:
        tensors = [KVCacheTensor(4096, [name], 0, 64) for t in tensors for name in t.layers]
    config = LayerwiseKVCacheConfig(3, tensors, groups)
    indices = {name: int(name.split(".")[2]) for name in main + states}
    config.placement = plan_cache_placement(
        config, {"main": True, "indexer": True, "state": True}, indices, {0}, set(), 2, 3
    )
    (bucket,) = config.placement.buckets
    assert bucket.resident_slots == ((states[0],),)
    assert bucket.reuse_slots == (tuple(main[::2] + states[1::2]), tuple(main[1::2] + states[2::2]))
    assert config.placement.predecessors == {7: (3,), 11: (7,), 4: (1,)}
    output, size = build_layerwise_cache_tensors(config)
    assert size == 3 * 64 * 3
    assert len(output) == 3
    assert {name for t in output for name in t.layers} == set(main + states)
    assert all(t.size == size and t.block_stride == 64 for t in output)
    assert len({t.offset for t in output}) == 3
    assert all(t.layer_stride == 0 for t in output)
