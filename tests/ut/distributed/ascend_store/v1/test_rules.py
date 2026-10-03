"""Verify statically selected KV rules without an execution graph."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import (
    KVMemoryRule,
    KVPoolRules,
    KVPoolRuleSpec,
    compile_kv_pool_rules,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    TPPartitionSpec,
)

from .helpers import make_backend_spec, make_schedule, make_topology


def _registration(topology, *, block_length: int = 32):
    bases = {}
    lengths = {}
    strides = {}
    layer_offsets = {}
    for group in topology.transfer_groups:
        group_bases = []
        group_lengths = []
        group_strides = []
        group_offsets = [0]
        for layer in group.layers:
            group_bases.append(1000 + group.group_id * 10000 + layer.physical_layer_id * 1000)
            group_lengths.append(block_length)
            group_strides.append(64)
            group_offsets.append(len(group_bases))
        bases[group.group_id] = group_bases
        lengths[group.group_id] = group_lengths
        strides[group.group_id] = group_strides
        layer_offsets[group.group_id] = group_offsets
    return bases, lengths, strides, layer_offsets


def _rule_spec(topology: KVPoolTopology, *, layerwise: bool = False) -> KVPoolRuleSpec:
    return KVPoolRuleSpec(topology, "fake", 64, use_layerwise=layerwise)


def test_rules_bind_static_facts_once_then_share_dynamic_rows(monkeypatch) -> None:
    topology = replace(
        make_topology(),
        tp_size=2,
        pp_size=2,
        dcp_size=2,
        pcp_rank=1,
        pcp_size=2,
        tp_partition=TPPartitionSpec(False, 2, 1),
    )
    spec = _rule_spec(topology)
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import compiler

    monkeypatch.setattr(
        compiler,
        "resolve_backend_spec",
        lambda _name: make_backend_spec(layerwise_access=None, requires_exists_before_put=True),
    )
    phi = compile_kv_pool_rules(spec)(*_registration(topology))

    assert type(phi) is KVPoolRules
    assert type(phi.memory) is KVMemoryRule
    assert phi.group_ids == (0,)
    assert phi.requires_store_observation
    assert phi.required_object_sizes is None
    assert KVMemoryRule.__slots__ == ("full", "partial", "store_full", "store_partial")
    assert phi.object_size(0) == 64
    assert phi.lookup_selection(("h0",), TokenRange(0, 4)).groups[0].chunk_mask is None

    hashes = ["h0", "h1", "h2"]
    load_rows = phi.load_rows(0, 12, hashes, [4, 5, 6], mask=(True, False, True))
    store_rows = phi.store_rows(0, 12, hashes, [4, 5, 6], mask=(True, False, True))
    assert load_rows[0].tolist() == [0, 8]
    assert load_rows[1].tolist() == [4, 4]
    assert load_rows[2] == ("h0", "h2")
    assert load_rows[3].tolist() == [4, 6]
    # Ownership applies to the already filtered candidate ordinal.
    assert store_rows[0].tolist() == [8]
    assert store_rows[2] == ("h2",)
    assert store_rows[3].tolist() == [6]

    lookup_axes = phi.lookup_keys(0, ("h0",))
    assert len(lookup_axes) == 8  # PP x DCP x effective head rank.
    assert len({axis[0] for axis in lookup_axes}) == 8
    assert phi.admit_store([True, False, False]).tolist() == [False, True, True]

    # Configuration and registration inputs are no longer consulted after binding.
    monkeypatch.setattr(
        compiler,
        "resolve_backend_spec",
        lambda _name: (_ for _ in ()).throw(AssertionError("static configuration was read again")),
    )
    ranges = phi.memory.partial(0, load_rows[3], load_rows[1])
    keys = phi.load_keys(0, load_rows[2])
    backend_keys, addresses, sizes = phi.format_ranges(keys, ranges)
    assert backend_keys == [axis[0] for axis in keys] + [axis[1] for axis in keys]
    assert addresses == [[1256, 2256], [1384, 2384]]
    assert sizes == [[32, 32], [32, 32]]


def test_rule_compiler_rejects_unsupported_static_compositions() -> None:
    cases = (
        (make_topology(tp_mismatch=True, consumer_pipeline_partitions=(1, 1)), False, "Consumer pipeline"),
        (make_topology(tp_mismatch=True), True, "TP mismatch"),
        (make_topology(consumer_pipeline_partitions=(1, 1)), True, "consumer pipeline"),
    )
    for topology, layerwise, message in cases:
        with pytest.raises(ValueError, match=message):
            compile_kv_pool_rules(
                _rule_spec(topology, layerwise=layerwise),
                layerwise_full_key=(lambda *_args: "key") if layerwise else None,
            )


def test_memory_rules_select_only_the_backend_layout_they_consume(monkeypatch) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import compiler

    cases = (
        ("bulk", make_topology(), make_schedule(), None),
        ("key-range", make_topology(), make_schedule(layerwise=True), LayerwiseAccessKind.KEY_RANGE),
        ("strided", make_topology(tp_mismatch=True), make_schedule(), None),
        (
            "pipeline-store",
            make_topology(consumer_pipeline_partitions=(1, 1)),
            make_schedule(),
            None,
        ),
    )
    for name, topology, schedule, layerwise_access in cases:
        monkeypatch.setattr(
            compiler,
            "resolve_backend_spec",
            lambda _name, access=layerwise_access: make_backend_spec(layerwise_access=access),
        )
        full_key = (
            (lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}")
            if schedule.requires_layerwise_backend
            else None
        )
        block_length = 31 if name == "bulk" else 32
        phi = compile_kv_pool_rules(
            _rule_spec(topology, layerwise=schedule.requires_layerwise_backend),
            layerwise_full_key=full_key,
        )(*_registration(topology, block_length=block_length))
        block_ids = np.asarray([1, 3], dtype=np.uint64)

        if name == "bulk":
            local, sizes, splits = phi.memory.partial(
                0,
                block_ids,
                np.asarray([3, 4], dtype=np.uint64),
            )
            assert local.tolist() == [1064, 2064, 1192, 2192]
            # Divide last: a 31-byte Block transfers floor(31 * 3 / 4).
            assert sizes.tolist() == [23, 23, 31, 31]
            assert splits.tolist() == [0, 2, 4]
        elif name == "key-range":
            first = phi.memory.full(0, block_ids, layer_id=0)
            second = phi.memory.full(0, block_ids, layer_id=1)
            assert first[0].tolist() == [1064, 1192]
            assert first[2].tolist() == [0, 0]
            assert second[0].tolist() == [2064, 2192]
            assert second[2].tolist() == [32, 32]
            assert first[4].tolist() == second[4].tolist() == [64, 64]
            assert phi.load_keys(0, ("a", "b")) == (("g0:p0:h0:a", "g0:p0:h0:b"),)
            _, _, _, offsets = phi.format_ranges(
                phi.load_keys(0, ("a", "b")),
                first,
                object_bases=None,
            )
            assert offsets == [[0], [0]]
        elif name == "strided":
            full = phi.memory.full(0, block_ids[:1])
            partial = phi.memory.partial(0, block_ids[:1], np.asarray([2], dtype=np.uint64))
            assert full[2].tolist() == [0, 8, 16]
            assert partial[2].tolist() == [0, 4, 8]
            assert partial[1].tolist() == [4] * 8
            assert len(phi.load_keys(0, ("a",))) == 2
        else:
            store = phi.memory.store_full(0, block_ids[:1])
            assert store[0].tolist() == [1064, 2064]
            assert store[2].tolist() == [0, 1, 2]
            assert "@pp_rank:0" in phi.store_keys(0, ("a",))[0][0]
            assert "@pp_rank:1" in phi.store_keys(0, ("a",))[1][0]


def test_hybrid_checkpoint_tail_and_gva_rules_keep_their_distinct_contracts(monkeypatch) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import compiler

    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0",)),),
            KeyMetadata("model", 0, 0, 0, 0, cache_family="attention"),
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(
                block_size=8,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            (KVPoolLayerTopology(1, ("layers.1",)),),
            KeyMetadata("model", 0, 0, 0, 1, cache_family="state"),
        ),
    )
    hybrid = KVPoolTopology(
        tp_rank=0,
        tp_size=1,
        pp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_size=1,
        put_step=1,
        cache_transfer_granularity=8,
        hash_block_size=4,
        tp_partition=TPPartitionSpec(False, 1, 1),
        groups=groups,
        transfer_group_ids=(0, 1),
        consumer_pipeline_partitions=None,
    )
    monkeypatch.setattr(
        compiler,
        "resolve_backend_spec",
        lambda _name: make_backend_spec(layerwise_access=None),
    )
    phi = compile_kv_pool_rules(_rule_spec(hybrid))(*_registration(hybrid))

    hybrid_consumer = replace(hybrid, consumer_pipeline_partitions=(1, 1))
    consumer_phi = compile_kv_pool_rules(_rule_spec(hybrid_consumer))(*_registration(hybrid_consumer))
    for group_id, expected_pp_rank in ((0, 0), (1, 1)):
        keys = consumer_phi.store_keys(group_id, ("h0",))
        ranges = consumer_phi.memory.store_full(group_id, (1,))
        assert len(keys) == 1
        assert f"@pp_rank:{expected_pp_rank}" in keys[0][0]
        assert consumer_phi.format_ranges(keys, ranges)[0] == [keys[0][0]]

    # Hybrid selects fine-grained Lookup for every transferable group.
    for group_id in (0, 1):
        lookup = phi.lookup_chunks(group_id, 8, ("h0", "h1"))
        assert lookup[0].tolist() == [0, 0]
        assert lookup[1].tolist() == [4, 8]
        assert lookup[2] == ("h0", "h1")

    # A fine Lookup tail remains usable before a complete grouped hash exists.
    tail = phi.load_rows(1, 4, ("h0",), (7,), tail_boundary_token=4)
    assert tail[0].tolist() == [0]
    assert tail[1].tolist() == [4]
    assert tail[2] == ("h0",)
    assert tail[3].tolist() == [7]
    assert "@group:1@cache_role:kv@cache_family:state@" in phi.load_keys(1, tail[2])[0][0]
    # A suffix beginning inside the Block still maps the whole physical Block.
    assert phi.load_rows(1, 8, ("h0", "h1"), (7,), start_token=4)[0].tolist() == [0]

    checkpoint = phi.checkpoint_rows(
        4,
        ("h0",),
        {1: 7},
        {0: (3,), 1: (99,)},
    )
    assert tuple(group_id for group_id, _ in checkpoint) == (1, 0)
    assert checkpoint[0][1][1].tolist() == [8]
    assert checkpoint[0][1][3].tolist() == [7]  # exact source, not the Block Table
    assert checkpoint[1][1][1].tolist() == [8]  # companion uses full physical extent
    assert phi.memory.partial(1, (7,), (4,))[1].tolist() == [32]
    assert phi.checkpoint_rows(4, ("h0",), {1: 0}, {0: (3,), 1: (99,)}) == ()

    layerwise = replace(make_topology(), tp_rank=1, tp_size=2, put_step=2)
    monkeypatch.setattr(
        compiler,
        "resolve_backend_spec",
        lambda _name: make_backend_spec(layerwise_access=LayerwiseAccessKind.GVA),
    )

    def full_key(group: int, value: str, head: int, stage: int) -> str:
        return f"g{group}:p{stage}:h{head}:{value}"

    binder = compile_kv_pool_rules(
        _rule_spec(layerwise, layerwise=True),
        layerwise_full_key=full_key,
    )
    with pytest.raises(ValueError, match="global object sizes"):
        binder(*_registration(layerwise))

    parallel_layerwise = replace(make_topology(group_ids=(0, 1)), pp_size=2)
    parallel_binder = compile_kv_pool_rules(
        _rule_spec(parallel_layerwise, layerwise=True),
        layerwise_full_key=full_key,
    )
    with pytest.raises(ValueError, match="group 1.*global object offset"):
        parallel_binder(
            *_registration(parallel_layerwise),
            object_sizes={0: 128, 1: 128},
            object_offsets={0: 0},
        )

    gva_phi = binder(
        *_registration(layerwise),
        object_sizes={0: 128},
        object_offsets={0: 16},
    )
    block_ids = np.asarray([1, 3], dtype=np.uint64)
    counts = np.asarray([2, 3], dtype=np.uint64)
    second_layer = gva_phi.memory.partial(0, block_ids, counts, layer_id=1)
    assert gva_phi.required_object_sizes(second_layer).tolist() == [64, 72]
    assert gva_phi.required_object_sizes(second_layer, (False, True)).tolist() == [72]
    assert second_layer[4].tolist() == [128, 128]

    # Admission owns the mask; GVA receives one compact base per selected key.
    remote, local, sizes = gva_phi.format_ranges(
        gva_phi.load_keys(0, ("a", "b")),
        second_layer,
        object_bases=np.asarray([20_000], dtype=np.uint64),
        selected_objects=(False, True),
    )
    assert remote.tolist() == [20_048]
    assert local.tolist() == [2192]
    assert sizes.tolist() == [24]
    # The non-leader rule returns before request hashes are interpreted.
    assert gva_phi.store_rows(0, 8, ("unused", "unused"), (1, 2))[0].size == 0
