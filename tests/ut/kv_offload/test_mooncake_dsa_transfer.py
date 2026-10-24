# SPDX-License-Identifier: Apache-2.0
import math
import random
from dataclasses import replace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_dsa_transfer as transfer

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    MAX_REGISTER_MEMORY_BYTES,
    DsaCacheLayout,
    DsaRegisterAtom,
    build_component_read,
    collect_bounded_register_regions,
    split_transfer_lists_at_region_boundaries,
)


def byte_pairs(plan):
    return [(dst + i, src + i) for dst, src, size in zip(*plan) for i in range(size)]


@pytest.mark.parametrize("cp,writers", [(1, 1), (1, 4), (8, 4), (16, 8), (2, 4), (4, 8)])
@pytest.mark.parametrize("pblock,dblock", [(4, 4), (4, 8), (8, 4)])
@pytest.mark.parametrize("start,end", [(0, 3), (5, 39), (20, 20), (0, 71)])
def test_main_exact_coverage_with_prefix_tail_and_page_geometry(cp, writers, pblock, dblock, start, end):
    source = tuple(100 + 3 * i for i in range(math.ceil(end / pblock / cp)))
    destination = tuple(200 - 2 * i for i in range(math.ceil(end / dblock)))
    local = DsaCacheLayout("main", 0, 10000, dblock * 2, dblock * 2 + 8, 1, dblock, "bf16")
    # Source pages are two tokens each, with padding between pages.
    remote = DsaCacheLayout("main", 0, 100000, 4, 8, pblock // 2, pblock, "bf16")
    seen = {}
    expected = {}
    for rank in range(writers):
        for shard in range(cp):
            plan = build_component_read(
                local,
                remote,
                source,
                destination,
                start,
                end,
                cp_size=cp,
                cp_rank=shard,
                writer_rank=rank,
                writer_size=writers,
                indexer=False,
            )
            for dst, src in byte_pairs(plan):
                assert dst not in seen
                seen[dst] = (shard, src)
            # Physical IDs have no relation to ordinal-based ownership.
            owned = {destination[g] for g in range(len(destination)) if g % writers == rank}
            assert all((dst - local.base) // local.stride in owned for dst, _ in byte_pairs(plan))
    for token in range(start, end):
        g, offset = divmod(token, dblock)
        p, poffset = divmod(token, pblock)
        src_page = source[p // cp] * remote.scale + poffset // 2
        for byte in range(2):
            expected[local.base + destination[g] * local.stride + offset * 2 + byte] = (
                p % cp,
                remote.base + src_page * remote.stride + poffset % 2 * 2 + byte,
            )
    assert seen == expected


@pytest.mark.parametrize("cp", [1, 8, 16])
@pytest.mark.parametrize("dtype,token_bytes", [("bf16", 2), ("int8", 1)])
def test_each_rank_receives_full_indexer_pages(cp, dtype, token_bytes):
    local = DsaCacheLayout("indexer", 0, 1000, 8 * token_bytes, 16 * token_bytes, 1, 8, dtype)
    remote = DsaCacheLayout("indexer", 0, 10000, 2 * token_bytes, 4 * token_bytes, 2 * cp, 4, dtype)
    source = (91, 27, 63, 20, 33)
    dest = (12, 51, 32)
    plans = [
        build_component_read(
            local, remote, source, dest, 5, 19, cp_size=cp, cp_rank=0, writer_rank=rank, writer_size=4, indexer=True
        )
        for rank in range(4)
    ]
    assert plans == [plans[0]] * 4
    expected = []
    for token in range(5, 19):
        page = source[token // (4 * cp)] * remote.scale + token % (4 * cp) // 2
        expected.extend(
            (
                local.base + dest[token // 8] * local.stride + token % 8 * token_bytes + i,
                remote.base + page * remote.stride + token % 2 * token_bytes + i,
            )
            for i in range(token_bytes)
        )
    assert byte_pairs(plans[0]) == expected


@pytest.mark.parametrize("change", [{"dtype": "int8"}, {"block_bytes": 7}, {"scale": 3}, {"stride": 1}])
def test_incompatible_geometry_fails_before_transfer(change):
    layout = DsaCacheLayout("main", 0, 1000, 8, 8, 1, 4, "bf16")
    with pytest.raises(ValueError):
        build_component_read(
            layout,
            replace(layout, **change),
            (1,),
            (2,),
            0,
            4,
            cp_size=1,
            cp_rank=0,
            writer_rank=0,
            writer_size=1,
            indexer=False,
        )


def test_missing_source_cannot_be_silently_truncated():
    layout = DsaCacheLayout("main", 0, 1000, 8, 8, 1, 4, "bf16")
    with pytest.raises(ValueError, match="coverage"):
        build_component_read(
            layout, layout, (), (2,), 0, 4, cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False
        )


@pytest.mark.parametrize("source,destination", [((5,), (1,)), ((1,), (5,))])
def test_physical_ids_must_fit_registered_capacity(source, destination):
    layout = DsaCacheLayout("main", 0, 1000, 8, 8, 1, 4, "bf16", capacity=5)
    with pytest.raises(ValueError, match="registered capacity"):
        build_component_read(
            layout, layout, source, destination, 0, 4, cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False
        )


@pytest.mark.parametrize(
    "size,expected",
    [
        (MAX_REGISTER_MEMORY_BYTES - 1, [MAX_REGISTER_MEMORY_BYTES - 1]),
        (MAX_REGISTER_MEMORY_BYTES, [MAX_REGISTER_MEMORY_BYTES]),
        (MAX_REGISTER_MEMORY_BYTES + 1, [MAX_REGISTER_MEMORY_BYTES, 1]),
        (2 * MAX_REGISTER_MEMORY_BYTES + 1, [MAX_REGISTER_MEMORY_BYTES, MAX_REGISTER_MEMORY_BYTES, 1]),
    ],
)
def test_oversized_register_atom_is_split_from_its_layout_base(size, expected):
    regions = collect_bounded_register_regions([DsaRegisterAtom(123, 123 + size, "npu:7", "host")])
    assert regions.ptrs == [123 + sum(expected[:index]) for index in range(len(expected))]
    assert regions.lengths == expected
    assert regions.locations == ["npu:7"] * len(expected)


def test_register_regions_merge_only_at_atom_boundaries():
    limit = 16
    regions = collect_bounded_register_regions(
        [
            DsaRegisterAtom(100, 108, "npu:0", "pool"),
            DsaRegisterAtom(110, 118, "npu:0", "pool"),
            DsaRegisterAtom(118, 126, "npu:0", "pool"),
            DsaRegisterAtom(1000, 1008, "*", "hbm"),
        ],
        max_region_bytes=limit,
    )
    assert regions.ptrs == [100, 110, 1000]
    assert regions.lengths == [8, 16, 8]
    assert regions.locations == ["npu:0", "npu:0", "*"]


def test_duplicate_oversized_alias_is_registered_only_once():
    atom = DsaRegisterAtom(
        100,
        100 + MAX_REGISTER_MEMORY_BYTES + 1,
        "*",
        "shared-storage",
    )
    regions = collect_bounded_register_regions([atom, atom])
    assert regions.ptrs == [100, 100 + MAX_REGISTER_MEMORY_BYTES]
    assert regions.lengths == [MAX_REGISTER_MEMORY_BYTES, 1]


def test_contained_alias_inside_oversized_chunk_is_safe():
    base = 100
    regions = collect_bounded_register_regions(
        [
            DsaRegisterAtom(base, base + MAX_REGISTER_MEMORY_BYTES + 10, "*", "shared-storage"),
            DsaRegisterAtom(
                base + MAX_REGISTER_MEMORY_BYTES, base + MAX_REGISTER_MEMORY_BYTES + 5, "*", "shared-storage"
            ),
        ]
    )
    assert regions.ptrs == [base, base + MAX_REGISTER_MEMORY_BYTES]
    assert regions.lengths == [MAX_REGISTER_MEMORY_BYTES, 10]


def test_overlapping_aliases_merge_only_when_union_fits_one_region():
    atoms = [
        DsaRegisterAtom(100, 112, "*", "shared-storage"),
        DsaRegisterAtom(108, 116, "*", "shared-storage"),
    ]
    regions = collect_bounded_register_regions(atoms, max_region_bytes=16)
    assert regions.ptrs == [100]
    assert regions.lengths == [16]
    with pytest.raises(ValueError, match="cross a registration boundary"):
        collect_bounded_register_regions(atoms, max_region_bytes=15)


def test_transfer_is_split_at_different_local_and_remote_edges():
    plan = split_transfer_lists_at_region_boundaries(
        [114],
        [1004],
        [20],
        local_base=100,
        remote_base=1000,
        max_region_bytes=16,
    )
    assert plan == ([114, 116, 126, 132], [1004, 1006, 1016, 1022], [2, 10, 6, 2])


def test_build_component_read_does_not_remerge_64_gib_registration_edge():
    size = MAX_REGISTER_MEMORY_BYTES + 1
    local = DsaCacheLayout("main", 0, 100, size, size, 1, 1, "bytes", 1)
    remote = DsaCacheLayout("main", 0, 1000, size, size, 1, 1, "bytes", 1)
    plan = build_component_read(
        local,
        remote,
        (0,),
        (0,),
        0,
        1,
        cp_size=1,
        cp_rank=0,
        writer_rank=0,
        writer_size=1,
        indexer=False,
    )
    assert plan == (
        [100, 100 + MAX_REGISTER_MEMORY_BYTES],
        [1000, 1000 + MAX_REGISTER_MEMORY_BYTES],
        [MAX_REGISTER_MEMORY_BYTES, 1],
    )


@pytest.mark.parametrize("indexer", [False, True])
def test_vectorized_randomized_byte_mapping_matches_scalar(indexer):
    rng = random.Random(14730)
    cache = {}
    for _ in range(400):
        cp, writers = rng.choice([1, 2, 3, 4, 8]), rng.choice([1, 2, 3, 4])
        block = rng.choice([4, 8, 12])
        # Include equal/different manager geometry and non-nested page grids.
        dblock = rng.choice([block, block, 8])
        lp = rng.choice([p for p in range(1, dblock + 1) if dblock % p == 0])
        rp = rng.choice([p for p in range(1, block + 1) if block % p == 0])
        size = rng.choice([1, 2, 4])
        start, end = sorted([rng.randrange(0, 90), rng.randrange(0, 90)])
        source = tuple(rng.sample(range(1, 64), math.ceil(end / (block * cp))))
        dest = tuple(rng.sample(range(1, 64), math.ceil(end / dblock)))
        local = DsaCacheLayout("x", 0, 10000, lp * size, lp * size + rng.randrange(4), dblock // lp, dblock, "x", 64)
        remote = DsaCacheLayout(
            "x", 0, 100000, rp * size, rp * size + rng.randrange(4),
            block * (cp if indexer else 1) // rp, block, "x", 64,
        )
        args = dict(cp_size=cp, cp_rank=rng.randrange(cp), writer_rank=rng.randrange(writers), writer_size=writers,
                    indexer=indexer)
        old_stats, new_stats = {}, {}
        expected = transfer._build_component_read_scalar(local, remote, source, dest, start, end,
                                                        statistics=old_stats, **args)
        actual = build_component_read(local, remote, source, dest, start, end,
                                      statistics=new_stats, mapping_cache=cache, **args)
        assert byte_pairs(actual) == byte_pairs(expected)
        assert len(byte_pairs(actual)) == len(set(byte_pairs(actual)))
        assert new_stats == old_stats
        assert len(cache) <= transfer.MAX_CACHED_DSA_MAPPINGS


@pytest.mark.parametrize("cp,writers", [(1, 1), (8, 2), (3, 4), (4, 6)])
@pytest.mark.parametrize("indexer", [False, True])
def test_aligned_path_avoids_scalar_and_preserves_all_owners(cp, writers, indexer):
    local = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x", 100)
    remote = replace(local, base=10000, scale=cp if indexer else 1)
    source, dest = tuple(range(30)), tuple(range(30))
    for shard in range(cp):
        for writer in range(writers):
            args = dict(cp_size=cp, cp_rank=shard, writer_rank=writer, writer_size=writers, indexer=indexer)
            expected = transfer._build_component_read_scalar(local, remote, source, dest, 1, 39, **args)
            with patch.object(
                transfer, "_build_component_read_scalar", side_effect=AssertionError("unexpected fallback")
            ):
                assert build_component_read(local, remote, source, dest, 1, 39, **args) == expected


def test_non_nested_pages_and_unequal_manager_blocks_use_scalar():
    local = DsaCacheLayout("x", 0, 1000, 8, 8, 3, 12, "x")
    remote = replace(local, block_bytes=12, stride=12, scale=2, base=10000)
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    for other in [remote, replace(local, block_tokens=4, scale=1)]:
        with patch.object(
            transfer, "_build_component_read_scalar", wraps=transfer._build_component_read_scalar
        ) as slow:
            build_component_read(local, other, (0, 1, 2), (3,), 0, 12, **args)
            slow.assert_called_once()


def test_cache_reuses_mapping_but_recomputes_each_layer_addresses():
    local = DsaCacheLayout("x", 0, 1000, 8, 16, 1, 4, "x", 20)
    remote = replace(local, base=10000)
    args = dict(cp_size=2, cp_rank=0, writer_rank=0, writer_size=2, indexer=False)
    cache = {}
    with patch.object(transfer, "_aligned_page_mapping", wraps=transfer._aligned_page_mapping) as mapper:
        build_component_read(local, remote, (3, 4), (1, 2, 3, 4), 0, 16, mapping_cache=cache, **args)
        other_local = replace(local, base=2000, stride=32, block_bytes=16, dtype="y", layer_name="y")
        other_remote = replace(remote, base=20000, stride=24, block_bytes=16, dtype="y", layer_name="y")
        actual = build_component_read(other_local, other_remote, (3, 4), (1, 2, 3, 4), 0, 16,
                                      mapping_cache=cache, **args)
        expected = transfer._build_component_read_scalar(other_local, other_remote, (3, 4), (1, 2, 3, 4), 0, 16, **args)
        assert actual == expected
        assert mapper.call_count == 1
    mapping = next(iter(cache.values()))
    assert not mapping.local_pages.flags.writeable
    # Capacity and dtype validation must not be bypassed by a cache hit.
    with pytest.raises(ValueError, match="capacity"):
        build_component_read(replace(local, capacity=1), remote, (3, 4), (1, 2, 3, 4), 0, 16,
                             mapping_cache=cache, **args)
    with pytest.raises(ValueError, match="dtype"):
        build_component_read(replace(local, dtype="bad"), remote, (3, 4), (1, 2, 3, 4), 0, 16,
                             mapping_cache=cache, **args)


def test_cache_keys_include_block_ids_interval_and_ownership():
    local = DsaCacheLayout("x", 0, 1000, 8, 16, 1, 4, "x")
    remote = replace(local, base=10000)
    cache = {}
    cases = [((3, 4), (1, 2, 3, 4), 0, 16, 0), ((4, 3), (1, 2, 3, 4), 0, 16, 0),
             ((3, 4), (4, 3, 2, 1), 0, 16, 0), ((3, 4), (1, 2, 3, 4), 3, 15, 0),
             ((3, 4), (1, 2, 3, 4), 0, 16, 1)]
    for source, dest, start, end, rank in cases:
        args = dict(cp_size=2, cp_rank=rank, writer_rank=rank, writer_size=2, indexer=False)
        assert build_component_read(local, remote, source, dest, start, end, mapping_cache=cache, **args) == \
            transfer._build_component_read_scalar(local, remote, source, dest, start, end, **args)
    assert len(cache) == transfer.MAX_CACHED_DSA_MAPPINGS


@pytest.mark.parametrize("start,end,rank", [(1, 1, 0), (0, 4, 1)])
def test_empty_or_unowned_endpoint_keeps_original_validation(start, end, rank):
    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")
    args = dict(cp_size=2, cp_rank=rank, writer_rank=0, writer_size=2, indexer=False)
    assert build_component_read(layout, layout, (), (), start, end, **args) == ([], [], [])
    with pytest.raises(ValueError, match="dtype"):
        build_component_read(layout, replace(layout, dtype="bad"), (), (), start, end, **args)


@pytest.mark.parametrize("change", [{"base": 2**63}, {"stride": 2**63}, {"base": -100}])
def test_large_addresses_preserve_python_integer_semantics(change):
    local = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")
    remote = replace(local, base=10000)
    local = replace(local, **change)
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    assert build_component_read(local, remote, (1,), (2,), 0, 4, **args) == \
        transfer._build_component_read_scalar(local, remote, (1,), (2,), 0, 4, **args)


def test_vectorized_registration_split_and_statistics_match_reference():
    size = MAX_REGISTER_MEMORY_BYTES // 2 + 1
    local = DsaCacheLayout("x", 0, 100, size, size, 1, 1, "x", 4)
    remote = replace(local, base=1000)
    old_stats, new_stats = {}, {}
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    expected = transfer._build_component_read_scalar(local, remote, (1, 2), (0, 1), 0, 2,
                                                    statistics=old_stats, **args)
    actual = build_component_read(local, remote, (1, 2), (0, 1), 0, 2, statistics=new_stats, **args)
    assert actual == expected
    assert old_stats == new_stats
    assert len(actual[0]) > 1


@pytest.mark.parametrize("field,value", [
    ("cp_size", 0), ("cp_rank", 2), ("writer_size", 0), ("writer_rank", 2),
    ("start_token", -1), ("end_token", -1),
])
def test_invalid_request_geometry_matches_scalar_error(field, value):
    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")
    args = dict(start_token=0, end_token=4, cp_size=2, cp_rank=0, writer_rank=0, writer_size=2, indexer=False)
    args[field] = value
    with pytest.raises(ValueError) as old:
        transfer._build_component_read_scalar(layout, layout, (1,), (2,), **args)
    with pytest.raises(ValueError) as new:
        build_component_read(layout, layout, (1,), (2,), **args)
    assert str(new.value) == str(old.value)


def test_vector_mapping_size_limit_uses_scalar(monkeypatch):
    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    monkeypatch.setattr(transfer, "MAX_VECTORIZED_DSA_SEGMENTS", 1)
    with patch.object(transfer, "_build_component_read_scalar", wraps=transfer._build_component_read_scalar) as scalar:
        actual = build_component_read(layout, layout, (1, 2), (3, 4), 0, 8, **args)
        scalar.assert_called_once()
    assert actual == transfer._build_component_read_scalar(layout, layout, (1, 2), (3, 4), 0, 8, **args)


def test_concurrent_endpoint_caches_do_not_share_mutable_state():
    from concurrent.futures import ThreadPoolExecutor

    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")

    def endpoint(rank):
        cache = {}
        source = tuple(5 + rank * 32 + i for i in range(16))
        dest = tuple(100 + rank * 32 + i for i in range(16))
        args = dict(cp_size=4, cp_rank=rank % 4, writer_rank=rank % 3, writer_size=3, indexer=False)
        actual = build_component_read(layout, layout, source, dest, 1, 63, mapping_cache=cache, **args)
        assert actual == transfer._build_component_read_scalar(layout, layout, source, dest, 1, 63, **args)
        return cache

    with ThreadPoolExecutor(max_workers=8) as pool:
        caches = list(pool.map(endpoint, range(16)))
    assert len({id(c) for c in caches}) == 16
    assert len({id(next(iter(c.values()))) for c in caches}) == 16


@pytest.mark.parametrize("physical_id", [-1, 2**63, 2**65])
def test_unusual_physical_ids_never_wrap_in_numpy(physical_id):
    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x")
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    actual = build_component_read(layout, layout, (physical_id,), (physical_id,), 0, 4, **args)
    assert actual == transfer._build_component_read_scalar(layout, layout, (physical_id,), (physical_id,), 0, 4, **args)


def test_main_mapping_expands_only_selected_block_segments():
    layout = DsaCacheLayout("x", 0, 1000, 256, 256, 1, 128, "x")
    args = (layout, layout, tuple(range(32)), tuple(range(256)), 0, 32768, 8)
    for shard, expected in [(0, 32), (1, 0)]:
        with patch.object(transfer.np, "arange", wraps=transfer.np.arange) as arange:
            mapping = transfer._aligned_page_mapping(*args, shard, 0, 2, False)
        assert mapping.counts.size == expected
        first_range = arange.call_args_list[0].args
        assert first_range == ((0, 256, 8) if shard == 0 else (256, 256, 1))


def test_missing_coverage_preserves_earlier_capacity_error():
    layout = DsaCacheLayout("x", 0, 1000, 8, 8, 1, 4, "x", 4)
    args = dict(cp_size=1, cp_rank=0, writer_rank=0, writer_size=1, indexer=False)
    with pytest.raises(ValueError, match="registered capacity"):
        build_component_read(layout, layout, (5,), (1, 2), 0, 8, **args)
