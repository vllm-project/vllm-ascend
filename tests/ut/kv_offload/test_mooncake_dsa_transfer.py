# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import replace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    DsaCacheLayout,
    build_component_read,
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
