# SPDX-License-Identifier: Apache-2.0

import struct
from dataclasses import replace
from math import prod

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.kda_transfer import (
    MooncakeKDAStateMetadata,
    kda_remote_rank_groups,
    kda_transfer_slices,
)


def metadata(tp=1, pcp=1, *, recover=False, layout="DS", mode="none"):
    dtypes: tuple[str, ...] = ("torch.bfloat16", "torch.float32")
    if recover:
        dtypes += ("torch.float32", "torch.bfloat16")
    return MooncakeKDAStateMetadata(
        head_shards=pcp if tp == 1 else tp,
        pcp_head_sharded=tp == 1 and pcp > 1,
        conv_layout=layout,
        dtypes=dtypes,
        num_speculative_tokens=5,
        recoverssm=recover,
        cache_mode=mode,
    )


def geometry(meta):
    heads, dim, history = 8 // meta.head_shards, 4, 8
    shapes = [(3 * heads * dim, history) if meta.conv_layout == "DS" else (history, 3 * heads * dim), (heads, dim, dim)]
    sizes = [2, 4]
    if meta.recoverssm:
        shapes += [(heads, 6, dim), (heads, 6, 2 * dim)]
        sizes += [4, 2]
    lengths = [prod(shape) * size for shape, size in zip(shapes, sizes)]
    # Independent padded page strides must not affect per-page offsets.
    return shapes, lengths, [length + 64 for length in lengths]


def slices(local, remote, local_rank, remote_rank):
    ls, ll, lst = geometry(local)
    rs, rl, rst = geometry(remote)
    return kda_transfer_slices(
        local, remote, ls, rs, ll, rl, lst, rst, local_head_rank=local_rank, remote_head_rank=remote_rank
    )


def state_bytes(meta, rank):
    """An independent global-coordinate oracle, never using transfer offsets."""
    heads, dim, history = 8 // meta.head_shards, 4, 8
    channels = [
        projection * 8 * dim + head * dim + element
        for projection in range(3)
        for head in range(rank * heads, (rank + 1) * heads)
        for element in range(dim)
    ]
    if meta.conv_layout == "DS":
        conv = [channel * history + row for channel in channels for row in range(history)]
    else:
        conv = [channel * history + row for row in range(history) for channel in channels]
    recurrent = [
        head * dim * dim + row * dim + col
        for head in range(rank * heads, (rank + 1) * heads)
        for row in range(dim)
        for col in range(dim)
    ]
    return [
        bytearray(struct.pack("<" + "H" * len(conv), *conv)),
        bytearray(struct.pack("<" + "I" * len(recurrent), *recurrent)),
    ]


@pytest.mark.parametrize("layout", ["DS", "SD"])
@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize(
    ("p_tp", "p_pcp", "d_tp"), [(1, 1, 8), (1, 8, 1), (1, 8, 2), (1, 8, 8), (2, 4, 8), (4, 2, 1), (8, 1, 2), (2, 1, 2)]
)
def test_native_pcp_to_recoverssm_reconstructs_exact_state(layout, mode, p_tp, p_pcp, d_tp):
    local, remote = (
        metadata(d_tp, recover=True, layout=layout, mode=mode),
        metadata(p_tp, p_pcp, layout=layout, mode=mode),
    )
    for d_rank in range(d_tp):
        expected = state_bytes(local, d_rank)
        _, lengths, _ = geometry(local)
        destination = [bytearray([0xA5] * length) for length in lengths]
        coverage = [bytearray(length) for length in lengths[:2]]
        groups = kda_remote_rank_groups(
            local,
            remote,
            tp_rank=d_rank,
            pcp_rank=0,
            tp_size=d_tp,
            pcp_size=1,
            remote_tp_size=p_tp,
            remote_pcp_size=p_pcp,
        )
        for group in groups:
            # PCP replicas must all describe the same head interval.
            ranks = {remote.head_rank(worker // p_pcp, worker % p_pcp, p_tp, p_pcp) for worker in group}
            assert len(ranks) == 1
            p_rank = ranks.pop()
            source = state_bytes(remote, p_rank)
            plan = slices(local, remote, d_rank, p_rank)
            assert len(plan) == (4 if layout == "DS" else 25)
            for cache, local_offset, remote_offset, size in plan:
                assert local_offset + size <= len(destination[cache])
                assert remote_offset + size <= len(source[cache])
                destination[cache][local_offset : local_offset + size] = source[cache][
                    remote_offset : remote_offset + size
                ]
                assert not any(coverage[cache][local_offset : local_offset + size])
                coverage[cache][local_offset : local_offset + size] = bytes([1]) * size
        assert destination[:2] == expected
        assert all(all(mask) for mask in coverage)
        # Per-step records are not persistent prefill state. Do not transfer,
        # clear or overwrite them: verify writes its valid span before commit.
        assert all(buffer == bytearray([0xA5] * len(buffer)) for buffer in destination[2:])


def test_pcp_replicas_are_candidates_not_extra_head_shards():
    groups = kda_remote_rank_groups(
        metadata(1, recover=True),
        metadata(2, 4),
        tp_rank=0,
        pcp_rank=0,
        tp_size=1,
        pcp_size=1,
        remote_tp_size=2,
        remote_pcp_size=4,
    )
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


@pytest.mark.parametrize(
    "local_change",
    [
        {"conv_layout": "SD"},
        {"dtypes": ("torch.float16", "torch.float32", "torch.float32", "torch.bfloat16")},
        {"num_speculative_tokens": 0},
        {"cache_mode": "align"},
        {"version": 2},
    ],
)
def test_incompatible_contract_is_rejected(local_change):
    with pytest.raises(ValueError):
        slices(replace(metadata(recover=True), **local_change), metadata(), 0, 0)


def test_reverse_handoff_is_rejected():
    with pytest.raises(ValueError, match="producer to native"):
        slices(metadata(), metadata(recover=True), 0, 0)


@pytest.mark.parametrize("field", ["shapes", "lens", "strides"])
def test_invalid_page_geometry_is_rejected(field):
    local, remote = metadata(recover=True), metadata()
    ls, ll, lst = geometry(local)
    rs, rl, rst = geometry(remote)
    if field == "shapes":
        rs[0] = (rs[0][0] + 1, rs[0][1])
    elif field == "lens":
        rl[0] += 1
    else:
        rst[0] = rl[0] - 1
    with pytest.raises(ValueError):
        kda_transfer_slices(local, remote, ls, rs, ll, rl, lst, rst, local_head_rank=0, remote_head_rank=0)


def test_unknown_topology_is_rejected():
    with pytest.raises(ValueError, match="topology"):
        metadata(2, 4).head_rank(0, 0, 1, 8)


def test_non_integer_shard_ratio_is_rejected():
    with pytest.raises(ValueError, match="integer ratio"):
        kda_remote_rank_groups(
            metadata(4), metadata(3), tp_rank=0, pcp_rank=0, tp_size=4, pcp_size=1, remote_tp_size=3, remote_pcp_size=1
        )
