"""Shared hash/block pairing, without Load/Store selection policy or Backend access."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import BlockHashMapping


@pytest.mark.parametrize(
    "block_size,hash_block_size,hashes,block_ids,start,end,tail,expected",
    (
        (4, 4, (b"a", b"b", b"c"), (7, 11, 13), 4, 12, None, ((1, b"b", 11), (2, b"c", 13))),
        (8, 4, (b"a", b"b", b"c", b"d", b"e", b"f"), (7, 11, 13), 8, 24, None, ((1, b"d", 11), (2, b"f", 13))),
        (8, 4, (b"a", b"b", b"c", b"d", b"e", b"f"), (11, 13), 8, 24, None, ((1, b"d", 11), (2, b"f", 13))),
        (12, 4, (b"a", b"b", b"c", b"d", b"e", b"f"), (7, 11), 0, 24, None, ((0, b"c", 7), (1, b"f", 11))),
        (12, 4, (b"a", b"b", b"c", b"d", b"e", b"f"), (11,), 12, 24, None, ((1, b"f", 11),)),
        (8, 4, (b"a", b"b", b"c", b"d", b"e"), (7, 11, 13), 8, 20, None, ((1, b"d", 11),)),
        (8, 4, (b"a", b"b", b"c", b"d", b"e"), (7, 11, 13), 8, 20, 20, ((1, b"d", 11), (2, b"e", 13))),
        (8, 4, (b"a", b"b", b"c", b"d", b"e", b"f"), (11, 13), 8, 20, 24, ((1, b"d", 11), (2, b"f", 13))),
        (4, 4, (b"a", b"b", b"c"), (-1, 0, 9), 0, 12, None, ((0, b"a", -1), (1, b"b", 0), (2, b"c", 9))),
        (4, 4, ("a", "b"), (2, 3), 0, 8, None, ((0, "a", 2), (1, "b", 3))),
        (4, 4, (), (1,), 0, 4, None, ()),
        (4, 4, (b"a",), (), 0, 4, None, ()),
        (4, 4, (b"a",), (1,), 4, 4, None, ()),
        (4, 4, (b"a",), (1,), 0, 0, None, ()),
    ),
)
def test_pairing_preserves_logical_position_and_caller_supplied_block_ids(
    block_size, hash_block_size, hashes, block_ids, start, end, tail, expected
):
    mapping = BlockHashMapping(block_size, hash_block_size)
    assert (
        tuple(mapping.pair_block_ids(hashes, block_ids, start_token=start, end_token=end, tail_boundary=tail))
        == expected
    )


@pytest.mark.parametrize("boundary", (16, 28))
def test_tail_must_cover_endpoint_and_stay_in_its_physical_block(boundary):
    with pytest.raises(ValueError, match="Tail-key boundary"):
        tuple(
            BlockHashMapping(8, 4).pair_block_ids(
                (b"a",) * 7, (1, 2, 3), start_token=0, end_token=20, tail_boundary=boundary
            )
        )


@pytest.mark.parametrize("boundary,hashes", ((0, (b"a",)), (6, (b"a", b"b")), (12, (b"a", b"b"))))
def test_exact_boundary_hash_rejects_unaligned_or_missing_identity(boundary, hashes):
    with pytest.raises(ValueError, match="Token boundary"):
        BlockHashMapping(8, 4).hash_at_boundary(boundary, hashes)


@pytest.mark.parametrize("block_size,hash_block_size", ((0, 4), (8, 0), (6, 4)))
def test_geometry_is_validated_once_at_binding(block_size, hash_block_size):
    with pytest.raises(ValueError, match="positive multiple"):
        BlockHashMapping(block_size, hash_block_size)


@pytest.mark.parametrize(
    "fine,start,end,expected",
    (
        (False, 8, 24, ((8, 8, b"d"), (16, 8, b"f"))),
        (True, 8, 20, ((8, 4, b"c"), (8, 8, b"d"), (16, 4, b"e"))),
        (True, 20, 22, ()),
        (False, 24, 24, ()),
    ),
)
def test_content_ranges_do_not_require_a_local_block_table(fine, start, end, expected):
    chunks = BlockHashMapping(8, 4).iter_hash_ranges(
        (b"a", b"b", b"c", b"d", b"e", b"f"), start_token=start, end_token=end, fine_grained=fine
    )
    assert tuple(chunks) == expected
