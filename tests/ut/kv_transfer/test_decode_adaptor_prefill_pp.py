"""Unit tests for ``decode_adaptor_prefill_pp`` layer-boundary splitting.

Covers the PP (pipeline parallel) key/addr/size re-splitting logic of
``ChunkedTokenDatabase`` used by the non-layerwise AscendStore PD path:

- Heterogeneous-layer models (DeepSeek-V4: dense + DSA multi-cache-spec
  layers + MTP tail layer) must be split exactly on physical-layer
  boundaries via ``group_layer_cache_entry_offsets``.
- Homogeneous-layer models without the offsets table keep the legacy
  uniform ``caches_per_layer`` split.
"""

from typing import Any

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
)

BASE_KEY = (
    "DSV4@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0"
    "@group:0@cache_role:kv@cache_family:default@hash_abcd"
)


def _make_db(
    partitions: list[int],
    num_layers: int,
    offsets: list[int] | None,
    total_entries: int,
) -> tuple[ChunkedTokenDatabase, Any]:
    meta = KeyMetadata(
        model_name="DSV4",
        head_or_tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
        kv_cache_group_id=0,
    )
    db = ChunkedTokenDatabase([meta], block_size=[32], partitions=partitions)
    db.set_group_buffers(
        group_kv_caches_base_addr={0: [10000 + 10 * i for i in range(total_entries)]},
        group_block_len={0: [16] * total_entries},
        group_block_stride=None,
        group_cache_families={0: "default"},
        group_num_layers={0: num_layers},
        group_layer_cache_entry_offsets={0: offsets} if offsets else {},
    )
    return db, meta


def _dsv4_layout() -> tuple[list[int], int, list[int]]:
    """DeepSeek-V4-like layout: 2 dense (1 entry) + 41 DSA (3 entries) + 1 MTP."""
    entries_per_layer = [1, 1] + [3] * 41 + [1]
    offsets = [0]
    for cnt in entries_per_layer:
        offsets.append(offsets[-1] + cnt)
    return entries_per_layer, len(entries_per_layer), offsets


class TestHeterogeneousLayerSplit:
    """DSV4-like layout with precise per-layer offsets."""

    def test_split_lands_on_layer_boundary(self):
        entries_per_layer, num_layers, offsets = _dsv4_layout()
        total = sum(entries_per_layer)  # 126
        db, _ = _make_db([21, 22], num_layers, offsets, total)

        new_key, new_addr, new_size = db.decode_adaptor_prefill_pp(
            [BASE_KEY],
            [[10000 + 10 * i for i in range(total)]],
            [[16] * total],
            kv_cache_group_id=0,
        )

        # layers 0..20 -> 2*1 + 19*3 = 59 entries; rest -> 67
        expected_p0 = sum(entries_per_layer[:21])
        assert len(new_addr[0]) == expected_p0 == 59
        assert len(new_addr[1]) == total - expected_p0 == 67

    def test_partition1_starts_at_first_entry_of_layer_21(self):
        _, num_layers, offsets = _dsv4_layout()
        total = offsets[-1]
        db, _ = _make_db([21, 22], num_layers, offsets, total)
        addrs = [[10000 + 10 * i for i in range(total)]]

        _, new_addr, _ = db.decode_adaptor_prefill_pp(
            [BASE_KEY], addrs, [[16] * total], kv_cache_group_id=0
        )

        layer21_start = 10000 + 10 * offsets[21]
        assert new_addr[1][0] == layer21_start

    def test_no_entries_lost_and_sizes_consistent(self):
        _, num_layers, offsets = _dsv4_layout()
        total = offsets[-1]
        db, _ = _make_db([21, 22], num_layers, offsets, total)

        _, new_addr, new_size = db.decode_adaptor_prefill_pp(
            [BASE_KEY],
            [[10000 + 10 * i for i in range(total)]],
            [[16] * total],
            kv_cache_group_id=0,
        )

        assert len(new_addr[0]) + len(new_addr[1]) == total
        assert len(new_size[0]) == len(new_addr[0])
        assert len(new_size[1]) == len(new_addr[1])

    def test_keys_tagged_with_pp_rank(self):
        _, num_layers, offsets = _dsv4_layout()
        total = offsets[-1]
        db, _ = _make_db([21, 22], num_layers, offsets, total)

        new_key, _, _ = db.decode_adaptor_prefill_pp(
            [BASE_KEY], [[0] * total], [[16] * total], kv_cache_group_id=0
        )

        assert len(new_key) == 2
        assert "@pp_rank:0" in new_key[0]
        assert "@pp_rank:1" in new_key[1]

    def test_mtp_tail_layer_folded_into_last_partition(self):
        # partitions sum to 43 (num_hidden_layers); the 44th (MTP) layer must
        # still be covered by the last partition.
        entries_per_layer, num_layers, offsets = _dsv4_layout()
        total = offsets[-1]
        db, _ = _make_db([21, 22], num_layers, offsets, total)

        _, new_addr, _ = db.decode_adaptor_prefill_pp(
            [BASE_KEY], [[10000 + 10 * i for i in range(total)]], [[16] * total],
            kv_cache_group_id=0,
        )

        mtp_first = 10000 + 10 * offsets[43]
        assert mtp_first in new_addr[1]

    def test_short_addr_list_is_clamped(self):
        # addr_list shorter than offsets table (partial register) must not
        # raise; bounds are clamped to len(addr_list).
        _, num_layers, offsets = _dsv4_layout()
        db, _ = _make_db([21, 22], num_layers, offsets, 126)

        _, new_addr, _ = db.decode_adaptor_prefill_pp(
            [BASE_KEY], [[10000 + 10 * i for i in range(30)]], [[16] * 30],
            kv_cache_group_id=0,
        )

        assert len(new_addr[0]) + len(new_addr[1]) == 30


class TestLegacyUniformSplit:
    """Homogeneous layout without offsets table keeps old behavior."""

    def test_uniform_split_without_offsets(self):
        db, _ = _make_db([2, 2], 8, offsets=None, total_entries=16)

        _, new_addr, _ = db.decode_adaptor_prefill_pp(
            [BASE_KEY], [[2000 + i for i in range(16)]], [[16] * 16],
            kv_cache_group_id=0,
        )

        # 16 entries / 8 layers -> caches_per_layer=2 -> 4 + 12
        assert len(new_addr[0]) == 4
        assert len(new_addr[1]) == 12

    def test_single_partition_passthrough(self):
        db, _ = _make_db([8], 8, offsets=None, total_entries=16)

        key, addr, size = db.decode_adaptor_prefill_pp(
            [BASE_KEY], [[1, 2, 3]], [[16, 16, 16]], kv_cache_group_id=0
        )

        assert key == [BASE_KEY]
        assert addr == [[1, 2, 3]]
        assert size == [[16, 16, 16]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
