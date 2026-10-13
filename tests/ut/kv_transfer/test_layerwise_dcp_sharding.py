# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Unit tests for DCP shard handling in the layerwise KV pool.

Covers the three adaptations introduced for decode-context parallel:

- ``KVPoolWorker._is_layerwise_save_leader``: exactly one rank per
  (pcp, dcp, head_or_tp) group must save/allocate; the plain
  ``tp_rank % put_step`` dedup drops every non-zero DCP shard.
- ``KVPoolWorker._global_group_alloc_size``: with DCP>1 and put_step>1
  the shared region must cover all shards, using a per-shard stride
  aligned up to the GVA hugepage size (hybrid DSA groups have
  non-uniform per-layer bytes).
"""

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

GVA_ALIGN = 2 * 1024 * 1024


def _bare_worker(**attrs) -> KVPoolWorker:
    """Construct a KVPoolWorker without running __init__.

    ``layout_dcp_size`` is the sharding factor for the shared GVA region. It is
    resolved from the peer topology in ``_init_kv_transfer_config`` and equals
    the local ``dcp_size`` for every symmetric topology, so these stubs default
    it to that and tests only override it for asymmetric (decode-only) cases.
    """
    attrs.setdefault("layout_dcp_size", attrs.get("dcp_size", 1))
    worker = object.__new__(KVPoolWorker)
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


class TestSaveLeader:
    def _mla_worker(self, tp_rank: int, dcp_size: int = 2) -> KVPoolWorker:
        # MLA: num_kv_head=1 < tp -> put_step == tp_size, head_or_tp_rank 0.
        return _bare_worker(
            tp_rank=tp_rank,
            tp_size=4,
            put_step=4,
            dcp_size=dcp_size,
            dcp_rank=tp_rank % dcp_size,
        )

    def test_dcp1_keeps_put_step_semantics(self):
        worker = _bare_worker(tp_rank=2, put_step=4, dcp_size=1, dcp_rank=0)
        assert worker._is_layerwise_save_leader() is False
        worker = _bare_worker(tp_rank=0, put_step=4, dcp_size=1, dcp_rank=0)
        assert worker._is_layerwise_save_leader() is True

    def test_mla_tp4_dcp2_leaders_are_shard_minima(self):
        # ranks: 0(d0) 1(d1) 2(d0) 3(d1); head=0 for all (put_step=4).
        # Shard 0 peers {0,2} -> leader 0; shard 1 peers {1,3} -> leader 1.
        leaders = [r for r in range(4) if self._mla_worker(r)._is_layerwise_save_leader()]
        assert leaders == [0, 1]

    def test_gqa_put_step1_all_ranks_lead(self):
        # GQA: put_step==1 -> every (dcp, head) combo is a singleton group.
        worker = _bare_worker(tp_rank=3, tp_size=4, put_step=1, dcp_size=2, dcp_rank=1)
        assert worker._is_layerwise_save_leader() is True

    def test_gqa_tp4_dcp2_put_step2(self):
        # heads: r0,r1 -> h0; r2,r3 -> h1. dcp: r0,d0 r1,d1 r2,d0 r3,d1.
        # (d0,h0)={0}, (d1,h0)={1}, (d0,h1)={2}, (d1,h1)={3}: all lead.
        for r in range(4):
            worker = _bare_worker(tp_rank=r, tp_size=4, put_step=2, dcp_size=2, dcp_rank=r % 2)
            assert worker._is_layerwise_save_leader() is True, f"rank {r}"


class TestGlobalGroupAllocSize:
    def _sized_worker(self, block_len: list[int], put_step: int, dcp_size: int) -> KVPoolWorker:
        return _bare_worker(
            group_block_len={0: block_len},
            group_num_layers={0: len(block_len)},
            num_layers=len(block_len),
            total_layers=len(block_len),
            page_size_bytes=4096,
            put_step=put_step,
            pcp_size=1,
            dcp_size=dcp_size,
        )

    def test_dcp1_unchanged(self):
        # Uniform 4 layers x 32KB -> plain per_layer * n_global.
        worker = self._sized_worker([32768] * 4, put_step=4, dcp_size=1)
        assert worker._global_group_alloc_size(0) == 32768 * 4

    def test_put_step1_no_shard_inflation(self):
        # GQA/MHA: every rank owns a distinct region key -> no x cp_scale.
        worker = self._sized_worker([32768] * 4, put_step=1, dcp_size=8)
        assert worker._global_group_alloc_size(0) == 32768 * 4

    def test_dcp2_shard_major_region(self):
        worker = self._sized_worker([32768] * 4, put_step=4, dcp_size=2)
        # sum=128KB -> stride aligned to 2MB; region = stride * cp_scale.
        expected = GVA_ALIGN * 2
        assert worker._global_group_alloc_size(0) == expected

    def test_nonuniform_layers_align_stride(self):
        # Hybrid DSA layout with an odd-sized trailing entry: the group byte
        # total is not 16B-aligned, so a raw sum-based shard stride would
        # produce misaligned GVA addresses on non-zero shards.
        block_len = [131072, 16384, 262144, 131072, 16384, 131072, 65534]
        total = sum(block_len)
        assert total % 16 != 0, "test premise: unaligned group total"
        worker = self._sized_worker(block_len, put_step=8, dcp_size=8)
        region = worker._global_group_alloc_size(0)
        stride = region // 8
        assert stride % GVA_ALIGN == 0, "shard stride must be GVA-aligned"
        assert stride >= total, "aligned stride must still cover the shard"


class TestDecodeOnlyDcpLayout:
    """Decode-only DCP: producer dcp==1, consumer dcp==N, pcp==1 on both sides.

    Both sides share ONE GVA region, so the un-sharded producer must adopt the
    peer-sharded layout factor instead of allocating a single-shard region that
    the sharded consumer would read past.
    """

    def _producer(self, block_len, put_step, layout_dcp_size):
        return _bare_worker(
            group_block_len={0: block_len},
            group_num_layers={0: len(block_len)},
            num_layers=len(block_len),
            total_layers=len(block_len),
            page_size_bytes=4096,
            put_step=put_step,
            pcp_size=1,
            dcp_size=1,  # producer side has no DCP
            dcp_rank=0,
            layout_dcp_size=layout_dcp_size,  # resolved from decode_dcp_size
        )

    @staticmethod
    def _consumer(block_len, put_step, dcp_size):
        return _bare_worker(
            group_block_len={0: block_len},
            group_num_layers={0: len(block_len)},
            num_layers=len(block_len),
            total_layers=len(block_len),
            page_size_bytes=4096,
            put_step=put_step,
            pcp_size=1,
            dcp_size=dcp_size,
            dcp_rank=0,
            layout_dcp_size=dcp_size,
        )

    def test_producer_region_covers_all_shards(self):
        # A dcp=1 producer paired with a dcp=8 consumer must size the region
        # for 8 shards, not 1.
        prod = self._producer([32768] * 4, put_step=4, layout_dcp_size=8)
        assert prod._global_group_alloc_size(0) == GVA_ALIGN * 8

    def test_producer_region_equals_sharded_consumer(self):
        # The whole point: both sides compute the SAME region size.
        block_len = [32768] * 4
        prod = self._producer(block_len, put_step=4, layout_dcp_size=2)
        cons = self._consumer(block_len, put_step=4, dcp_size=2)
        assert prod._global_group_alloc_size(0) == cons._global_group_alloc_size(0)

    def test_symmetric_layout_factor_unchanged(self):
        # Symmetric deployments resolve layout_dcp_size == dcp_size, so the
        # region math is bit-identical to the pre-adaptation behaviour.
        block_len = [32768] * 4
        sym = self._consumer(block_len, put_step=4, dcp_size=8)
        assert sym._global_group_alloc_size(0) == GVA_ALIGN * 8


class TestDecodeOnlyDcpMetadata:
    """Peer-topology helpers that drive the shared layout factor."""

    @staticmethod
    def _helpers():
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            infer_decode_only_dcp,
            infer_peer_cp_sizes,
            resolve_layout_dcp_size,
        )

        return infer_peer_cp_sizes, infer_decode_only_dcp, resolve_layout_dcp_size

    def test_producer_side_resolves_peer_sharded_layout(self):
        peer_fn, only_fn, layout_fn = self._helpers()
        peer = peer_fn("kv_producer", {"decode_dcp_size": 8, "decode_pcp_size": 1}, 1, 1)
        assert peer == (8, 1)
        assert only_fn(1, 1, *peer) is True
        assert layout_fn(True, 1, peer[0]) == 8

    def test_consumer_side_resolves_same_layout(self):
        peer_fn, only_fn, layout_fn = self._helpers()
        peer = peer_fn("kv_consumer", {"prefill_dcp_size": 1, "prefill_pcp_size": 1}, 8, 1)
        assert peer == (1, 1)
        assert only_fn(8, 1, *peer) is True
        assert layout_fn(True, 8, peer[0]) == 8

    def test_symmetric_topology_keeps_local_dcp(self):
        peer_fn, only_fn, layout_fn = self._helpers()
        peer = peer_fn("kv_producer", {}, 2, 1)
        assert peer == (2, 1)
        assert only_fn(2, 1, *peer) is False
        assert layout_fn(False, 2, peer[0]) == 2

    def test_other_asymmetric_shapes_are_not_supported(self):
        _, only_fn, _ = self._helpers()
        # both sides sharded, unequal sizes -> not decode-only
        assert only_fn(2, 1, 4, 1) is False
        # any pcp > 1 -> not decode-only
        assert only_fn(1, 2, 8, 2) is False
        assert only_fn(1, 1, 8, 2) is False


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
