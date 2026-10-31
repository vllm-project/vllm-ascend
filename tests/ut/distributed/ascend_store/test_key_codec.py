# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source identity must survive CPD serialization without reader config."""

import json
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity


class TestPoolKeyIdentity(unittest.TestCase):
    def test_wire_round_trip_preserves_existing_key_formats(self):
        cases = (
            (PoolKeyIdentity("memcache-v1", "model", 1, 1), 0, 0, "model@hash@3"),
            (PoolKeyIdentity("memcache-v1", "model", 2, 2), 1, 1, "model@1@pp1@hash@3"),
            (PoolKeyIdentity("mooncake-block-v1", "model", 1, 1), 0, 0, "model@hash@3"),
            (
                PoolKeyIdentity("mooncake-block-v1", "model", 1, 2, namespace="source-layout"),
                0,
                1,
                "model@source-layout@pp_rank:1@hash@3",
            ),
            (
                PoolKeyIdentity("mooncake-hybrid-v1", "model", 2, 2, layout_id="layout", group_block_sizes=(16, 32)),
                1,
                1,
                "model@mooncake_hybrid_v1:layout@pp_rank:1@group:1@block:32@hash@3",
            ),
            (
                PoolKeyIdentity("mooncake-hybrid-v1", "model", 2, 1, layout_id="layout", group_block_sizes=(16, 32)),
                1,
                0,
                "model@mooncake_hybrid_v1:layout@group:1@block:32@hash@3",
            ),
        )
        for identity, group, stage, expected in cases:
            with self.subTest(identity=identity):
                received = PoolKeyIdentity.from_wire(json.loads(json.dumps(identity.to_wire())))
                self.assertEqual(received, identity)
                self.assertEqual(received.bind().make_full_key(group, "hash", 3, stage), expected)

    def test_reader_uses_source_identity_without_rehashing(self):
        source = PoolKeyIdentity(
            "mooncake-hybrid-v1",
            "source-model",
            2,
            2,
            namespace="source-topology",
            layout_id="source-layout",
            group_block_sizes=(16, 32),
        )
        reader_local = PoolKeyIdentity(
            "mooncake-hybrid-v1",
            "source-model",
            1,
            1,
            layout_id="reader-layout",
            group_block_sizes=(64,),
        )
        wire = json.loads(json.dumps(source.to_wire()))
        with (
            patch.object(mooncake_layerwise, "hybrid_layout_id", side_effect=AssertionError("must not rehash")),
            patch.object(
                mooncake_layerwise, "layerwise_topology_namespace", side_effect=AssertionError("must not rehash")
            ),
        ):
            key = PoolKeyIdentity.from_wire(wire).bind().make_full_key(1, "hash", 3, 1)
        self.assertEqual(key, "source-model@mooncake_hybrid_v1:source-layout@pp_rank:1@group:1@block:32@hash@3")
        self.assertNotEqual(key, reader_local.bind().make_full_key(0, "hash", 0, 0))

    def test_partial_key_and_pd_tail_id_have_distinct_protocols(self):
        identity = PoolKeyIdentity("memcache-v1", "model", 2, 2)
        self.assertEqual(identity.partial_key("req", 1, 2, 35, 3, 1), "model@partial@req@1@2@35@pp1@3")
        tail = "pd-tail-v1:attempt:35"
        self.assertEqual(identity.bind().make_full_key(1, tail, 3, 1), f"model@1@pp1@{tail}@3")
        with self.assertRaisesRegex(ValueError, "only supported by MemCache"):
            PoolKeyIdentity("mooncake-block-v1", "model", 1, 1).partial_key("req", 0, 0, 3, 0)

    def test_rebound_lookup_covers_all_source_stages_and_ranks(self):
        builder = PoolKeyIdentity.from_wire(PoolKeyIdentity("memcache-v1", "model", 2, 2).to_wire()).bind()
        self.assertEqual(
            builder.make_hit_check_keys(1, "hash", 2),
            [
                "model@1@pp0@hash@0",
                "model@1@pp0@hash@1",
                "model@1@pp1@hash@0",
                "model@1@pp1@hash@1",
            ],
        )

    def test_wire_is_detached_from_immutable_identity(self):
        identity = PoolKeyIdentity("mooncake-hybrid-v1", "model", 2, 1, layout_id="layout", group_block_sizes=(16, 32))
        wire = identity.to_wire()
        restored = PoolKeyIdentity.from_wire(wire)
        wire["group_block_sizes"][0] = 128
        self.assertEqual(identity.group_block_sizes, (16, 32))
        self.assertEqual(restored, identity)
        with self.assertRaises(FrozenInstanceError):
            identity.pp_size = 3

    def test_malformed_or_unknown_identity_is_rejected(self):
        base = PoolKeyIdentity("memcache-v1", "model", 1, 1).to_wire()
        for change in (
            {"format": "future-v99"},
            {"num_groups": 0},
            {"pp_size": True},
            {"model_name": ""},
            {"namespace": "not-supported"},
            {"extra": 1},
            {"group_block_sizes": "16"},
            {"group_block_sizes": [16]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PoolKeyIdentity.from_wire(base | change)
        with self.assertRaises(ValueError):
            PoolKeyIdentity.from_wire({"format": "memcache-v1"})
        with self.assertRaises(ValueError):
            PoolKeyIdentity.from_wire([])
        for kwargs in (
            {"format": "mooncake-block-v1", "pp_size": 2},
            {"format": "mooncake-hybrid-v1", "layout_id": "layout", "group_block_sizes": [16, 32]},
            {"format": "mooncake-hybrid-v1", "layout_id": "layout", "group_block_sizes": [False]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PoolKeyIdentity.from_wire(base | kwargs)

    def test_invalid_source_coordinates_are_rejected(self):
        builder = PoolKeyIdentity("memcache-v1", "model", 2, 2).bind()
        for group, head, stage in ((2, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 2), (True, 0, 0)):
            with self.subTest(group=group, head=head, stage=stage), self.assertRaises(ValueError):
                builder.make_full_key(group, "hash", head, stage)
        with self.assertRaises(ValueError):
            builder.make_full_key(0, "", 0, 0)
