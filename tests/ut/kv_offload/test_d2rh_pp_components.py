# SPDX-License-Identifier: Apache-2.0
"""Regression for PP-dependent hybrid KV alias layouts."""

import unittest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_d2rh_connector import (
    resolve_group_cache_slot_pairs as resolve,
)


class TestPPCacheComponents(unittest.TestCase):
    def test_swa_does_not_copy_compressed_attention_or_state(self):
        spec = {"layer_names": ["layer.22.swa"], "layer_cache_indices": {"layer.22.swa": [4]}}
        self.assertEqual(resolve(spec, [22], 22, {"layer.22.swa": [4]}, 7, 7), [(4, 4)])

    def test_peer_component_order_can_differ(self):
        spec = {"layer_names": ["layer.22.swa"], "layer_cache_indices": {"layer.22.swa": [4]}}
        self.assertEqual(resolve(spec, [22], 22, {"layer.22.swa": [1]}, 7, 7), [(4, 1)])

    def test_multiple_entries_in_one_group(self):
        spec = {
            "layer_names": ["indexer", "attention"],
            "layer_cache_indices": {"indexer": [0, 1, 2], "attention": [3]},
        }
        self.assertEqual(
            resolve(spec, [22, 22], 22, {"indexer": [2, 3, 4], "attention": [0]}, 7, 7),
            [(0, 2), (1, 3), (2, 4), (3, 0)],
        )

    def test_legacy_single_component(self):
        self.assertEqual(resolve({}, [0], 0, {}, 1, 1), [(0, 0)])

    def test_missing_peer_component_fails_closed(self):
        spec = {"layer_names": ["swa"], "layer_cache_indices": {"swa": [4]}}
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            resolve(spec, [22], 22, {}, 7, 7)

    def test_out_of_range_fails_closed(self):
        spec = {"layer_names": ["swa"], "layer_cache_indices": {"swa": [4]}}
        with self.assertRaisesRegex(RuntimeError, "Out-of-range"):
            resolve(spec, [22], 22, {"swa": [99]}, 7, 7)

    def test_pp_boundary_alias_corruption_reproducer(self):
        # Observed layout: on D, layer20.SWA aliases layer22.attention.
        # On P's second stage, layer22.attention aliases layer22.SWA instead.
        # Group2 owns SWA only. P cache slot3 contains layer22's SWA bytes at
        # group2 block IDs, NOT layer20's bytes expected at D's aliased address.
        source = {3: "layer22_swa", 4: "layer22_swa"}
        destination_addresses = {3: "D_layer20_swa", 4: "D_layer22_swa"}
        old = {"D_layer20_swa": "layer20_swa"}
        for slot in [3, 4]:
            old[destination_addresses[slot]] = source[slot]
        self.assertEqual(old["D_layer20_swa"], "layer22_swa")  # reproduces old corruption
        fixed = {"D_layer20_swa": "layer20_swa"}
        spec = {"layer_names": ["layer22.swa"], "layer_cache_indices": {"layer22.swa": [4]}}
        for local, remote in resolve(spec, [22], 22, {"layer22.swa": [4]}, 7, 7):
            fixed[destination_addresses[local]] = source[remote]
        self.assertEqual(fixed["D_layer20_swa"], "layer20_swa")
        self.assertEqual(fixed["D_layer22_swa"], "layer22_swa")
