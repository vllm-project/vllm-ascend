# SPDX-License-Identifier: Apache-2.0
"""Exercise the D2RH PP transfer plan with a mocked transport."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda *args: {"pp_size": 2})
            )
        )
        self.worker._get_hop1_layer_pairs = lambda *args: d2rh.D2RHThread._get_hop1_layer_pairs(self.worker, *args)
        self.spec: dict[str, object] = {"layer_names": ["layer.0", "layer.1"], "kv_cache_spec_type": "AttentionSpec"}

    def test_pp_stage_filter_and_renumber(self):
        self.assertEqual(self.worker._get_hop1_layer_pairs(self.spec, [0, 1], {"layer.1": 7}), [(1, 7)])

    def test_indexer_and_draft_are_matched_by_name(self):
        spec = {"layer_names": ["model.layers.22.indexer", "mtp.layers.43"]}
        self.assertEqual(
            self.worker._get_hop1_layer_pairs(spec, [108, 43], {"mtp.layers.43": 44, "model.layers.22.indexer": 110}),
            [(108, 110), (43, 44)],
        )

    def test_pp_missing_names_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "requires cache layer names"):
            self.worker._get_hop1_layer_pairs(self.spec, [0, 1], {})

    def test_pp1_legacy_identity(self):
        self.worker.vllm_config.kv_transfer_config.get_from_extra_config = lambda *args: {"pp_size": 1}
        self.assertEqual(self.worker._get_hop1_layer_pairs(self.spec, [0, 1], {}), [(0, 0), (1, 1)])

    def test_pp1_missing_layer_rejected(self):
        self.worker.vllm_config.kv_transfer_config.get_from_extra_config = lambda *args: {"pp_size": 1}
        with self.assertRaisesRegex(RuntimeError, "does not contain layer"):
            self.worker._get_hop1_layer_pairs(self.spec, [0, 1], {"layer.0": 0})

    def test_shared_metadata_layer_is_transferred_once(self):
        spec = {"layer_names": ["indexer", "attention", "next"]}
        self.assertEqual(
            self.worker._get_hop1_layer_pairs(spec, [0, 0, 1], {"indexer": 7, "attention": 7, "next": 8}),
            [(0, 7), (1, 8)],
        )

    def test_legacy_duplicate_metadata_layer_is_transferred_once(self):
        self.worker.vllm_config.kv_transfer_config.get_from_extra_config = lambda *args: {"pp_size": 1}
        self.assertEqual(self.worker._get_hop1_layer_pairs({}, [1, 1, 0], {}), [(1, 1), (0, 0)])

    def test_misaligned_names_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "misaligned"):
            self.worker._get_hop1_layer_pairs(self.spec, [0], {"layer.0": 0})

    def prepare_transfer(self, missing=False, host_hit=False):
        w = self.worker
        w.remote_local_block_map = {"r": {(0, 1, 0): 3}}
        w.log_full_block_map = False
        w.remote_metadata_lock = nullcontext()
        w.kv_group2layeridx = {0: (self.spec, [0, 1])}
        w.cpu_kv_caches_base_addr = [[1000], [2000]]
        w.cpu_block_len_per_addr = [[8], [8]]
        w.cpu_block_stride_per_addr = [[16], [16]]
        w.cpu_block_size_scale = [[1], [1]]
        w.kv_caches_base_addr = {"e": {10: [[3000]], 11: [[4000]]}}
        w.remote_kv_group2layeridx = {
            "e": {
                10: {0: ({"layer_names": ["layer.0"]}, [0])},
                11: {0: ({"layer_names": ["layer.0" if missing else "layer.1"]}, [0])},
            }
        }
        w.remote_block_size_scale = {"e": {10: [[1]], 11: [[1]]}}
        w.remote_block_stride_per_addr = {"e": {10: [[32]], 11: [[64]]}}
        w.remote_te_port = {"e": {10: 1010, 11: 1011}}
        w.block_size = 32
        w.group_compress_ratios = {0: 1}
        self.calls: list[tuple[str, list[int], list[int], list[int]]] = []
        self.releases: list[tuple[object, ...]] = []

        def transfer(session: str, local_addrs: list[int], remote_addrs: list[int], lengths: list[int]) -> int:
            self.calls.append((session, local_addrs, remote_addrs, lengths))
            return 0

        w.engine = SimpleNamespace(batch_transfer_sync_read=transfer)
        w._send_done_recv_signal = lambda *args: self.releases.append(args)
        transport_patch = patch.multiple(
            d2rh,
            resolve_remote_host_for_handshake_port=lambda *args: ("host", "e"),
            _get_group_pull_field=lambda p, k: p[k],
            _is_sliding_group_spec=lambda _: False,
            base_group_concurrent_contiguous=lambda remote, local: ([[x] for x in remote], [[x] for x in local]),
            split_if_not_byte_contiguous=lambda remote, local, **kwargs: (remote, local),
        )
        transport_patch.start()
        self.addCleanup(transport_patch.stop)
        return {
            "request_id": "r",
            "remote_request_id": "r",
            "remote_engine_id": "e",
            "remote_host": "host",
            "remote_port": 10,
            "remote_block_ids": ([1],),
            "remote_handshake_ports": [10, 11],
            "cache_hits": {(0, 1, 0)} if host_hit else set(),
            "group_pulls_by_port": [
                [{"group_id": 0, "remote_tp_offset": 0, "num_group_pulls": 1, "prefill_pp_rank": rank}]
                for rank in (0, 1)
            ],
        }

    def test_full_hop1_uses_remote_stage_stride(self):
        d2rh.D2RHThread._transfer_kv_cache_all_groups(self.worker, self.prepare_transfer())
        self.assertEqual(self.calls, [("host:1010", [1048], [3032], [8]), ("host:1011", [2048], [4064], [8])])
        self.assertEqual(len(self.releases), 2)

    def test_full_hop1_selects_only_group_owned_components(self):
        req = self.prepare_transfer()
        w = self.worker
        self.spec["layer_cache_indices"] = {"layer.0": [1], "layer.1": [1]}
        w.cpu_kv_caches_base_addr = [[1000, 1100], [2000, 2100]]
        w.cpu_block_len_per_addr = [[8, 8], [8, 8]]
        w.cpu_block_stride_per_addr = [[16, 16], [16, 16]]
        w.cpu_block_size_scale = [[1, 1], [1, 1]]
        for port, name, ptr, stride in [(10, "layer.0", 3000, 32), (11, "layer.1", 4000, 64)]:
            w.kv_caches_base_addr["e"][port] = [[ptr, ptr + 100]]
            w.remote_block_size_scale["e"][port] = [[1, 1]]
            w.remote_block_stride_per_addr["e"][port] = [[stride, stride]]
            w.remote_kv_group2layeridx["e"][port][0][0]["layer_cache_indices"] = {name: [1]}
        d2rh.D2RHThread._transfer_kv_cache_all_groups(w, req)
        self.assertEqual(self.calls, [("host:1010", [1148], [3132], [8]), ("host:1011", [2148], [4164], [8])])

    def test_incompatible_component_metadata_rejected_before_transfer(self):
        req = self.prepare_transfer()
        self.spec["layer_cache_indices"] = {"layer.0": [0], "layer.1": [0]}
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            d2rh.D2RHThread._transfer_kv_cache_all_groups(self.worker, req)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.releases, [])

    def test_full_hop1_shared_layer_keeps_all_distinct_slots(self):
        req = self.prepare_transfer()
        w = self.worker
        self.spec["layer_names"] = ["layer.0", "indexer.0", "layer.1"]
        self.spec["layer_cache_indices"] = {"layer.0": [0], "indexer.0": [1], "layer.1": [0]}
        w.kv_group2layeridx[0] = (self.spec, [0, 0, 1])
        w.cpu_kv_caches_base_addr[0] = [1000, 1100]
        w.cpu_block_len_per_addr[0] = [8, 8]
        w.cpu_block_stride_per_addr[0] = [16, 16]
        w.cpu_block_size_scale[0] = [1, 1]
        w.kv_caches_base_addr["e"][10] = [[3000, 3100]]
        w.remote_block_stride_per_addr["e"][10] = [[32, 32]]
        w.remote_block_size_scale["e"][10] = [[1, 1]]
        w.remote_kv_group2layeridx["e"][10][0] = (
            {"layer_names": ["layer.0", "indexer.0"], "layer_cache_indices": {"layer.0": [0], "indexer.0": [1]}},
            [0, 0],
        )
        w.remote_kv_group2layeridx["e"][11][0][0]["layer_cache_indices"] = {"layer.1": [0]}
        d2rh.D2RHThread._transfer_kv_cache_all_groups(w, req)
        self.assertEqual(
            self.calls,
            [("host:1010", [1048, 1148], [3032, 3132], [8, 8]), ("host:1011", [2048], [4064], [8])],
        )

    def test_missing_stage_rejected_before_transfer(self):
        req = self.prepare_transfer(missing=True)
        with self.assertRaisesRegex(RuntimeError, "Incomplete D2RH PP layer coverage"):
            d2rh.D2RHThread._transfer_kv_cache_all_groups(self.worker, req)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.releases, [])

    def test_hop1_packed_view_covers_aliased_component(self):
        req = self.prepare_transfer()
        w = self.worker
        self.spec["layer_cache_indices"] = {"layer.0": [0, 1], "layer.1": [0]}
        w.cpu_kv_caches_base_addr[0] = [1000, 1000]
        w.cpu_block_len_per_addr[0] = [8, 16]
        w.cpu_block_stride_per_addr[0] = [16, 16]
        w.cpu_block_size_scale[0] = [1, 1]
        w.kv_caches_base_addr["e"][10] = [[3000, 3000]]
        w.remote_block_size_scale["e"][10] = [[1, 1]]
        w.remote_block_stride_per_addr["e"][10] = [[32, 32]]
        w.remote_kv_group2layeridx["e"][10][0][0]["layer_cache_indices"] = {"layer.0": [0, 1]}
        w.remote_kv_group2layeridx["e"][11][0][0]["layer_cache_indices"] = {"layer.1": [0]}
        d2rh.D2RHThread._transfer_kv_cache_all_groups(w, req)
        self.assertEqual(self.calls, [("host:1010", [1048], [3032], [16]), ("host:1011", [2048], [4064], [8])])

    def test_host_hit_skips_both_stage_reads(self):
        d2rh.D2RHThread._transfer_kv_cache_all_groups(self.worker, self.prepare_transfer(host_hit=True))
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.releases), 2)
