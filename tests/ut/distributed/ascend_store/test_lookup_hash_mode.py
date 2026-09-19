# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    HBMCachedBlockHashList,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupHashMode, get_block_hashes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler, LookupKeyClient
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
# isort: on


def _config(lookup_hash_mode: str | None = None) -> MagicMock:
    config = MagicMock()
    extra_config = {"backend": "mooncake"}
    if lookup_hash_mode is not None:
        extra_config["lookup_hash_mode"] = lookup_hash_mode
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.kv_connector_extra_config = extra_config
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = 16
    config.cache_config.hash_block_size = 16
    config.model_config.model = "org/llama-7b"
    config.model_config.max_model_len = 1024
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    config.additional_config = {"enable_kvpp": False}
    config.kv_events_config = None
    return config


def _start_patch(test: unittest.TestCase, target: str, **kwargs):
    patcher = patch(target, **kwargs)
    mocked = patcher.start()
    test.addCleanup(patcher.stop)
    return mocked


def _worker(test: unittest.TestCase) -> KVPoolWorker:
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
    _start_patch(test, f"{module}.get_tensor_model_parallel_rank", return_value=0)
    _start_patch(test, f"{module}.get_tensor_model_parallel_world_size", return_value=1)
    pcp_group = _start_patch(test, f"{module}.get_pcp_group")
    pcp_group.return_value.world_size = 1
    pcp_group.return_value.rank_in_group = 0
    _start_patch(test, f"{module}.get_decode_context_model_parallel_world_size", return_value=1)
    _start_patch(test, f"{module}.get_decode_context_model_parallel_rank", return_value=0)
    backend_import = _start_patch(test, f"{module}.importlib")
    backend_import.import_module.return_value = MagicMock()
    return KVPoolWorker(_config(), use_layerwise=False)


def _hashes(num_blocks: int) -> list[bytes]:
    return [bytes([index % 251]) * 32 for index in range(num_blocks)]


class _FakePrefixManager:
    @classmethod
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec, **kwargs):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        for block_hash in list(block_hashes)[: max_length // kv_cache_spec.block_size]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        return computed, len(computed[0]) * kv_cache_spec.block_size


class TestLookupHashMode(unittest.TestCase):
    def test_full_and_suffix_preserve_lookup_semantics_at_hbm_boundaries(self):
        worker = _worker(self)
        worker.cache_coordinator = None
        worker.token_database.block_size = [64]
        worker.m_store.exists.return_value = [1]
        request = MagicMock(
            prompt_token_ids=list(range(96)),
            num_tokens=96,
            request_id="r1",
            block_hashes=["h0", "h1", "h2", "h3", "h4", "h5"],
        )
        scheduler_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"

        for hbm_hit_tokens, expected_new_tokens in ((0, 64), (32, 32)):
            results = []
            for configured_mode, mode in ((None, LookupHashMode.FULL), ("suffix", LookupHashMode.SUFFIX)):
                with self.subTest(mode=mode.value, hbm_hit_tokens=hbm_hit_tokens):
                    with patch(f"{scheduler_module}.importlib") as backend_import:
                        backend_import.import_module.return_value = MagicMock()
                        scheduler = KVPoolScheduler(_config(configured_mode), use_layerwise=False)
                    scheduler.client = MagicMock()
                    scheduler.client.lookup.side_effect = worker.lookup_scheduler
                    worker.m_store.reset_mock()

                    results.append(scheduler.get_num_new_matched_tokens(request, hbm_hit_tokens))
                    omitted_hashes = hbm_hit_tokens // 16 if mode is LookupHashMode.SUFFIX else 0
                    scheduler.client.lookup.assert_called_once_with(
                        96,
                        request.block_hashes[omitted_hashes:],
                        [0],
                        hbm_hit_tokens=hbm_hit_tokens,
                        lookup_hash_mode=mode,
                    )
                    keys = worker.m_store.exists.call_args.args[0]
                    self.assertEqual(len(keys), 1)
                    self.assertTrue(keys[0].endswith("@h3"))
                    load_spec = scheduler.load_specs[request.request_id]
                    self.assertEqual(load_spec.kvpool_cached_tokens, 64)
                    self.assertEqual(load_spec.vllm_cached_tokens, hbm_hit_tokens)

            self.assertEqual(results, [(expected_new_tokens, False)] * 2)

        for configured_mode in (None, "suffix"):
            with (
                self.subTest(configured_mode=configured_mode, hbm_hit_tokens=96),
                patch(f"{scheduler_module}.importlib") as backend_import,
                patch(f"{scheduler_module}.LookupKeyClient") as client_cls,
            ):
                backend_import.import_module.return_value = MagicMock()
                scheduler = KVPoolScheduler(_config(configured_mode), use_layerwise=False)
                self.assertEqual(scheduler.get_num_new_matched_tokens(request, 96), (0, False))
                client_cls.assert_not_called()

        worker._lookup_with_coordinator = MagicMock(return_value=64)
        for mode, payload in ((LookupHashMode.FULL, request.block_hashes), (LookupHashMode.SUFFIX, [])):
            with self.subTest(mode=mode.value, direct_worker_hbm_hit_tokens=96):
                worker.m_store.reset_mock()
                worker._lookup_with_coordinator.reset_mock()
                result = worker.lookup_scheduler(96, payload, [0], hbm_hit_tokens=96, lookup_hash_mode=mode)
                self.assertEqual(result, 96)
                worker._lookup_with_coordinator.assert_not_called()
                worker.m_store.exists.assert_not_called()

        with patch(f"{scheduler_module}.importlib") as backend_import:
            backend_import.import_module.return_value = MagicMock()
            with self.assertRaisesRegex(ValueError, "lookup_hash_mode must be one of: full, suffix"):
                KVPoolScheduler(_config("invalid"), use_layerwise=False)

    def test_lookup_protocol_transmits_mode_and_hbm_offset(self):
        scheduler_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
        with (
            patch(f"{scheduler_module}.make_zmq_socket") as make_socket,
            patch(f"{scheduler_module}.zmq"),
            patch(f"{scheduler_module}.MsgpackEncoder") as encoder_cls,
        ):
            socket = make_socket.return_value
            socket.recv.return_value = (32).to_bytes(4, "big")
            encoder_cls.return_value.encode.side_effect = [[b"hashes"], [b"groups"], [b"mode"]]
            client = LookupKeyClient(_config())

            result = client.lookup(64, [b"\xaa\xbb"], hbm_hit_tokens=16, lookup_hash_mode=LookupHashMode.SUFFIX)
            self.assertEqual(result, 32)
            self.assertEqual(encoder_cls.return_value.encode.call_args_list[2].args[0], "suffix")
            socket.send_multipart.assert_called_once_with(
                [(64).to_bytes(4, "big"), b"groups", (16).to_bytes(4, "big"), b"mode", b"hashes"], copy=False
            )

        connector_module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
        with (
            patch(f"{connector_module}.threading.Thread") as thread_cls,
            patch(f"{connector_module}.make_zmq_socket") as make_socket,
            patch(f"{connector_module}.MsgpackDecoder") as decoder_cls,
        ):
            pool_worker = MagicMock()
            pool_worker.lookup_scheduler.return_value = 48
            socket = make_socket.return_value
            decoder_cls.return_value.decode.side_effect = [[0, 1], "suffix", ["aabb", "ccdd"]]
            server = LookupKeyServer(pool_worker, _config())

            def recv_once(copy=False):
                server.running = False
                return [(64).to_bytes(4, "big"), b"groups", (32).to_bytes(4, "big"), b"mode", b"hashes"]

            socket.recv_multipart.side_effect = recv_once
            thread_cls.call_args.kwargs["target"]()

            pool_worker.lookup_scheduler.assert_called_once_with(
                64,
                ["aabb", "ccdd"],
                [0, 1],
                use_layerwise=False,
                hbm_hit_tokens=32,
                lookup_hash_mode=LookupHashMode.SUFFIX,
            )
            socket.send.assert_called_once_with((48).to_bytes(4, "big"))

    def test_suffix_logical_view_preserves_full_coordinates(self):
        full_hashes = _hashes(97)
        for hash_block_size in (8, 16, 32):
            for group_scale_factor in (1, 2, 3, 8, 24):
                group_block_size = hash_block_size * group_scale_factor
                full_group_hashes = get_block_hashes(full_hashes, group_block_size, hash_block_size)
                for omitted_hashes in range(len(full_hashes)):
                    logical_hashes = HBMCachedBlockHashList(full_hashes[omitted_hashes:], omitted_hashes)
                    suffix_group_hashes = get_block_hashes(logical_hashes, group_block_size, hash_block_size)
                    lookup_start_group = omitted_hashes // group_scale_factor
                    self.assertEqual(
                        list(suffix_group_hashes[lookup_start_group:]),
                        list(full_group_hashes[lookup_start_group:]),
                        (hash_block_size, group_scale_factor, omitted_hashes),
                    )

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_FakePrefixManager,
        ):
            coordinator = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(
                        ["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
                    )
                ],
                scheduler_block_size=16,
                hash_block_size=16,
                group_block_sizes=[16],
                group_cache_families=["c1"],
            )

        hit_lengths = []
        for logical_hashes in (
            [b"h0", b"h1", b"h2", b"h3"],
            HBMCachedBlockHashList([b"h2", b"h3"], num_hbm_cached_hashes=2),
        ):

            def query_group_hits(group_id, group_block_hashes, lookup_mask):
                self.assertEqual(group_id, 0)
                self.assertIsNone(lookup_mask)
                self.assertEqual(list(group_block_hashes[2:]), [b"h2", b"h3"])
                return group_block_hashes[:3]

            hit_lengths.append(coordinator.find_reachable_hit_tokens(logical_hashes, 64, query_group_hits))

        self.assertEqual(hit_lengths, [48, 48])
