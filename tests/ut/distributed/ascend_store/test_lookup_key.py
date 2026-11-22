import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec
from zmq.error import ZMQError

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import LookupKeyClient, LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LookupResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )


class TestReplayLookupProtocol(unittest.TestCase):
    def test_real_rpc_preserves_save_boundary_and_requires_all_replicas(self):
        """Exercise client, server, worker and managers with only store I/O faked."""
        for ranks in (1, 2, 4):
            with self.subTest(ranks=ranks), tempfile.TemporaryDirectory(prefix="kv-replay-") as rpc_dir:
                groups = [
                    KVCacheGroupSpec(
                        ["layer.0"],
                        FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=1,
                            dtype=torch.float32,
                        ),
                    ),
                    KVCacheGroupSpec(
                        ["layer.1"],
                        MambaSpec(
                            block_size=16,
                            shapes=((4,),),
                            dtypes=(torch.float32,),
                            mamba_cache_mode="align",
                        ),
                    ),
                ]
                worker = KVPoolWorker.__new__(KVPoolWorker)
                worker.cache_coordinator = AscendStoreCoordinator(
                    groups,
                    16,
                    16,
                    [16, 16],
                    ["default", "default"],
                    use_eagle=True,
                )
                worker.num_kv_cache_groups = 2
                worker.max_model_len = 64
                worker.kv_send_thread = None
                worker.token_database = MagicMock()
                worker.m_store = MagicMock()
                hashes = [bytes([i]) * 32 for i in range(4)]
                present = {
                    f"g{group}:{h.hex()}@{rank}"
                    for group, blocks in ((0, range(4)), (1, (1, 3)))
                    for h in (hashes[i] for i in blocks)
                    for rank in range(ranks)
                }

                def keys(token_len, block_hashes, mask_num, kv_cache_group_id, chunk_mask):
                    for i, h in enumerate(block_hashes[: token_len // 16]):
                        if i * 16 >= mask_num and (chunk_mask is None or chunk_mask[i]):
                            value = h if isinstance(h, str) else h.hex()
                            yield i * 16, (i + 1) * 16, f"g{kv_cache_group_id}:{value}", h

                worker.token_database.process_token_key_strings.side_effect = keys
                key_variants = MagicMock(
                    side_effect=lambda key, group, all_ranks, ranks=ranks: [f"{key}@{rank}" for rank in range(ranks)]
                )
                worker.m_store.exists.side_effect = lambda values, present=present: [
                    int(key in present) for key in values
                ]
                with (
                    patch.object(worker, "_expand_lookup_key_variants", key_variants),
                    patch(
                        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.envs.VLLM_RPC_BASE_PATH",
                        rpc_dir,
                    ),
                ):
                    server = LookupKeyServer(worker, _config())
                    client = LookupKeyClient(_config())
                    try:
                        self.assertEqual(
                            client.lookup_with_replay(64, hashes, [0, 1], replay_token_limit=48),
                            LookupResult(64, 32),
                        )
                        self.assertEqual(worker.m_store.exists.call_count, 2)
                        self.assertTrue(all(c.args[2] for c in key_variants.call_args_list))
                        present.remove(f"g1:{hashes[1].hex()}@{ranks - 1}")
                        self.assertEqual(
                            client.lookup_with_replay(64, hashes, [0, 1], replay_token_limit=48),
                            LookupResult(64, 0),
                        )
                        self.assertEqual(worker.m_store.exists.call_count, 4)
                        # The legacy wire format and reset remain usable on the
                        # same socket after paired responses.
                        self.assertEqual(client.lookup(64, hashes, [0, 1]), 64)
                        self.assertTrue(client.reset())
                        worker.m_store.remove_all.assert_called_once()
                        worker.m_store.exists.side_effect = RuntimeError("store unavailable")
                        self.assertEqual(
                            client.lookup_with_replay(64, hashes, [0, 1], replay_token_limit=48),
                            LookupResult(0, 0),
                        )
                    finally:
                        client.close()
                        server.close()

    def test_invalid_or_failed_response_cannot_become_a_hit(self):
        def response(raw, replay):
            return raw.to_bytes(4, "big") + replay.to_bytes(4, "big")

        for reply in (b"\x00", b"", response(80, 32), response(64, 64), response(16, 32), ZMQError("timeout")):
            with (
                self.subTest(reply=reply),
                patch(
                    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.make_zmq_socket"
                ) as socket_factory,
            ):
                socket = socket_factory.return_value
                if isinstance(reply, Exception):
                    socket.recv.side_effect = reply
                else:
                    socket.recv.return_value = reply
                client = LookupKeyClient(_config())
                try:
                    self.assertEqual(
                        client.lookup_with_replay(64, [b"a"], [0, 1], replay_token_limit=48),
                        LookupResult(0, 0),
                    )
                    self.assertTrue(client.closed)
                    self.assertEqual(
                        client.lookup_with_replay(64, [b"a"], [0, 1], replay_token_limit=48),
                        LookupResult(0, 0),
                    )
                    socket.send_multipart.assert_called_once()
                finally:
                    client.close()

    def test_server_error_response_fails_closed(self):
        worker = MagicMock()
        worker.lookup_scheduler_with_replay.side_effect = ValueError("incompatible layout")
        with (
            tempfile.TemporaryDirectory(prefix="kv-replay-") as rpc_dir,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.envs.VLLM_RPC_BASE_PATH", rpc_dir
            ),
        ):
            server = LookupKeyServer(worker, _config())
            client = LookupKeyClient(_config())
            try:
                self.assertEqual(client.lookup_with_replay(64, [b"a"], [0], replay_token_limit=48), LookupResult(0, 0))
                self.assertTrue(client.closed)
            finally:
                client.close()
                server.close()
