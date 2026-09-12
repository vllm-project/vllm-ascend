import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh  # noqa: E402


class TestD2RHAdapter(unittest.TestCase):
    def test_agent_metadata_uses_current_mooncake_schema(self):
        metadata = d2rh.MooncakeAgentMetadata(
            engine_id="engine",
            te_rpc_port=1234,
            kv_group2layeridx={0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])},
            block_size=16,
            kv_caches_base_addr=[[1000, 2000]],
            block_size_scale=[[1, 1]],
            num_blocks=8,
            block_lens=[[128, 128]],
            block_strides=[[256, 256]],
            local_ip="127.0.0.1",
        )

        self.assertEqual(metadata.block_strides, [[256, 256]])
        self.assertEqual(metadata.kv_group2layeridx[0][1], [0])

    def test_build_start_pull_params_carries_group_pulls(self):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler._prefill_tp_size = 4
        scheduler._decode_tp_size = 2
        scheduler._prefill_pp_size = 1
        scheduler.num_key_value_heads = 4
        scheduler.is_deepseek_mla = False
        scheduler.use_sparse = False
        scheduler.tp_size = 2
        scheduler.kv_cache_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16, num_kv_heads=4),
                layer_names=["layer.0"],
            )
        ]

        params = scheduler._build_start_pull_params(
            "req",
            {
                "remote_request_id": "remote-req",
                "remote_port": 30000,
                "remote_host": "p-host",
                "remote_engine_id": "p-engine",
                "remote_block_ids": ([1, 2],),
            },
            decode_tp_rank=0,
        )

        self.assertEqual(params["remote_handshake_ports"], [30000, 30001])
        self.assertEqual(len(params["group_pulls_by_port"]), 2)
        self.assertEqual(params["group_pulls_by_port"][0][0].group_id, 0)
        self.assertEqual(params["group_pulls_by_port"][1][0].is_group_transfer_end, True)

    def test_cpu_staging_block_map_is_group_aware_for_hop2(self):
        recv_thread = object.__new__(d2rh.KVCacheRecvingThread)
        recv_thread.remote_local_block_map = {
            "remote-req": {
                (0, 10): 3,
                (0, 11): 4,
                (1, 7): 5,
            }
        }
        recv_thread.cpu_host = "127.0.0.1"

        req_meta = {
            "request_id": "req",
            "remote_request_id": "remote-req",
            "remote_block_ids": ([10, 11], [7]),
            "remote_engine_id": "p-engine",
            "remote_host": "p-host",
            "remote_handshake_port": 30000,
        }

        with patch.object(d2rh.BaseKVCacheRecvingThread, "_transfer_kv_cache_all_groups") as mock_transfer:
            d2rh.KVCacheRecvingThread._transfer_kv_cache_all_groups(recv_thread, req_meta)

        transferred_meta = mock_transfer.call_args.args[0]
        self.assertEqual(transferred_meta["remote_block_ids"], ([3, 4], [5]))
        self.assertEqual(transferred_meta["remote_engine_id"], d2rh.CPU_STAGING_ENGINE_ID)
        self.assertEqual(transferred_meta["remote_handshake_port"], d2rh.CPU_STAGING_HANDSHAKE_PORT)

    def test_d2rh_block_map_helpers_keep_manager_api_as_block_ids(self):
        block_map = d2rh._build_block_map(([10, 11], [20]), ([0, 1], [0]))
        self.assertEqual(block_map, {(0, 10): 0, (0, 11): 1, (1, 20): 0})
        self.assertEqual(d2rh._group_block_map_values(block_map), ([0, 1], [0]))

    def test_cpu_cache_manager_reuses_freed_blocks(self):
        manager = d2rh.D2RHCPUCacheManager(2)
        first = manager.alloc_block_map(([10, 11],))
        self.assertEqual(first, {(0, 10): 0, (0, 11): 1})
        self.assertIsNone(manager.alloc_block_map(([12],)))
        manager.free_block_map({(0, 10): 0})
        self.assertEqual(manager.alloc_block_map(([12],)), {(0, 12): 0})

    def test_cpu_cache_manager_allocates_blocks_per_group(self):
        manager = d2rh.D2RHCPUCacheManager(4)

        block_map = manager.alloc_block_map(([10, 11], [20, 21]))
        self.assertEqual(block_map, {(0, 10): 0, (0, 11): 1, (1, 20): 2, (1, 21): 3})
        self.assertIsNone(manager.alloc_block_map(([], [22])))

        manager.free_block_map({(1, 20): 2})
        self.assertEqual(manager.alloc_block_map(([], [22])), {(1, 22): 2})

    def test_cpu_cache_manager_group_allocation_is_atomic(self):
        manager = d2rh.D2RHCPUCacheManager(1)

        self.assertIsNone(manager.alloc_block_map(([10], [20, 21])))
        self.assertEqual(manager.alloc_block_map(([10], [])), {(0, 10): 0})


if __name__ == "__main__":
    unittest.main()
