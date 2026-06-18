import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_mock_modules():
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm.envs = SimpleNamespace(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1)
    sys.modules["vllm"] = vllm

    modules = {
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.kv_transfer": types.ModuleType("vllm.distributed.kv_transfer"),
        "vllm.distributed.kv_transfer.kv_connector": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
        "vllm.distributed.kv_transfer.kv_connector.factory": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.factory"
        ),
        "vllm.distributed.kv_transfer.kv_connector.utils": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.utils"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.base": types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.base"
        ),
        "vllm.distributed.parallel_state": types.ModuleType("vllm.distributed.parallel_state"),
        "vllm.distributed.utils": types.ModuleType("vllm.distributed.utils"),
        "vllm.logger": types.ModuleType("vllm.logger"),
        "vllm.logging_utils": types.ModuleType("vllm.logging_utils"),
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.math_utils": types.ModuleType("vllm.utils.math_utils"),
        "vllm.utils.network_utils": types.ModuleType("vllm.utils.network_utils"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.core": types.ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.output": types.ModuleType("vllm.v1.core.sched.output"),
        "vllm.v1.kv_cache_interface": types.ModuleType("vllm.v1.kv_cache_interface"),
        "vllm.v1.request": types.ModuleType("vllm.v1.request"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.utils": types.ModuleType("vllm.v1.worker.utils"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backend": types.ModuleType("vllm.v1.attention.backend"),
        "vllm.forward_context": types.ModuleType("vllm.forward_context"),
    }
    sys.modules.update(modules)
    for package_name in (
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ):
        modules[package_name].__path__ = []

    modules["vllm.config"].VllmConfig = object
    modules["vllm.distributed"].get_pcp_group = lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    modules["vllm.distributed.kv_transfer.kv_connector.utils"].BlockIds = tuple
    modules["vllm.distributed.kv_transfer.kv_connector.factory"].KVConnectorFactory = type(
        "KVConnectorFactory",
        (),
        {
            "_registry": {},
            "register_connector": classmethod(
                lambda cls, name, module, class_name: cls._registry.update({name: (module, class_name)})
            ),
        },
    )

    base = modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    base.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {})
    base.KVConnectorHandshakeMetadata = type("KVConnectorHandshakeMetadata", (), {})
    base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
    base.KVConnectorRole = SimpleNamespace(SCHEDULER="scheduler", WORKER="worker")
    base.SupportsHMA = type("SupportsHMA", (), {})

    ps = modules["vllm.distributed.parallel_state"]
    ps.get_pp_group = lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    ps.get_tensor_model_parallel_rank = lambda: 0
    ps.get_tensor_model_parallel_world_size = lambda: 1
    ps.get_tp_group = lambda: SimpleNamespace(rank_in_group=0, world_size=1)

    modules["vllm.distributed.utils"].get_pp_indices = lambda n, r, s: (0, n)
    modules["vllm.logger"].logger = MagicMock()
    modules["vllm.logging_utils"].ColoredFormatter = object
    modules["vllm.logging_utils"].NewLineFormatter = object
    modules["vllm.utils.math_utils"].cdiv = lambda a, b: (a + b - 1) // b
    modules["vllm.utils.network_utils"].get_ip = lambda: "127.0.0.1"
    modules["vllm.utils.network_utils"].make_zmq_path = lambda proto, host, port: f"{proto}://{host}:{port}"
    modules["vllm.utils.network_utils"].make_zmq_socket = MagicMock()
    modules["vllm.v1.core.sched.output"].SchedulerOutput = object

    kv = modules["vllm.v1.kv_cache_interface"]

    class FullAttentionSpec:
        def __init__(self, block_size=16, num_kv_heads=None):
            self.block_size = block_size
            self.num_kv_heads = num_kv_heads

    class SlidingWindowSpec(FullAttentionSpec):
        def __init__(self, block_size=16, sliding_window=32):
            super().__init__(block_size)
            self.sliding_window = sliding_window

    class MambaSpec:
        pass

    class UniformTypeKVCacheSpecs:
        def __init__(self, kv_cache_specs):
            self.kv_cache_specs = kv_cache_specs

    kv.FullAttentionSpec = FullAttentionSpec
    kv.KVCacheConfig = object
    kv.MambaSpec = MambaSpec
    kv.SlidingWindowSpec = SlidingWindowSpec
    kv.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    modules["vllm.v1.request"].RequestStatus = SimpleNamespace(FINISHED_LENGTH_CAPPED="finished")
    modules["vllm.v1.worker.utils"].extract_layer_index = lambda layer_name, _: int(layer_name.rsplit(".", 1)[-1])

    fake_engine = types.ModuleType("mooncake.engine")
    fake_engine.TransferEngine = MagicMock()
    sys.modules["mooncake.engine"] = fake_engine
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
    ascend_parallel_state = types.ModuleType("vllm_ascend.distributed.parallel_state")
    ascend_parallel_state.get_p_tp_group = lambda: SimpleNamespace(device_group=None)
    sys.modules["vllm_ascend.distributed.parallel_state"] = ascend_parallel_state
    ascend_distributed_utils = types.ModuleType("vllm_ascend.distributed.utils")
    ascend_distributed_utils.get_decode_context_model_parallel_rank = lambda: 0
    ascend_distributed_utils.get_decode_context_model_parallel_world_size = lambda: 1
    sys.modules["vllm_ascend.distributed.utils"] = ascend_distributed_utils
    ascend_utils = types.ModuleType("vllm_ascend.utils")
    ascend_utils.enable_custom_op = lambda: False
    ascend_utils.is_vl_model = lambda _: False
    sys.modules["vllm_ascend.utils"] = ascend_utils


_install_mock_modules()

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
        first = manager.alloc_blocks(([10, 11],))
        self.assertEqual(first, ([0, 1],))
        self.assertIsNone(manager.alloc_blocks(([12],)))
        manager.free_blocks(([0],))
        self.assertEqual(manager.alloc_blocks(([12],)), ([0],))

    def test_cpu_cache_manager_allocates_blocks_per_group(self):
        manager = d2rh.D2RHCPUCacheManager(4)

        block_ids = manager.alloc_blocks(([10, 11], [20, 21]))
        self.assertEqual(block_ids, ([0, 1], [2, 3]))
        self.assertIsNone(manager.alloc_blocks(([], [22])))

        manager.free_blocks(([], [2]))
        self.assertEqual(manager.alloc_blocks(([], [22])), ([], [2]))

    def test_cpu_cache_manager_group_allocation_is_atomic(self):
        manager = d2rh.D2RHCPUCacheManager(1)

        self.assertIsNone(manager.alloc_blocks(([10], [20, 21])))
        self.assertEqual(manager.alloc_blocks(([10], [])), ([0], []))


if __name__ == "__main__":
    unittest.main()
