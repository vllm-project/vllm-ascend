import collections
import os
import time

import ray
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.v1.core.kv_cache_utils import _report_kv_cache_config
from worm_server.mla import __init__
from worm_server.utils import initialize_fake_model_parallel

logger = init_logger(__name__)


@ray.remote(resources={"NPU": 1})
class WORMServlet:
    def __init__(self, rank: int, global_world_size: int, tensor_parallel_size: int, node_id: str, phy_npu_id: int):
        ## Patch the __init__ method for deepseek models
        ## NOTE: Temporary fix for missing arguments when models call
        ## MultiHeadLatentAttentionWrapper rather than AscendMultiHeadLatentAttentionWrapper
        MultiHeadLatentAttentionWrapper.__init__ = __init__

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(phy_npu_id)
        self.master_addr = os.environ.get("MASTER_ADDR")
        self.master_port = int(os.environ.get("MASTER_PORT"))
        self.model_path = os.environ["MODEL_PATH"]
        self.node_id = node_id
        self.phy_npu_id = phy_npu_id
        self.rank = rank
        self.device_id = 0  # each ray actor only sees it's own physical device
        self.local_rank = self.device_id  # in vllm it was used for DP visible devices concept, we dont need it
        self.global_world_size = global_world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.model_world_size = tensor_parallel_size  # initially it will load as per DP=1 and TP degree
        self.expert_parallel_size = self.model_world_size
        self.data_parallel_size = 1  # needs to be 1 for init_distributed_environment to be working for global_world
        self.pipeline_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        assert (
            self.global_world_size % self.tensor_parallel_size == 0,
            f"TP SIZE {self.tensor_parallel_size} should be a divisor of WORLD SIZE {self.global_world_size}",
        )  # Scaling can only happen by TP

        import torch
        from vllm.config import VllmConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from worm_ipc.allocator import IPCSafeAllocator

        KV_GPU_BLOCKS = int(os.getenv("KV_GPU_BLOCKS", 0))

        if KV_GPU_BLOCKS < 1:
            KV_GPU_BLOCKS = None

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.8,
            num_gpu_blocks_override=KV_GPU_BLOCKS,
            max_model_len=1,
            max_num_seqs=1,
            block_size=int(os.getenv("BLOCK_SIZE", 64)),
            served_model_name="worm",
            enforce_eager=True,
            enable_prefix_caching=False,
            distributed_executor_backend="external_launcher",
        )

        self.vllm_config: VllmConfig = engine_args.create_engine_config(usage_context=UsageContext.ENGINE_CONTEXT)
        self.vllm_config.parallel_config.rank = rank
        self.device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(self.device)

        self._pg_cache = {}
        self._npu_ipc_init()
        self.IPCSafeAllocator = IPCSafeAllocator(
            dtype_str=str(self.vllm_config.model_config.dtype).replace("torch.", ""), device_id=self.device_id
        )

        print(
            f"WORMServlet Initialized -> Rank:{self.rank}, Node -> {node_id}, Physical NPU #{os.environ['ASCEND_RT_VISIBLE_DEVICES']}, Total visible devices:{torch.npu.device_count()}"
        )

    def _npu_ipc_init(self):
        self.ipc_config = {
            "dp_rank": int(self.rank // self.tensor_parallel_size),
            "dp_size": int(self.global_world_size / self.tensor_parallel_size),
            "tp_rank": int(self.rank % self.tensor_parallel_size),
            "tp_size": int(self.tensor_parallel_size),
            "device_id": self.local_rank,
        }
        from worm_ipc.npu_ipc_servlet import NPUIPCServlet

        self.ipc_engine = NPUIPCServlet(self.ipc_config)

    """ model and hccl init functions """

    def _init_global_world(self):
        import torch.distributed as dist

        # EVERY rank runs vLLM's world init (creates cpu_group via gloo over [0..7])
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=self.global_world_size,
            rank=self.rank,
            distributed_init_method=f"tcp://{self.master_addr}:{self.master_port}",
            local_rank=self.local_rank,
            backend="hccl",
        )
        initialize_fake_model_parallel(
            rank=self.rank,
            world_size=self.global_world_size,
            tensor_model_parallel_size=self.tensor_parallel_size,
            pipeline_model_parallel_size=self.pipeline_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            backend="hccl",
        )
        return (self.rank, dist.is_initialized(), dist.get_world_size())

    def _destroy_global_world(self):
        import torch.distributed as dist
        from vllm.distributed.parallel_state import destroy_distributed_environment

        if dist.is_initialized():
            # dist.destroy_process_group()
            destroy_distributed_environment()
        return "ok"

    def _init_model(self):
        import torch
        from vllm.model_executor.model_loader import get_model_loader, utils
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.utils.torch_utils import set_default_torch_dtype

        vllm_config = self.vllm_config
        model_config = vllm_config.model_config
        loader = get_model_loader(vllm_config.load_config)
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)

        with set_default_torch_dtype(model_config.dtype):
            with target_device, self.IPCSafeAllocator:
                model = utils.initialize_model(vllm_config=vllm_config, model_config=model_config)
                ## From /home/tim/elasticmoe_debug/elasticmoe/vllm/model_executor/model_loader/base_loader.py
                logger.info("Model initialized on %s ...", target_device)

            loader.load_weights(model, model_config)
            logger.info("Weights loaded on %s ...", target_device)

            process_weights_after_loading(model, model_config, target_device)
            logger.info("Weights processed on %s ...", target_device)

        self.model = model.eval()

        self.named_parameters = collections.OrderedDict(self.model.named_parameters())
        self.ipc_engine.set_model_params(self.named_parameters)
        self.ipc_engine.check_all_weights_on_npu(self.named_parameters)
        return "ok"

    def _init_kv_cache(self, num_gpu_blocks=None, kv_cache_configs=[None]):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        with set_current_vllm_config(self.vllm_config):
            self.model_runner = NPUModelRunner(vllm_config=self.vllm_config, device=self.device)

        if kv_cache_configs[0] is not None:
            self.model_runner.kv_cache_config = kv_cache_configs[0]
            self.kv_cache_config = kv_cache_configs[0]

            assert len(kv_cache_configs) == 1, "Expected each servlet to have config for one instance of kv_cache"
            with self.IPCSafeAllocator:
                # Initialize the memory buffer for KV cache
                kv_caches = self.model_runner._allocate_kv_cache_tensors(kv_cache_configs[0])

        # if kv_cache_config is None: # Initial model loading for TP ranks (DP=1)
        else:
            # Get all kv cache needed by the model
            kv_cache_specs = self.model_runner.get_kv_cache_spec()

            kv_cache_specs = [kv_cache_specs]
            has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)

            # Attention free models don't need memory for kv cache
            available_gpu_memory = [26414821376] * len(kv_cache_specs)

            assert len(kv_cache_specs) == len(available_gpu_memory)

            kv_cache_configs = get_kv_cache_configs(self.vllm_config, kv_cache_specs, available_gpu_memory)

            self.model_runner.kv_cache_config = kv_cache_configs[0]

            for kv_cache_config in kv_cache_configs:
                kv_cache_config.num_blocks = num_gpu_blocks

                # Manually enforing num_gpu_blocks on kv_cache_config
                for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                    kv_cache_tensor.size = int(kv_cache_tensor.size / kv_cache_config.num_blocks * num_gpu_blocks)

                if len(kv_cache_config.kv_cache_groups) > 0:
                    _report_kv_cache_config(self.vllm_config, kv_cache_config)

            assert len(kv_cache_configs) == 1, "Expected each servlet to have config for one instance of kv_cache"
            with self.IPCSafeAllocator:
                kv_caches = self.model_runner._allocate_kv_cache_tensors(kv_cache_configs[0])

        self.kv_caches = kv_caches
        self.kv_cache_config = kv_cache_configs
        self.ipc_engine.set_kv_caches(self.kv_caches)

    """ utlity and meta data dict broadcasting functions """

    def breakpoint(self, rank=None, type="debugpy"):
        if (not rank) or (rank and self.rank == rank):
            if type == "debugpy":
                import debugpy

                debug_port = 2500 + self.rank
                self.print(f"Listening DEBUGPY on port {debug_port}")
                debugpy.listen(("0.0.0.0", debug_port))
                debugpy.wait_for_client()  # <-- blocks here until VS Code attaches
                debugpy.breakpoint()  # <-- optional, drop you right on this line
            else:
                from ray.util import pdb as ray_pdb

                ray_pdb.set_trace()

    def peek_env(self):
        import os

        return {
            "rank": self.rank,
            "node": self.node_id,
            "HCCL_IF_IP": os.environ.get("HCCL_IF_IP"),
            "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
        }

    def print(self, msg):
        print(f"Rank {self.rank}: " + str(msg))

    def get_attribute(self, name: str):
        if not hasattr(self, name):
            self.print(f"{self.__class__.__name__} has no attribute '{name}'")
            return None
        else:
            return getattr(self, name)

    def set_attribute(self, name: str, value_or_oid):
        import ray

        value = ray.get(value_or_oid) if isinstance(value_or_oid, ray.ObjectRef) else value_or_oid
        setattr(self, name, value)
        return "ok"

    def generate_metadata_dict(self):
        params_shape_dict = collections.OrderedDict()
        for param_name, param in self.model.named_parameters():
            params_shape_dict[param_name] = (param.shape, param.dtype)
        self.metadata_dict = {
            "params_shape_dict": params_shape_dict,
            "kv_cache_config": self.kv_cache_config,
        }
        return self.metadata_dict

    def broadcast_metadata_dict(self, peers=None):  # peers: list[ray.actor.ActorHandle]
        import ray

        if peers is None:
            peers = self.peers
        self.metadata_dict = self.generate_metadata_dict()
        oid = ray.put(self.metadata_dict)
        futures = [
            peer.set_attribute.remote(name="metadata_dict", value_or_oid=oid) for peer in peers if peer is not self
        ]
        return futures

    def health(self):
        return {
            "rank": self.rank,
            "role": self.rank < self.model_world_size,
        }

    def model_health(self):
        import torch

        if not hasattr(self, "named_parameters"):
            return None

        w13_weight = torch.sum(self.named_parameters["model.layers.4.mlp.experts.w13_weight"]).item()

        lm_head = torch.sum(self.named_parameters["lm_head.weight"]).item()

        kv_cache_shape = [cache.shape for cache in self.kv_caches["model.layers.2.self_attn.attn"]]
        return {"w13_weight": w13_weight, "lm_head": lm_head, "kv_cache_shape": kv_cache_shape}

    """ hccl communication functions """

    def _build_new_process_groups(self, ranks):
        import torch.distributed as dist

        pg = dist.new_group(ranks=ranks, backend="hccl")
        dist.barrier()
        self.print(f"Creating New Group on {ranks}")
        return pg

    def warm_up(self):
        import torch
        import torch.distributed as dist

        warm = torch.zeros(1, dtype=torch.int32, device="npu:0")
        if self.rank == 0:
            warm.fill_(123)
        self.print("warm up done")
        dist.broadcast(warm, src=0)

    def barrier_world(self):
        import torch.distributed as dist

        dist.barrier()
        return "ok"

    def ping_allreduce(self, val: int = 1):
        import torch
        import torch.distributed as dist

        x = torch.tensor([val], dtype=torch.int32, device="npu:0")
        dist.all_reduce(x)
        return int(x.item())

    def ring_probe(self):
        import torch
        import torch.distributed as dist

        torch.npu.set_device("npu:0")
        world = self.global_world_size
        src = self.rank
        dst = (self.rank + 1) % world

        buf = torch.full((16,), src, dtype=torch.float32, device="npu:0")
        if self.rank % 2 == 0:
            # even ranks send then recv
            w = dist.isend(buf, dst=dst)
            w.wait()
            recv = torch.empty_like(buf)
            dist.recv(recv, src=(self.rank - 1 + world) % world)
        else:
            # odd ranks recv then send
            recv = torch.empty_like(buf)
            dist.recv(recv, src=(self.rank - 1 + world) % world)
            w = dist.isend(buf, dst=dst)
            w.wait()
        return int(recv[0].item())  # should equal src of left neighbor

    """ SCALING FUNCTIONS """

    def scaleup(self, num_npus=4):
        import collections

        import torch
        import torch.distributed as dist

        # pick device for Ascend before any comms
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

        free_ranks = list(range(self.model_world_size, self.global_world_size))
        destination_ranks = free_ranks[:num_npus]  # (kept; not used later but harmless)
        source_ranks = list(range(self.model_world_size))
        source_rank_map = {
            tp_rank: list(range(tp_rank, self.model_world_size, self.tensor_parallel_size))
            for tp_rank in range(self.tensor_parallel_size)
        }
        destination_rank_map = {
            tp_rank: list(
                range(self.model_world_size + tp_rank, self.model_world_size + num_npus, self.tensor_parallel_size)
            )
            for tp_rank in range(self.tensor_parallel_size)
        }

        start_time = time.time()
        for tp_rank in range(self.tensor_parallel_size):
            src_rank = source_rank_map[tp_rank][0]

            if self.rank == src_rank:
                # Source: send all params to each destination in a consistent order
                handles = []
                for param_name, param in self.named_parameters.items():
                    # param is already a tensor on NPU; send order is per-named_parameters()

                    # self.print(f"!!! {self.named_parameters[param_name]=}")
                    for dst_rank in destination_rank_map[tp_rank]:
                        try:
                            w = dist.isend(param, dst=dst_rank)  # default group = WORLD/HCCL
                            handles.append(w)
                        except (NotImplementedError, RuntimeError):
                            # Fallback to blocking send if async P2P unsupported
                            dist.send(param, dst=dst_rank)

                # Wait for async sends to complete
                for h in handles:
                    h.wait()

            elif self.rank in destination_rank_map[tp_rank]:
                # Destination: receive tensors in the SAME order as the sender's sends.
                self.named_parameters = collections.OrderedDict()
                handles = []

                for param_name, (param_size, param_dtype) in self.metadata_dict["params_shape_dict"].items():
                    # Allocate receive buffer (dtype must match sender; your model is float16)

                    with self.IPCSafeAllocator:
                        recv_buf = torch.empty(param_size, dtype=param_dtype, device=device)
                    self.named_parameters[param_name] = recv_buf

                    try:
                        w = dist.irecv(recv_buf, src=src_rank)  # default group = WORLD/HCCL
                        handles.append(w)
                    except (NotImplementedError, RuntimeError):
                        dist.recv(recv_buf, src=src_rank)

                # Wait for all irecvs
                for h in handles:
                    h.wait()

                self.ipc_engine.set_model_params(self.named_parameters)

                # Init KV Caches
                self._init_kv_cache(kv_cache_configs=self.metadata_dict["kv_cache_config"])

            else:
                # Non-participating ranks for this tp_rank just wait
                pass

        self.print(f"TOTAL TIME TAKEN IN SCALEUP {time.time() - start_time}")
        self.model_world_size += num_npus
        self.data_parallel_size += int(num_npus / self.tensor_parallel_size)
        return "ok"

    def scaledown(self, num_npus=2):
        import torch.distributed as dist

        scaled_down_model_size = self.model_world_size - num_npus
        if scaled_down_model_size < self.tensor_parallel_size:
            self.print(
                f"Can't scale down below the TP Size limit. Target size:{scaled_down_model_size} < TP Size:{self.tensor_parallel_size}"
            )
            return "cant_scale"
        else:
            retiring_ranks = list(range(scaled_down_model_size, self.model_world_size))

            if self.rank in retiring_ranks:
                self.ipc_engine.reset()
                self.IPCSafeAllocator.deallocate_all()
                del self.named_parameters
                del self.kv_caches
                del self.kv_cache_config

            dist.barrier()
            self.model_world_size -= num_npus
            self.data_parallel_size -= int(num_npus / self.tensor_parallel_size)
            return "ok"
