import math
import os
import pickle
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Optional

import torch
import vllm.envs as envs
import zmq
from vllm.config import KVTransferConfig, VllmConfig
from vllm.utils import get_dtype_size, logger, make_zmq_socket
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.distributed.cpu_offload_manager.cpu_kv_cache_manager import \
    CPUKVCacheManager


@dataclass
class MLAConfig:
    nope_dim: int
    rope_dim: int


def get_cpu_offload_connector(vllm_config: VllmConfig) -> KVTransferConfig:
    if vllm_config.kv_transfer_config is not None:
        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config.kv_connector == "CPUOffloadingConnector":
            return kv_transfer_config
        elif kv_transfer_config.kv_connector == "MultiConnector":
            ktcs = kv_transfer_config.kv_connector_extra_config.get(
                "connectors")
            for ktc in ktcs:
                kv_transfer_config = KVTransferConfig(**ktc)
                if kv_transfer_config.kv_connector == "CPUOffloadingConnector":
                    return kv_transfer_config
    return None


class MetadataServer:
    METADATA_SERVER_ADDRESS = f"ipc://{envs.VLLM_RPC_BASE_PATH}/metadata.ipc"
    DEFAULT_CPU_SWAP_SPACE_GB = 800

    class ZMQRPCClient:

        def __init__(self, identity=f"worker-{os.getpid()}"):
            logger.info(f"metadata client for worker {identity} started")
            self.ctx = zmq.Context()  # type: ignore
            self.socket = make_zmq_socket(
                self.ctx,
                MetadataServer.METADATA_SERVER_ADDRESS,
                zmq.DEALER,  # type: ignore
                bind=False,
                identity=identity.encode(),
                linger=0)

        def call(self, func_name: str, *args, **kwargs) -> Any:
            request = (func_name, args, kwargs)
            self.socket.send(b"", zmq.SNDMORE)  # type: ignore
            self.socket.send(pickle.dumps(request))
            _ = self.socket.recv()
            response = pickle.loads(self.socket.recv())
            result, error = response
            if error:
                logger.exception(f"call metadata sever error: {error}")
                raise error
            if func_name == "init_cpu_kv_caches":
                (memory_dict, layer_size, layer_dtype, mla_config) = result
                # shared_memory_dict is recorded in self to close
                self.shared_memory_dict = memory_dict
                result = {}
                for key, shm in memory_dict.items():
                    tensor = torch.frombuffer(
                        shm.buf, dtype=layer_dtype).reshape(layer_size)
                    if mla_config is not None:
                        tensor = tensor.split(
                            [mla_config.nope_dim, mla_config.rope_dim], dim=-1)
                    result[key] = tensor
            return result

        def __del__(self):
            # will be finalized by outer process
            self.socket.close()
            self.ctx.term()
            if hasattr(self, 'shared_memory_dict'):
                for shm in self.shared_memory_dict.values():
                    shm.close()

    def __init__(self, vllm_config: VllmConfig):
        self.world_size = vllm_config.parallel_config.world_size
        self.pipeline_parallel_size = vllm_config.parallel_config.pipeline_parallel_size
        kv_transfer_config = get_cpu_offload_connector(vllm_config)
        assert kv_transfer_config is not None
        available_memory_gb = kv_transfer_config.get_from_extra_config(
            "cpu_swap_space_gb", MetadataServer.DEFAULT_CPU_SWAP_SPACE_GB)
        self.available_memory = available_memory_gb * 1024 * 1024 * 1024
        logger.info(f"cpu swap space: {self.available_memory} bytes")
        self.ctx = zmq.Context()  # type: ignore
        self.socket = make_zmq_socket(
            self.ctx,
            MetadataServer.METADATA_SERVER_ADDRESS,
            zmq.ROUTER,  # type: ignore
            bind=True,
            linger=0)
        self.functions: dict[str, Callable] = {
            "init_cpu_kv_caches": self.init_cpu_kv_caches,
            "post_init": self.post_init,
            "ready": self.ready,
        }
        self.shared_memory = {}  # type: ignore
        self.num_cpu_blocks = -1

    @staticmethod
    def _safe_create_shared_memory(name: str, size: int) -> SharedMemory:
        try:
            existing_shm = SharedMemory(name=name, create=False)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        return SharedMemory(name=name, create=True, size=size)

    def ready(self):
        return True

    def init_cpu_kv_caches(
        self,
        pp_rank: int,
        tp_rank: int,
        kv_cache_specs: dict[str, AttentionSpec],
        mla_config: MLAConfig,
    ) -> tuple[dict[str, SharedMemory], tuple[int, ...], torch.dtype,
               MLAConfig]:
        logger.info(f"receive pp rank: {pp_rank}, tp rank: {tp_rank}")
        # follow the assumption that each layer has the same spec
        layer = next(iter(kv_cache_specs.values()))
        assert all([
            layer.page_size_bytes == any.page_size_bytes
            for any in kv_cache_specs.values()
        ])
        # mla shares the same kv cache among different tp
        if layer.use_mla:
            tp_rank = 0
        if (pp_rank, tp_rank) in self.shared_memory:
            return self.shared_memory[(pp_rank, tp_rank)]
        available_memory = self.available_memory
        shared_memory_dict = {}
        if layer.use_mla:
            available_memory //= self.pipeline_parallel_size
            available_memory //= len(kv_cache_specs)
            num_blocks = available_memory // layer.page_size_bytes
            layer_size = (num_blocks, layer.block_size, layer.num_kv_heads,
                          layer.head_size)  # type: ignore
        else:
            available_memory //= self.world_size
            available_memory //= len(kv_cache_specs)
            num_blocks = available_memory // layer.page_size_bytes
            layer_size = (2, num_blocks, layer.block_size, layer.num_kv_heads,
                          layer.head_size)  # type: ignore
        nbytes = math.prod(layer_size) * get_dtype_size(layer.dtype)
        for layer_name in kv_cache_specs.keys():
            # only this format can share during ZeroMQ+pickle
            shared_memory_dict[
                layer_name] = MetadataServer._safe_create_shared_memory(
                    f"cpu_kv_cache_{pp_rank}_{tp_rank}_{layer_name}", nbytes)
        if layer.use_mla:
            assert mla_config is not None
            assert layer.head_size == mla_config.rope_dim + mla_config.nope_dim
            self.shared_memory[(pp_rank,
                                tp_rank)] = (shared_memory_dict, layer_size,
                                             layer.dtype, mla_config)
        else:
            self.shared_memory[(pp_rank,
                                tp_rank)] = (shared_memory_dict, layer_size,
                                             layer.dtype, None)
        if self.num_cpu_blocks == -1 or num_blocks < self.num_cpu_blocks:
            self.num_cpu_blocks = num_blocks
        self.layer = layer
        return self.shared_memory[(pp_rank, tp_rank)]

    def post_init(self):
        # different processors in data parallel may call multiple times
        if hasattr(self, 'cpu_block_manager'):
            return
        # do shared_memory() at least once
        logger.info(f"assign cpu num blocks: {self.num_cpu_blocks}")
        assert self.num_cpu_blocks >= 0
        self.cpu_block_manager = CPUKVCacheManager(self.layer,
                                                   self.num_cpu_blocks)
        self.functions.update({
            "get_matched_num_and_touch":
            self.cpu_block_manager.get_matched_num_and_touch,
            "allocate_slots":
            self.cpu_block_manager.allocate_slots,
            "record_request_cache_and_free_slots":
            self.cpu_block_manager.record_request_cache_and_free_slots,
            "cache_and_free_slots":
            self.cpu_block_manager.cache_and_free_slots,
        })

    def serve_step(self):
        client_id = self.socket.recv()
        _ = self.socket.recv()
        raw_msg = self.socket.recv()
        try:
            func_name, args, kwargs = pickle.loads(raw_msg)
        except Exception as e:
            response = (None, Exception(f"Invalid request: {str(e)}"))
        else:
            if func_name in self.functions:
                try:
                    result = self.functions[func_name](*args, **kwargs)
                    response = (result, None)  # type: ignore
                except Exception as e:
                    logger.exception(f"metadata execute error: {e}")
                    response = (None, e)  # type: ignore
            else:
                response = (None, NameError(f"Function {func_name} not found"))
        self.socket.send(client_id, zmq.SNDMORE)  # type: ignore
        self.socket.send(b"", zmq.SNDMORE)  # type: ignore
        self.socket.send(pickle.dumps(response))

    def shutdown(self):
        self.socket.close()
        self.ctx.term()
        socket_path = MetadataServer.METADATA_SERVER_ADDRESS.replace(
            "ipc://", "")
        if os.path.exists(socket_path):
            os.remove(socket_path)
        for cached in self.shared_memory.values():
            for shm in cached[0].values():
                shm.close()
                shm.unlink()


class MetadataServerProc:

    @staticmethod
    def run_metadata_server(vllm_config: VllmConfig):
        if (not vllm_config.cache_config.enable_prefix_caching
                or get_cpu_offload_connector(vllm_config) is None):
            return

        shutdown_requested = False

        def _signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        # signal.signal(signal.SIGTERM, _signal_handler)
        # signal.signal(signal.SIGINT, _signal_handler)
        metadata_server: Optional[MetadataServer] = None
        try:
            metadata_server = MetadataServer(vllm_config)
            logger.info("Metadata server started.")
            while True:
                metadata_server.serve_step()
        except SystemExit:
            logger.info("Metadata server exiting.")
            raise
        except Exception as e:
            logger.exception(f"Metadata server error: {e}.")
            raise e
        finally:
            if metadata_server is not None:
                metadata_server.shutdown()
