# Standard
import json
import os
from dataclasses import dataclass

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.utils import get_ip, logger

from vllm_ascend.distributed.mooncake.backend.backend import Backend
from vllm_ascend.distributed.mooncake.backend.mooncake_transfer_engine import \
    global_te


class MooncakeBackend(Backend):

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        self.config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()
        if self.config.protocol == "ascend":
            local_hostname = get_ip()
            transfer_engine = global_te.get_transfer_engine(local_hostname,
                                                            device_name=None)
            self.local_seg = local_hostname + ":" + str(
                transfer_engine.get_rpc_port())
            ret = self.store.setup(self.local_seg, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address,
                                   transfer_engine.get_engine())
        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        try:
            config = ReplicateConfig()
            config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = True
            res = self.store.batch_put_from_multi_buffers(
                keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to put key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {keys},error:{e}")

    def get(self, keys: list[str], addrs: list[list[int]],
            sizes: list[list[int]]):
        try:
            res = self.store.batch_get_into_multi_buffers(
                keys, addrs, sizes, True)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys}, res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {keys}, error:{e}")


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    use_ascend_direct: bool

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            use_ascend_direct=config.get("use_ascend_direct", False))

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)

