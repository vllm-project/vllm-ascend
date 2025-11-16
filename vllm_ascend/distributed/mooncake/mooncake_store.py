# Standard
import os

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.utils import get_ip, logger

from vllm_ascend.distributed.mooncake.config_data import MooncakeEngineKey
from vllm_ascend.distributed.mooncake.transfer_engine import get_global_te

from .config_data import MooncakeStoreConfig

METADATA_BYTES_LEN = 24
BASE_PORT = int(os.getenv("VLLM_BASE_PORT", "8790"))


class Mooncakestore():

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = parallel_config.tensor_parallel_size
        dp_rank = parallel_config.data_parallel_rank_local
        all_device_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
        if not all_device_ids:
            device_ids_list = list(
                range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        else:
            device_ids_list = list(map(int, all_device_ids.split(',')))
        assert len(device_ids_list) > tp_rank
        device_id = device_ids_list[tp_rank]
        self.config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()
        if self.config.protocol == "ascend" and not self.config.use_ascend_direct:
            local_hostname = get_ip() + ":" + str(BASE_PORT + int(device_id)) + \
                ":npu_" + str(device_id)
            ret = self.store.setup(local_hostname, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address)
        else:
            local_hostname = get_ip()
            transfer_engine = get_global_te(local_hostname, device_name=None)
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

    def exists(self, key: MooncakeEngineKey) -> bool:
        return self.store.is_exist(key.to_string()) == 1

    def batch_exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def register_buffer(self, ptr, length):
        return self.store.register_buffer(ptr, length)

    def get_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        try:
            res = self.store.batch_get_into_multi_buffers(
                keys, addrs, sizes, True)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {keys}. {e}")

    def put_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
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

    def get(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        expect_res = sum(size)
        key_str = key.to_string()
        try:
            res = self.store.batch_get_into_ascend(key_str, addr, size)
            if res[0] != expect_res:
                logger.error(f"Failed to get key: [{key_str}] .")
        except Exception:
            logger.error(f"Failed to get key: [{key_str}] .")
        return res

    def put(self, key: MooncakeEngineKey, addr: list[int], size: list[int]):
        key_str = key.to_string()
        try:
            ret = self.store.batch_put_from_ascend(key_str, addr, size)
            if ret[0] != 0:
                logger.error(f"Failed to put key {key_str}.")
        except Exception:
            logger.error(f"Failed to put key {key_str}.")

        return ret

    def close(self):
        self.store.close()
        logger.info("Closed the mooncake store connection")
