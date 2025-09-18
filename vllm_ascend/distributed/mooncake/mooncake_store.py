# Standard
import os

# Third Party
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.utils import logger

from vllm_ascend.distributed.mooncake.config_data import MooncakeEngineKey

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
        if self.config.protocol == "ascend":
            local_hostname = self.config.local_hostname + ":" + str(BASE_PORT + int(device_id)) + \
                ":npu_" + str(device_id)
        else:
            local_hostname = self.config.local_hostname
        self.store = MooncakeDistributedStore()
        ret = self.store.setup(local_hostname, self.config.metadata_server,
                               self.config.global_segment_size,
                               self.config.local_buffer_size,
                               self.config.protocol, self.config.device_name,
                               self.config.master_server_address)
        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

    def set_kv_caches(self, kvcache):
        self.kvcache = list(kvcache)

    def exists(self, key: MooncakeEngineKey) -> bool:
        return self.store.is_exist(key.to_string()) == 1

    def batch_exists(self, keys: list[str]) -> list[bool]:
        return self.store.batch_is_exist(keys)

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