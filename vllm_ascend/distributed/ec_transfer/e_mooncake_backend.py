from vllm.config import ParallelConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend


class EMooncakeBackend(MooncakeBackend):
    def __init__(self, parallel_config: ParallelConfig):
        super().__init__(parallel_config=parallel_config)

    def put_tensor_info(self, key: str, value: bytes | None):
        if self.store is not None and value is not None:
            """Put value to Mooncake Store"""
            try:
                self.store.put(key, value)
            except TypeError as err:
                logger.error("Failed to put value into Mooncake Store: %s", err)
                raise TypeError("Mooncake Store Put Type Error.") from err

    def get_tensor_info(self, key: str) -> bytes | None:
        if self.store is not None:
            """Get value from Mooncake Store"""
            try:
                data = self.store.get(key)
            except TypeError as err:
                logger.error("Failed to get value from Mooncake Store: %s", err)
                raise TypeError("Mooncake Store Get Type Error.") from err
            return data

        return None
