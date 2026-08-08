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

    def put_tensor_single(self, key: str, addr: int, size: int):
        try:
            assert self.store is not None
            self.store.put_from(key, addr, size)
        except Exception as err:
            logger.error("Failed to put tensor into Mooncake Store: %s", err)
            raise

    def get_tensor_single(self, key: str, addr: int, size: int):
        try:
            assert self.store is not None
            self.store.get_into(key, addr, size)
        except Exception as err:
            logger.error("Failed to get tensor from Monncake Store: %s", err)
            raise

    def exist_single(self, key: str):
        assert self.store is not None
        return self.store.is_exist(key)

    def register_buffer_single(self, addr: int, length: int):
        assert self.store is not None
        ret = self.store.register_buffer(addr, length)
        if ret != 0:
            logger.error("Failed to register buffer for Mooncake Store: %s", ret)
            raise
