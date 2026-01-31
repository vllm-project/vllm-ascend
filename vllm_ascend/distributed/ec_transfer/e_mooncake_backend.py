import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from vllm.config import ParallelConfig
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import MooncakeBackend

logger = init_logger(__name__)


class EMooncakeBackend(MooncakeBackend):
    def __init__(self, parallel_config: ParallelConfig):
        super().__init__(parallel_config=parallel_config)

    def put_single(self, key: str, value: torch.Tensor | None) -> None:
        if value is not None:
            """Put KVCache to Mooncake Store"""
            value_bytes = safetensors_save(
                {
                    "tensor": value,
                }
            )
            try:
                self.store.put(key, value_bytes)
            except TypeError as err:
                logger.error("Failed to put value into Mooncake Store: %s", err)
                raise TypeError("Mooncake Store Put Type Error.") from err

    def get_single(self, key: str) -> torch.Tensor | None:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err

        if data:
            loaded_tensors = safetensors_load(data)
            tensor = loaded_tensors["tensor"]
            return tensor

        return None

    def exist_single(self, key: str):
        return self.store.is_exist(key)
