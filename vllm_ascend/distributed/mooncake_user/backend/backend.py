from abc import ABC, abstractmethod

from vllm.config import ParallelConfig


class Backend(ABC):

    def __init__(self, parallel_config: ParallelConfig):
        pass

    def set_device(self):
        pass

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, key: str) -> list[int]:
        pass

    @abstractmethod
    def put(self, key: str, addrs: list[int], sizes: list[int]):
        pass

    @abstractmethod
    def get(self, key: str, addrs: list[int], sizes: list[int]):
        pass

    @abstractmethod
    def get_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        pass

    @abstractmethod
    def put_batch(self, keys: list[str], addrs: list[list[int]],
                  sizes: list[list[int]], block_ids: list[int]):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_buffer(self, key: str):
        pass

    @abstractmethod
    def remove(self, key: str):
        pass

    @abstractmethod
    def put_from(self, key: str, addr: int, size: int):
        pass
