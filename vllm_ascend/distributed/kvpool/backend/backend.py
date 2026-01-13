from abc import ABC, abstractmethod

from vllm.config import ParallelConfig


class Backend(ABC):
    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig):
        raise NotImplementedError

    @abstractmethod
    def set_device(self):
        raise NotImplementedError

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        raise NotImplementedError

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
