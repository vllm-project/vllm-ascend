from abc import ABC, abstractmethod

from vllm.config import ParallelConfig


class Backend(ABC):
    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig):
        pass

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    def register_additional_buffer(self, ptrs: list[int], lengths: list[int]):
        """Register buffers allocated after the backend's initial registration."""
        self.register_buffer(ptrs, lengths)

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass
