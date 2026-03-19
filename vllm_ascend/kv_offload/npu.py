from collections.abc import Iterator

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend  # type: ignore
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend.kv_offload.cpu_npu import CpuNpuOffloadingHandler
from vllm_ascend.utils import vllm_version_is


class NPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, kv_cache_config)

        num_cpu_blocks = self.extra_config.get("num_cpu_blocks")
        if not num_cpu_blocks:
            raise Exception("num_cpu_blocks must be specified in kv_connector_extra_config")
        self.num_cpu_blocks: int = num_cpu_blocks

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            backend = None
            if vllm_version_is("0.17.0"):
                kv_events_config = self.vllm_config.kv_events_config
                enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
                backend = CPUBackend(block_size=self.offloaded_block_size, num_blocks=self.num_cpu_blocks)
            else:
                kv_events_config = self.vllm_config.kv_events_config
                enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
                assert len(self.gpu_block_size) == 1
                gpu_block_size = self.gpu_block_size[0]
                offloaded_block_size = gpu_block_size * self.block_size_factor
                backend = CPUBackend(block_size=offloaded_block_size, num_blocks=self.num_cpu_blocks)

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(backend=backend, enable_events=enable_events)
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(backend=backend, enable_events=enable_events)
            else:
                raise ValueError(f"Unknown eviction policy: {self.eviction_policy}. Supported policies: lru, arc")

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            if vllm_version_is("0.17.0"):
                self._handler = CpuNpuOffloadingHandler(
                    attn_backends=attn_backends,
                    gpu_block_size=self.gpu_block_size,
                    cpu_block_size=self.offloaded_block_size,
                    num_cpu_blocks=self.num_cpu_blocks,
                    gpu_caches=kv_caches,
                )
            else:
                assert len(self.gpu_block_size) == 1
                gpu_block_size = self.gpu_block_size[0]
                self._handler = CpuNpuOffloadingHandler(
                    attn_backends=attn_backends,
                    gpu_block_size=gpu_block_size,
                    cpu_block_size=gpu_block_size * self.block_size_factor,
                    num_cpu_blocks=self.num_cpu_blocks,
                    gpu_caches=kv_caches,
                )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler
