from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_ascend import envs


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
        self._host_side_backend = self._use_host_side_backend()

    def _use_host_side_backend(self) -> bool:
        backend = str(self.extra_config.get("backend", "")).lower()
        return (
            bool(self.extra_config.get("host_side", False))
            or backend in ("host_side", "host-side", "acl_host", "acl-host")
            or envs.VLLM_ASCEND_KV_HOST_SIDE
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
            assert len(self.gpu_block_size) == 1
            gpu_block_size = self.gpu_block_size[0]
            offloaded_block_size = gpu_block_size * self.block_size_factor
            self._manager = CPUOffloadingManager(
                block_size=offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            if self._host_side_backend:
                from vllm_ascend.kv_offload.shmem_host import HostSideOffloadingHandler

                max_host_memory_bytes = self.extra_config.get(
                    "host_side_max_memory_bytes",
                    envs.VLLM_ASCEND_KV_HOST_SIDE_MAX_MEMORY_BYTES,
                )
                if max_host_memory_bytes is not None:
                    max_host_memory_bytes = int(max_host_memory_bytes)
                alloc_chunk_blocks = int(self.extra_config.get("host_side_alloc_chunk_blocks", 256))
                self._handler = HostSideOffloadingHandler(
                    block_size_factor=self.block_size_factor,
                    num_cpu_blocks=self.num_cpu_blocks,
                    gpu_caches=kv_caches,
                    max_host_memory_bytes=max_host_memory_bytes,
                    alloc_chunk_blocks=alloc_chunk_blocks,
                )
            else:
                raise NotImplementedError(
                    "NPUOffloadingSpec requires host-side backend with the "
                    "current vLLM CanonicalKVCaches offload API. Set "
                    "kv_connector_extra_config.host_side=true or "
                    "VLLM_ASCEND_KV_HOST_SIDE=1."
                )

        assert self._handler is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handler