# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU streams, mapped host memory and DMA for the common HiSparse runtime."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.hisparse.backend import HiSparseBackend

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


class AscendHiSparseBackend(HiSparseBackend):
    def __init__(self, *, dma_lifetime: Any | None = None) -> None:
        from .backend_dma import HostDMALifetime

        self._host_dma = HostDMALifetime() if dma_lifetime is None else dma_lifetime

    # The common shared pool currently exchanges CUDA IPC events. Private
    # per-rank pools preserve the same scheduler and transfer completion rules.
    supports_shared_host_pool = False
    stats_dtype = torch.int64

    def check_available(self) -> None:
        from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import optimized_profile_enabled
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if not optimized_profile_enabled() or get_ascend_device_type() != AscendDeviceType.A5:
            raise RuntimeError("The optimized HiSparse backend requires its explicit paired A5 profile")
        from vllm_ascend.patch.worker.patch_hisparse_streams import install

        install()
        from vllm_ascend.ops.hisparse_optimized import HiSparseNPUOps
        from vllm_ascend.utils import bootstrap_custom_op_env

        if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
            # A5 disables the global custom-op selection policy. HiSparse still
            # needs its explicit DMA dependency; load the extension lazily here
            # without changing that policy for unrelated model operators.
            bootstrap_custom_op_env()
            try:
                importlib.import_module("vllm_ascend.vllm_ascend_C")
            except ImportError as error:
                raise RuntimeError("Ascend HiSparse requires the compiled Ascend DMA extension.") from error
            if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
                raise RuntimeError("Ascend HiSparse requires the Ascend swap_blocks_batch operator.")
        if not callable(HiSparseNPUOps):
            raise RuntimeError("Ascend HiSparse device kernels are unavailable.")

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm_ascend.ops.hisparse_optimized import MappedHostTensor

        tensor._hisparse_registration = MappedHostTensor(tensor)
        self._host_dma.retain_host(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        self._host_dma.drain()
        registration = tensor._hisparse_registration
        registration.close()
        self._host_dma.release_host(tensor)
        del tensor._hisparse_registration

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        from vllm_ascend.ops.hisparse_optimized import HiSparseNPUOps, ResolverWorkspace
        from vllm_ascend.patch.worker.patch_hisparse_invalidation import install

        install()
        group = runtime.index_group
        if runtime is group.leader:
            workspace = ResolverWorkspace(
                max_rows=runtime.max_swap_rows,
                hot_size=runtime.region_stride,
                max_union_rows=runtime.max_union_rows,
                device=runtime.device,
            )
        else:
            leader_ops = group.leader.cache_ops
            if not isinstance(leader_ops, HiSparseNPUOps):
                raise RuntimeError("Bind the HiSparse leader before follower caches.")
            workspace = leader_ops.workspace
        mapping = runtime.registered_host_pool._hisparse_registration
        ops = HiSparseNPUOps(mapping, workspace)
        ops.prefetch_gather_global_workers = 56
        from .runtime_adapter import bind_runtime

        bind_runtime(runtime)
        from .page_dma_descriptors_optimized import install as install_page_descriptors
        from .resident_update_installer import install as install_resident_updates

        install_page_descriptors()
        install_resident_updates()
        from vllm_ascend.patch.worker.patch_hisparse_recycled_groups import install as install_recycled

        install_recycled()
        if runtime.eager_host_mirror is False:
            from vllm.config import get_current_vllm_config

            from vllm_ascend.patch.worker.patch_hisparse_lazy_mtp_mirror import install as install_lazy_mtp_mirror

            install_lazy_mtp_mirror(get_current_vllm_config())
        return ops

    def validate_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        from .binding_optimized import get_hisparse_indexer_specs

        get_hisparse_indexer_specs(kv_cache_config)

    def prepare_kv_cache_views(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
    ) -> None:
        from vllm_ascend.kv_cache_dtype import prepare_native_hisparse_view_dtypes

        from .binding_optimized import _bind_indexer_views, get_hisparse_indexer_specs

        prepare_native_hisparse_view_dtypes(forward_context, kv_caches)

        def device_tensors(value):
            if isinstance(value, torch.Tensor):
                if value.device.type == "npu":
                    yield value
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from device_tensors(item)

        self._host_dma.replace_device_owners(tensor for value in kv_caches.values() for tensor in device_tensors(value))
        _bind_indexer_views(kv_caches, get_hisparse_indexer_specs(kv_cache_config), forward_context)

    def gather_resident_rows(self, source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor) -> None:
        from vllm_ascend.ops.triton.hisparse_resident_gather import try_gather_resident_rows

        if try_gather_resident_rows(source, destination, row_ids):
            return
        if source.dtype == destination.dtype and source.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = source.view(torch.uint8)
            destination = destination.view(torch.uint8)
        super().gather_resident_rows(source, destination, row_ids)

    def create_ipc_event(self) -> Any:
        raise NotImplementedError("Ascend HiSparse requires private per-rank Host pools.")

    def export_event(self, event: Any) -> Any:
        raise NotImplementedError("Ascend HiSparse does not export NPU IPC events.")

    def import_event(self, device: torch.device, handle: Any) -> Any:
        raise NotImplementedError("Ascend HiSparse does not import NPU IPC events.")

    def create_stream(self, device: torch.device, *, generic: bool = False) -> Any:
        from vllm_ascend.patch.worker.patch_hisparse_streams import HiSparseStream

        return HiSparseStream(device=device)

    def create_lru_init(self, size: int, device: torch.device) -> torch.Tensor:
        # CANN Range does not support int16 outputs. This runs only at allocation.
        return torch.arange(size, dtype=torch.int32, device=device).to(torch.int16)

    def create_event(self) -> Any:
        from vllm_ascend.patch.worker.patch_hisparse_streams import create_event

        return create_event()

    def current_stream(self) -> Any:
        # HiSparse switches between model and copy streams. Query the active
        # stream here: the generic Ascend helper caches its first result.
        return torch.npu.current_stream()

    def stream_context(self, stream: Any, *, generic: bool = False) -> Any:
        return torch.npu.stream(stream)

    def is_current_stream_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def swap_blocks_batch(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor) -> None:
        self._host_dma.submit(src, dst, sizes)
