# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Core HiSparse backend: NPU streams, mapped Host memory and DMA."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.hisparse.backend import HiSparseBackend

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


class _CaptureAwareStream(torch.Stream):
    """Propagate a captured dependency before PTA allocates CANN workspace.

    PTA's native NPUEvent wait drains Host submission queues during capture;
    its generic event binding does not. Keep generic event points and stream
    identity, adding the missing submission boundary only during capture.
    Remove when the supported PTA generic binding has the same behavior.
    """

    def _finish_capture_propagation(self) -> None:
        if torch.npu.is_current_stream_capturing():
            native = torch.npu.Stream(
                stream_id=self.stream_id, device_index=self.device_index, device_type=self.device_type
            )
            # Obtaining the native handle drains its Host task queue. This is
            # not a device synchronization and does not run during graph replay.
            _ = native.npu_stream

    def wait_event(self, event) -> None:
        super().wait_event(event)
        self._finish_capture_propagation()

    def wait_stream(self, stream) -> None:
        super().wait_stream(stream)
        self._finish_capture_propagation()


def capture_aware_stream(stream: torch.Stream) -> torch.Stream:
    if stream.device.type != "npu" or isinstance(stream, _CaptureAwareStream):
        return stream
    return _CaptureAwareStream(stream.stream_id, stream.device_index, stream.device_type)


class AscendHiSparseBackend(HiSparseBackend):
    # Inherit Core's private Host-pool policy; NPU event IPC is not implemented.
    stats_dtype = torch.int64

    def check_available(self) -> None:
        from vllm_ascend.patch.worker.patch_hisparse_host_pool import install
        from vllm_ascend.utils import bootstrap_custom_op_env

        install()
        # Import the device kernels before the connector allocates Host storage.
        importlib.import_module("vllm_ascend.hisparse.cache_ops")
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

    def allocate_private_host_pool(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_ascend.hisparse.cache_ops import allocate_mapped_host

        payload = allocate_mapped_host((size,), torch.int8)
        registration = payload._hisparse_registration
        registered = registration.tensor.view(torch.int8)
        registered._hisparse_registration = registration
        return registered[:size], registered

    def pin_tensor(self, tensor: torch.Tensor) -> None:
        from vllm_ascend.hisparse.cache_ops import MappedHostTensor

        tensor._hisparse_registration = MappedHostTensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        registration = tensor._hisparse_registration
        registration.close()
        del tensor._hisparse_registration

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps, ResolverWorkspace

        group = runtime.index_group
        # Runtime copied the model group's stream reference at layer creation.
        # Adapt the live runtime binding before any graph captures use it.
        group.copy_stream = capture_aware_stream(group.copy_stream)
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
        return HiSparseNPUOps(mapping, workspace)

    def validate_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        from .binding import get_hisparse_indexer_specs

        get_hisparse_indexer_specs(kv_cache_config)

    def prepare_kv_cache_views(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
        forward_context: dict[str, Any],
    ) -> None:
        from .binding import _bind_indexer_views, get_hisparse_indexer_specs

        _bind_indexer_views(kv_caches, get_hisparse_indexer_specs(kv_cache_config), forward_context)

    def is_current_stream_capturing(self) -> bool:
        return torch.npu.is_current_stream_capturing()

    def gather_resident_rows(self, source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor) -> None:
        # NPU advanced indexing does not support FP8. Copy all packed components
        # as bytes, retaining Core's addressing and completion ownership.
        fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
        if source.dtype in fp8:
            source = source.view(torch.uint8)
        if destination.dtype in fp8:
            destination = destination.view(torch.uint8)
        super().gather_resident_rows(source, destination, row_ids)

    def swap_blocks_batch(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor) -> None:
        torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
