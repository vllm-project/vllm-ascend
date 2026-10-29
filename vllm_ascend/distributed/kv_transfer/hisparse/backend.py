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
    # NPU event IPC is not implemented here. Keep private pools per TP rank;
    # the common connector still owns transfer completion and TP coordination.
    supports_shared_host_pool = False
    supports_speculative_group_prefetch = True
    stats_dtype = torch.int64

    def check_available(self) -> None:
        from vllm_ascend.ops.hisparse import HiSparseNPUOps
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
        from vllm_ascend.ops.hisparse import MappedHostTensor

        tensor._hisparse_registration = MappedHostTensor(tensor)

    def unpin_tensor(self, tensor: torch.Tensor) -> None:
        registration = tensor._hisparse_registration
        registration.close()
        del tensor._hisparse_registration

    def create_cache_ops(self, runtime: HiSparseRuntime) -> Any:
        from vllm_ascend.ops.hisparse import HiSparseNPUOps, ResolverWorkspace

        group = runtime.index_group
        if group.backend_state is None:
            group.backend_state = ResolverWorkspace(
                max_rows=runtime.max_swap_rows,
                hot_size=runtime.region_stride,
                max_union_rows=runtime.max_union_rows,
                device=runtime.device,
            )
        mapping = runtime.registered_host_pool._hisparse_registration
        return HiSparseNPUOps(mapping, group.backend_state)

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

    def create_stream(self, device: torch.device, *, generic: bool = False) -> Any:
        return torch.npu.Stream(device=device)

    def create_lru_init(self, size: int, device: torch.device) -> torch.Tensor:
        # CANN Range does not support int16 outputs. This runs only at allocation.
        return torch.arange(size, dtype=torch.int32, device=device).to(torch.int16)

    def create_event(self, *, interprocess: bool = False) -> Any:
        if interprocess:
            raise NotImplementedError(
                "Ascend HiSparse requires private per-rank host pools; NPU event IPC is unavailable."
            )
        return torch.npu.Event()

    def export_event(self, event: Any) -> Any:
        raise NotImplementedError("Ascend HiSparse does not export NPU IPC events.")

    def import_event(self, device: torch.device, handle: Any) -> Any:
        raise NotImplementedError("Ascend HiSparse does not import NPU IPC events.")

    def gather_resident_rows(self, source: torch.Tensor, destination: torch.Tensor, row_ids: torch.Tensor) -> None:
        """Copy packed rows as bytes, including RoPE and scale payloads in C8.

        The source may have gaps between physical pages. Reinterpret FP8 views
        before indexing because NPU advanced indexing does not support FP8.
        No quantization, dequantization or host scalar reads occur here.
        """
        if source.ndim != 3 or destination.ndim != 2 or row_ids.ndim != 1:
            raise ValueError("HiSparse resident gather expects paged source, flat destination and one row-ID vector.")
        if source.device != destination.device or source.device != row_ids.device:
            raise ValueError("HiSparse resident gather tensors must share one device.")
        if destination.shape[0] != row_ids.numel() or (
            source.shape[-1] * source.element_size() != destination.shape[-1] * destination.element_size()
        ):
            raise ValueError("HiSparse resident gather requires matching row counts and payload byte widths.")
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        if source.dtype in fp8_dtypes:
            source = source.view(torch.uint8)
        if destination.dtype in fp8_dtypes:
            destination = destination.view(torch.uint8)
        rows = row_ids.to(torch.long).clamp_min(0)
        block_size = source.shape[1]
        payload = source[rows // block_size, rows % block_size]
        if payload.dtype != destination.dtype:
            payload = payload.contiguous().view(destination.dtype)
        destination.copy_(payload)

    def gather_fp8_prefill(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        block_table: torch.Tensor,
        workspace_starts: torch.Tensor,
        batch_size: int,
        *,
        host_cache: torch.Tensor,
        host_row_ids: torch.Tensor,
        device_row_ids: torch.Tensor,
    ) -> None:
        # Native SFA stages opaque rows via gather_prefill_cache and consumes
        # the Ascend C8 layout directly; CUDA fp8_ds_mla has a different layout.
        raise NotImplementedError(
            "Ascend HiSparse uses native SFA prefill with packed KV rows; "
            "the CUDA fp8_ds_mla prefill conversion path is unsupported."
        )

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
        torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
