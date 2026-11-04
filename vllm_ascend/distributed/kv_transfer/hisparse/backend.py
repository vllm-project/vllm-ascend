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

    def swap_blocks_batch(self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor) -> None:
        torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
