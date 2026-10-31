# mypy: ignore-errors
"""NPU implementation of vLLM's KVTransferBackend (transfer contract v0).

A's kv-hints branch adds this seam so a platform can supply its own block-copy
path. vllm-ascend's existing NPUOffloadingWorker does not go through it, but the
seam is required when the upstream CPUOffloadingWorker is used on NPU.
"""

from __future__ import annotations

import torch
from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.transfer_backend import (
    KVTransferBackend,
    register_kv_transfer_backend,
)

_DIRECTION_H2D = 0
_DIRECTION_D2H = 1


class NPUKVTransferBackend(KVTransferBackend):
    """KV block copies through Ascend's swap_blocks_batch op."""

    def select_copy_op(
        self,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        host_memory_is_pinned: bool = True,
    ):
        direction = _DIRECTION_D2H if gpu_to_cpu else _DIRECTION_H2D

        def copy_op(src, dst, sizes, is_src_access_order_any: bool = False) -> None:
            # src/dst are 1-D int64 pointer tensors, sizes byte counts.
            torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, direction)

        return copy_op

    def descriptor_ptr_dtype(self) -> torch.dtype:
        return torch.int64

    def validate_device_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.device.type == "npu", f"expected npu tensor, got {tensor.device}"

    def host_register(self, ptr: int, num_bytes: int) -> bool:
        # Ascend has no cudaHostRegister equivalent; pageable host copies work.
        return False

    def host_unregister(self, ptr: int) -> None:
        return None


def register_npu_kv_transfer_backend() -> None:
    register_kv_transfer_backend("npu", NPUKVTransferBackend())
