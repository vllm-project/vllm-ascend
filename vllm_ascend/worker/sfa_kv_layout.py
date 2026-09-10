# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy dense token-concat SFA cache views, independent of vLLM/NPU.

The supported parent is ND (blocks, tokens, 1, NoPE + RoPE), FP16/BF16.
Padding belongs to the allocator, not to the registered logical parent.
"""

import math

import torch


def split_sfa_kv_parent(
    raw: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
    nope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an exact logical raw region; never copy or include padding.

    ``shape`` must already incorporate the validated manager/kernel block
    conversion. A larger underlying allocation and aligned nonzero offset are
    allowed, but raw itself must contain exactly the dense parent's bytes.
    """
    if raw.layout != torch.strided or raw.dtype != torch.int8 or raw.ndim != 1 or raw.stride() != (1,):
        raise ValueError("SFA raw must be a contiguous flat int8 tensor")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SFA token-concat requires unquantized FP16 or BF16")
    if (
        len(shape) != 4
        or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in shape)
        or shape[2] != 1
        or not isinstance(nope_dim, int)
        or isinstance(nope_dim, bool)
        or not 0 < nope_dim < shape[3]
    ):
        raise ValueError("Unsupported SFA head geometry")
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    if raw.storage_offset() % dtype_bytes:
        raise ValueError("SFA raw storage offset must be aligned to the cache dtype")
    if raw.numel() != math.prod(shape) * dtype_bytes:
        raise ValueError("SFA raw bytes must match the dense parent shape exactly; padded layouts are unsupported")
    parent = raw.view(dtype).view(shape)
    return parent[..., :nope_dim], parent[..., nope_dim:]


def get_sfa_kv_parent(nope: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Validate component views and reconstruct their dense parent without copy.

    Independent legacy caches and padded/interleaved pages are rejected. The
    returned tensor retains storage ownership. Bounds are checked against the
    storage; original raw-slice boundaries cannot be inferred from arbitrary
    views, so allocation callers must use ``split_sfa_kv_parent`` to validate
    their exact logical raw region first.
    """
    if nope.layout != torch.strided or rope.layout != torch.strided:
        raise ValueError("SFA components must use strided ND storage")
    if nope.device != rope.device or nope.dtype != rope.dtype:
        raise ValueError("SFA components must have the same device and dtype")
    if nope.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SFA token-concat requires unquantized FP16 or BF16")
    if (
        nope.ndim != 4
        or rope.ndim != 4
        or nope.shape[:-1] != rope.shape[:-1]
        or nope.shape[2] != 1
        or any(d <= 0 for d in (*nope.shape, rope.shape[-1]))
    ):
        raise ValueError("Unsupported SFA component geometry")
    if nope.device.type == "meta":
        raise ValueError("SFA parent reconstruction requires physical storage")
    if nope.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr():
        raise ValueError("SFA components must share one storage")
    width = nope.shape[-1] + rope.shape[-1]
    shape = (*nope.shape[:-1], width)
    strides = (shape[1] * width, width, width, 1)
    if nope.stride() != strides or rope.stride() != strides:
        raise ValueError("SFA components must have dense token-concat parent strides; padding is unsupported")
    if rope.storage_offset() != nope.storage_offset() + nope.shape[-1]:
        raise ValueError("SFA RoPE offset must immediately follow NoPE within each token")
    end_bytes = (nope.storage_offset() + math.prod(shape)) * nope.element_size()
    if nope.storage_offset() < 0 or end_bytes > min(nope.untyped_storage().nbytes(), rope.untyped_storage().nbytes()):
        raise ValueError("SFA parent exceeds the shared storage bounds")
    return torch.as_strided(nope, shape, strides, storage_offset=nope.storage_offset())


def should_use_sfa_kv_parent_layout(kv_transfer_config: object | None) -> bool:
    """Decide whether unquantized SFA main KV uses the token-concat parent layout.

    Local inference with no KV transfer uses this layout. With transfer
    configured, only the native route-A connector is adapted; unadapted
    connectors (including MultiConnector and external implementations
    reusing the native name) keep the established separate contiguous layout.
    This is a storage-layout policy, not a statement about operator compatibility.
    """
    if kv_transfer_config is None:
        return True
    return getattr(kv_transfer_config, "kv_connector", None) == "SfaRemoteD2HConnector" and getattr(
        kv_transfer_config, "kv_connector_module_path", None
    ) in (None, "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector")
