# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Own the HiSparse padding mask outside ACL graph pools.

Detached candidate: importing this module does not activate it. The explicit
backend activation allocates one bounded mask per index group, before capture.
The common method below is source-pinned and differs only at the GE expression.
Remove this adapter when common provides a backend metadata-workspace hook.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import in_piecewise_cudagraph

from vllm_ascend.patch.worker.patch_hisparse_streams import create_event, native_current_stream

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle

COMMON_RUNTIME_SHA256 = "e63b6a9abf25754e9eef10b8a90a3d3ed1e320d16924e0a1052487ab4a4cbc46"


@dataclass(frozen=True)
class PaddingMaskOwner:
    rows: torch.Tensor
    request_ids: torch.Tensor
    copy_stream: object
    mask: torch.Tensor


def _prepare_group_owner(group) -> PaddingMaskOwner:
    """Cold only; reject rebinding rather than invalidate captured addresses."""
    rows = group.row_indices
    request_ids = group.request_ids
    if rows.dtype != torch.int32 or rows.ndim != 1 or not rows.is_contiguous():
        raise ValueError("HiSparse padding rows require contiguous int32 rank-one storage")
    if request_ids.dtype != rows.dtype or request_ids.shape != rows.shape or request_ids.device != rows.device:
        raise ValueError("HiSparse persistent request-id owner differs from row capacity")
    owner = getattr(group, "_ascend_padding_mask_owner", None)
    if owner is None:
        owner = PaddingMaskOwner(rows, request_ids, group.copy_stream, torch.empty_like(rows, dtype=torch.bool))
        group._ascend_padding_mask_owner = owner
    elif owner.rows is not rows or owner.request_ids is not request_ids or owner.copy_stream is not group.copy_stream:
        raise RuntimeError("Recreate HiSparse group after draining old graphs before rebinding metadata storage")
    return owner


def _padding_mask(group, num_tokens: int, num_valid_rows: torch.Tensor) -> torch.Tensor:
    owner = group._ascend_padding_mask_owner
    if owner.rows is not group.row_indices or owner.request_ids is not group.request_ids:
        raise RuntimeError("HiSparse padding metadata storage changed after capture")
    if owner.copy_stream is not group.copy_stream or not 0 <= num_tokens <= owner.mask.numel():
        raise RuntimeError("HiSparse padding owner stream/capacity changed")
    mask = owner.mask[:num_tokens]
    torch.ge(group.row_indices[:num_tokens], num_valid_rows, out=mask)
    return mask


def prepare_group(runtime) -> None:
    """Explicit NPU-backend activation, called during Host-cache binding."""
    if runtime.device.type != "npu":
        return
    if getattr(runtime, "_ascend_runtime_backend_bound", False):
        return
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Create HiSparse padding owner before graph capture")
    group = runtime.index_group
    if group.row_indices.device != runtime.device:
        raise ValueError("HiSparse padding rows must reside on the runtime NPU")
    _prepare_group_owner(group)
    install()


def install() -> None:
    from vllm.v1.hisparse.runtime import HiSparseRuntime

    original = HiSparseRuntime._resolve_and_stage_group
    if getattr(original, "_ascend_padding_mask_owned", False):
        return
    path = Path(inspect.getsourcefile(original))
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != COMMON_RUNTIME_SHA256:
        raise RuntimeError("Unreviewed common HiSparse runtime; padding-owner patch was not installed")

    @wraps(original)
    def resolve(self, *args, **kwargs):
        if self.device.type != "npu":
            return original(self, *args, **kwargs)
        return _owned_resolve_and_stage_group(self, *args, **kwargs)

    resolve._ascend_padding_mask_owned = True
    resolve._ascend_original = original
    HiSparseRuntime._resolve_and_stage_group = resolve


def _owned_resolve_and_stage_group(
    self,
    *,
    resident: HiSparseCacheHandle,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    logical_topk_indices: torch.Tensor,
    block_size: int,
    return_valid_counts: bool,
    shared_rows: slice,
    attention_indices_out: torch.Tensor | None,
    valid_counts_out: torch.Tensor | None,
    num_valid_rows: torch.Tensor,
) -> None:
    group = self.index_group
    compute_stream = native_current_stream()
    if group.logical_topk_ready is not None and not in_piecewise_cudagraph():
        group.copy_stream.wait_event(group.logical_topk_ready)
    else:
        group.copy_stream.wait_stream(compute_stream)
    with group.copy_stream:
        # CUDA-graph padding rows past the batch's tokens map to request 0;
        # mark them -1 so residency resolution skips them.
        num_tokens = logical_topk_indices.shape[0]
        request_ids = group.request_ids[:num_tokens]
        request_ids.copy_(req_id_per_token)
        request_ids.masked_fill_(_padding_mask(group, num_tokens, num_valid_rows), -1)
        self._resolve_residency(
            resident=resident,
            req_id_per_token=request_ids,
            block_table=block_table,
            topk_indices=logical_topk_indices,
            block_size=block_size,
            return_valid_counts=return_valid_counts,
            shared_rows=shared_rows,
            attention_indices_out=attention_indices_out,
            valid_counts_out=valid_counts_out,
        )
        runtimes = [self]
        if resident.decode_batch:
            runtimes.extend(group.followers)
        for runtime in runtimes:
            runtime._swap_rows(shared_rows)
            if runtime._layer_ready_event is None:
                runtime._layer_ready_event = create_event()
            runtime._layer_ready_event.record(group.copy_stream)
            runtime._swap_staged = True
