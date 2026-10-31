# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend compatibility kernel for sampling-mask packing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

_BLOCK_SIZE = 4096


@triton.jit
def _pack_sampling_mask_kernel_ascend(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    cu_num_logits_ptr,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack the finite-logit support of one request-major output slot.

    Output slot ``i`` belongs to request ``i // ROWS_PER_REQUEST`` and reads
    logits row ``cu_num_logits[req] + i % ROWS_PER_REQUEST``. Slots beyond the
    request's sampled-token count are inactive and produce an empty mask.
    """
    output_row = tl.program_id(0)
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    is_active = slot_idx < tl.load(num_sampled_tokens_ptr + req_idx)
    # Inactive slots may point past the last logits row; clamp them to the
    # request's first row and rely on `is_active` to clear the result. This
    # keeps control flow uniform, which Triton-Ascend lowers more reliably
    # than an early return.
    source_row = tl.load(cu_num_logits_ptr + req_idx) + tl.where(is_active, slot_idx, 0)
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=valid & is_active,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        # Triton-Ascend reduces an i1 tensor as a boolean. Cast before the sum
        # so counts contains the number of finite logits, not the tile count.
        count += tl.sum(keep_i32)

        keep = tl.trans(tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)))
        bit_weights = (1 << tl.arange(0, 8))[:, None]
        packed = tl.sum(keep * bit_weights, axis=0).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + output_row * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + output_row, count)


def _launch_pack_kernel(
    logits: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    rows_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_output_rows = num_sampled_tokens.shape[0] * rows_per_request
    vocab_size = logits.shape[1]
    packed_mask = torch.empty(
        (num_output_rows, (vocab_size + 7) // 8),
        dtype=torch.uint8,
        device=logits.device,
    )
    counts = torch.empty(num_output_rows, dtype=torch.int32, device=logits.device)
    if num_output_rows == 0:
        return packed_mask, counts
    _pack_sampling_mask_kernel_ascend[(num_output_rows,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        cu_num_logits,
        num_sampled_tokens,
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        ROWS_PER_REQUEST=rows_per_request,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return packed_mask, counts


def _warn_compact_ids_unsupported() -> None:
    logger.warning_once(
        "The compact token-ID path of SamplingMaskTensors is temporarily "
        "unsupported on Ascend; falling back to packed-mask decoding. Sampling "
        "results remain unchanged, but CPU decoding may be slower.",
        scope="process",
    )


def _bind(args: tuple, kwargs: dict, names: tuple[str, ...], defaults: dict) -> dict:
    """Bind positional and keyword arguments the way a Python signature would."""
    if len(args) > len(names):
        raise TypeError(f"from_logits() takes at most {len(names) + 1} positional arguments, got {len(args) + 1}")
    positional = dict(zip(names, args))
    params = {**defaults, **positional}
    for key, value in kwargs.items():
        if key not in names:
            raise TypeError(f"from_logits() got an unexpected keyword argument '{key}'")
        if key in positional:
            raise TypeError(f"from_logits() got multiple values for argument '{key}'")
        params[key] = value
    missing = [name for name in names if name not in params]
    if missing:
        raise TypeError(f"from_logits() missing required argument(s): {', '.join(missing)}")
    return params


def sampling_mask_from_logits_npu(
    cls: type[SamplingMaskTensors],
    logits: torch.Tensor,
    *args,
    **kwargs,
) -> SamplingMaskTensors:
    """Ascend replacement for ``SamplingMaskTensors.from_logits``.

    Supports every upstream layout of the API:

    * three fields: ``from_logits(logits, num_sampled_tokens)``
    * four fields: ``from_logits(logits, num_sampled_tokens, max_num_kept)``
    * five fields (vLLM #59359): ``from_logits(logits, cu_num_logits,
      num_sampled_tokens, max_num_kept, rows_per_request=1)``

    Compact token IDs are never produced: ``token_ids`` is zero-width so the
    upstream ``tolists()`` decodes every non-empty row from the exact bitmask.
    This avoids the cumsum-based dynamic scatter that hangs on TA 3.2.2.
    """
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    fields = cls._fields
    vocab_size = logits.shape[1]

    if "rows_per_request" in fields:
        params = _bind(
            args,
            kwargs,
            ("cu_num_logits", "num_sampled_tokens", "max_num_kept", "rows_per_request"),
            defaults={"rows_per_request": 1},
        )
        rows_per_request = int(params["rows_per_request"])
        packed_mask, counts = _launch_pack_kernel(
            logits, params["cu_num_logits"], params["num_sampled_tokens"], rows_per_request
        )
        _warn_compact_ids_unsupported()
        token_ids = torch.empty((counts.shape[0], 0), dtype=torch.int32, device=logits.device)
        return cls(token_ids, packed_mask, counts, vocab_size, rows_per_request)

    # Legacy layouts: one logits row per request, in request order.
    params = _bind(
        args,
        kwargs,
        ("num_sampled_tokens", "max_num_kept"),
        defaults={"max_num_kept": None},
    )
    cu_num_logits = torch.arange(logits.shape[0] + 1, dtype=torch.int32, device=logits.device)
    packed_mask, counts = _launch_pack_kernel(logits, cu_num_logits, params["num_sampled_tokens"], 1)
    if "token_ids" in fields:
        _warn_compact_ids_unsupported()
        token_ids = torch.empty((counts.shape[0], 0), dtype=torch.int32, device=logits.device)
        return cls(token_ids, packed_mask, counts, vocab_size)
    return cls(packed_mask, counts, vocab_size)
