# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 INT8 CSA packed-cache and mixed-quant attention ABI.

The A3 operator consumes one 608-byte record per cache token:

    [0, 128)     64 BF16 RoPE values
    [128, 576)   448 signed INT8 nope values
    [576, 590)   7 BF16 scales (one per 64 nope values)
    [590, 608)   padding

A3 does not provide ``kv_compress_epilog``, so this module owns both the
quantization/packing producer and the cache write.  The attention call uses
``mixed_quant_sparse_flash_mla`` with ``quant_mode=1`` and INT8 cache tensors.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

try:
    import cann_ops_transformer  # noqa: F401
except ImportError as exc:  # pragma: no cover - configuration error
    raise RuntimeError(
        "A3 INT8 CSA requires a CANN 9.3 cann_ops_transformer build with "
        "INT8 mixed_quant_sparse_flash_mla support"
    ) from exc


NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
KV_RECORD_BYTES = 608
QUANT_GROUP_SIZE = 64
NOPE_GROUPS = NOPE_HEAD_DIM // QUANT_GROUP_SIZE
METADATA_SIZE = 2048
def _pack_int8_kv(values: torch.Tensor) -> torch.Tensor:
    """Quantize one RoPE-applied latent row set into 608-byte INT8 records."""
    if values.ndim != 2 or values.shape[-1] != NOPE_HEAD_DIM + ROPE_HEAD_DIM:
        raise ValueError(f"A3 INT8 CSA expects [tokens, 512], got {tuple(values.shape)}")
    tokens = values.shape[0]
    nope = values[:, :NOPE_HEAD_DIM].float().contiguous()
    rope = values[:, NOPE_HEAD_DIM:].to(torch.bfloat16).contiguous()
    groups = nope.view(tokens, NOPE_GROUPS, QUANT_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (amax / 127.0).to(torch.bfloat16)
    quantized = (
        torch.round(groups / scale.float().unsqueeze(-1))
        .clamp(-127, 127)
        .to(torch.int8)
    )

    packed = torch.zeros((tokens, KV_RECORD_BYTES), dtype=torch.int8, device=values.device)
    packed_u8 = packed.view(torch.uint8)
    packed_bf16 = packed.view(torch.bfloat16)
    packed_bf16[:, :ROPE_HEAD_DIM] = rope
    packed_u8[:, 128:576] = quantized.reshape(tokens, NOPE_HEAD_DIM).view(torch.uint8)
    packed_bf16[:, 288:295] = scale
    return packed


@triton.jit(do_not_specialize=["tokens", "width"])
def _write_packed_cache(
    cache,
    updates,
    slots,
    tokens,
    width,
    pages: tl.constexpr,
    page_size: tl.constexpr,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    update_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    paired: tl.constexpr,
    columns: tl.constexpr,
):
    columns_idx = tl.arange(0, columns)
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        first = tl.load(slots + token * slot_stride).to(tl.int64)
        if paired:
            block = first
            offset = tl.load(slots + token * slot_stride + 1).to(tl.int64)
        else:
            block = first // page_size
            offset = first % page_size
        if (first >= 0) & (block < pages) & (offset >= 0) & (offset < page_size):
            value = tl.load(updates + token * update_stride + columns_idx, columns_idx < width, other=0)
            address = block * page_stride + offset * row_stride + columns_idx
            tl.store(cache + address, value, columns_idx < width)


def _write_packed_cache_tensor(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    """Write packed rows while preserving the allocator's physical-page stride."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.stride(-1) != 1:
        raise ValueError("Cache must be [pages, page_size, 1, width] with contiguous rows")
    if updates.ndim != 2 or updates.shape[-1] != cache.shape[-1]:
        raise ValueError("Packed updates must be [tokens, cache_width]")
    if updates.dtype != cache.dtype or updates.stride(-1) != 1:
        raise ValueError("Packed updates must match the cache dtype and be row-contiguous")
    if slots.ndim != 1 or slots.shape[0] != updates.shape[0]:
        raise ValueError("Slots must be one-dimensional with one entry per update row")
    tokens = updates.shape[0]
    if tokens == 0:
        return
    _write_packed_cache[(min(get_vectorcore_num(), tokens),)](
        cache,
        updates,
        slots,
        tokens,
        cache.shape[-1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        updates.stride(0),
        slots.stride(0),
        False,
        triton.next_power_of_2(cache.shape[-1]),
        multibuffer=False,
    )


def write_attention_cache(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    """Pack one SWA/compressed latent batch and scatter it into the INT8 cache."""
    if kind not in ("win", "cmp"):
        raise ValueError(f"unsupported A3 INT8 cache kind: {kind}")
    packed = _pack_int8_kv(values)
    slots = flat_slots[: packed.shape[0]].contiguous()
    _write_packed_cache_tensor(cache, packed, slots)


def _pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    if indices.shape[-1] < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices[:, None, :].to(torch.int32).contiguous()


def qsmla(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None,
    metadata,
    compressed_indices: torch.Tensor,
    *,
    window_size: int,
    sinks: torch.Tensor | None,
    softmax_scale: float,
    compressed_lengths: torch.Tensor | None = None,
    compress_ratio: int = 1,
    index_topk: int | None = None,
) -> torch.Tensor:
    """Run the A3 INT8 CSA operator for one layer."""
    if cmp_kv is None:
        raise ValueError("A3 INT8 CSA requires a compressed KV cache")
    topk = index_topk if index_topk is not None else compressed_indices.shape[-1]
    cmp_indices = _pad_sparse_indices(compressed_indices, topk)
    task_metadata = metadata.swa.smla_metadata
    if task_metadata is None:
        raise RuntimeError("A3 INT8 CSA metadata was not built")
    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(task_metadata))
    num_reqs = metadata.swa.num_reqs
    ori_block_table = metadata.swa.block_table[:num_reqs]
    ori_seq_lens = metadata.swa.seq_lens[:num_reqs]
    if metadata.swa.replay_block_table is not None:
        # Bounded replay keeps only the retained SWA prefix in the cache;
        # rebase the original KV view exactly as the bf16 fallback does.
        ori_kv = ori_kv[metadata.swa.replay_cache_blocks, metadata.swa.replay_cache_offsets]
        ori_kv = ori_kv.reshape(-1, metadata.swa.storage_block_size, 1, ori_kv.shape[-1])
        ori_block_table = metadata.swa.replay_block_table[:num_reqs]
        ori_seq_lens = metadata.swa.replay_seq_lens[:num_reqs]
    ori_mask_mode = metadata.swa.ori_mask_mode
    win_left = metadata.swa.ori_win_left if ori_mask_mode == 4 else -1
    win_right = metadata.swa.ori_win_right if ori_mask_mode == 4 else -1
    cmp_residual = metadata.attention.cmp_residual
    output, _ = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
        q,
        quant_mode=1,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=metadata.attention.block_table[:num_reqs],
        cu_seqlens_q=metadata.swa.query_start_loc[: num_reqs + 1],
        seqused_ori_kv=ori_seq_lens,
        seqused_cmp_kv=metadata.attention.cache_seq_lens[:num_reqs],
        cmp_residual_kv=(
            None if compress_ratio == 1 or cmp_residual is None else cmp_residual[:num_reqs]
        ),
        sinks=None if sinks is None else sinks.detach().float().contiguous(),
        metadata=task_metadata,
        rope_head_dim=ROPE_HEAD_DIM,
        softmax_scale=softmax_scale,
        cmp_ratio=compress_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=3,
        ori_win_left=win_left,
        ori_win_right=win_right,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return output


def build_smla_metadata(
    *,
    num_heads_q: int,
    head_dim: int,
    num_reqs: int,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    cmp_seq_lens: torch.Tensor,
    cmp_residual: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_ori_kv: int,
    max_seqlen_cmp_kv: int,
    index_topk: int,
    compress_ratio: int,
    ori_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
) -> torch.Tensor:
    """Build the fixed 2048-int32 launch metadata for the A3 INT8 CSA path."""
    return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
        num_heads_q,
        1,
        head_dim,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seq_lens,
        seqused_cmp_kv=cmp_seq_lens,
        cmp_residual_kv=cmp_residual if compress_ratio != 1 else None,
        batch_size=num_reqs,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        cmp_topk=index_topk,
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=compress_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left if ori_mask_mode == 4 else -1,
        ori_win_right=ori_win_right if ori_mask_mode == 4 else -1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )


class A3Int8PackedCacheOps:
    """Packed INT8 CSA cache ABI selected by the A3 device adaptor."""

    uses_a5_packed_cache = False
    uses_a3_int8_packed_cache = True
    metadata_size = METADATA_SIZE
    pack_int8_kv = staticmethod(_pack_int8_kv)
    write_attention_cache = staticmethod(write_attention_cache)
    qsmla = staticmethod(qsmla)
    build_smla_metadata = staticmethod(build_smla_metadata)
