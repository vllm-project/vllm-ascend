# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Centralized DSA attention-KV execution plan (SparseFlashMla vs sharedkv)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm_ascend.ops.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata

DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2

# How compressor/SWA KV is written into the attn page cache.
KvScatterKind = Literal["nd_update_v2", "nd_update_", "kv_compress_epilog"]


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """DSA attention-KV execution plan (not indexer KV).

    Owns SparseFlashMla vs quant/sharedkv choices so callers do not scatter
    ``attn_kv_dtype == torch.bfloat16`` checks.
    """

    attn_kv_dtype: torch.dtype | None
    uses_sparse_flash_mla: bool
    layout_kv: str
    compressor_slot_mapping_format: int
    requires_block_offset_slots: bool
    pack_kv_head_dim_extra: bool
    sparse_attn_op: Callable[..., Any]
    sparse_attn_metadata_op: Callable[..., Any]
    sparse_attn_base_kwargs: dict[str, Any]
    sparse_attn_metadata_extra_kwargs: dict[str, Any]
    # A2/A3 sharedkv and SparseFlashMla need device=; A5 FP8 quant metadata does not.
    include_metadata_device: bool
    # A2/A3 sharedkv and A5 BF16 SparseFlashMla accept runtime cu_seqlens_* kwargs;
    # A5 FP8 quant sharedkv ignores them (DeviceOperator historically no-op'd).
    applies_sparse_attn_runtime_kwargs: bool
    # A2/A3: nd_update_v2; A5 BF16: inplace nd_update_; A5 FP8: kv_compress_epilog.
    kv_scatter_kind: KvScatterKind

    def metadata_kwargs(self, device) -> dict[str, Any]:
        kwargs = dict(self.sparse_attn_metadata_extra_kwargs)
        if self.include_metadata_device:
            kwargs["device"] = str(device)
        return kwargs

    def sparse_attn_kwargs(self) -> dict[str, Any]:
        """Copy of base kwargs for a single sparse-attn invocation."""
        return dict(self.sparse_attn_base_kwargs)

    def format_slot_mapping(self, slot_mapping: torch.Tensor, block_size: int) -> torch.Tensor:
        """Format flat slot ids for the plan's scatter/layout.

        Block/offset plans return ``[T, 2]``. Invalid flat slots (``slot < 0``)
        become ``[-1, -1]`` so scatter can skip them — do not use ``%`` on
        negatives (yields a fake in-range offset). Flat plans pass through.
        """
        if not self.requires_block_offset_slots:
            return slot_mapping
        valid = slot_mapping >= 0
        invalid = torch.full_like(slot_mapping, -1)
        block_idx = torch.where(valid, torch.div(slot_mapping, block_size, rounding_mode="floor"), invalid)
        offset = torch.where(valid, slot_mapping % block_size, invalid)
        return torch.stack([block_idx, offset], dim=-1).to(dtype=torch.int32)

    def kv_compress_scatter(self, cache: torch.Tensor, x: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Scatter compressor/SWA KV into the attention page cache."""
        if self.kv_scatter_kind == "nd_update_":
            import torch_npu

            if slot_mapping.dim() != 2 or slot_mapping.shape[-1] != 2:
                raise ValueError(
                    "SparseFlashMla BF16 slot_mapping must have shape "
                    f"[num_tokens, 2], got {tuple(slot_mapping.shape)}."
                )
            # Experimental: pass invalid [-1, -1] indices straight into
            # npu_scatter_nd_update_ (no remap-to-token0). Used to check whether
            # the NPU op tolerates -1 pads under ACLGraph / multistream DSA.
            # Keep fixed [T, 2] shape; do not filter/gather valid rows only.
            indices = slot_mapping.to(dtype=torch.int64).contiguous()
            updates = x.reshape((slot_mapping.shape[0],) + tuple(cache.shape[2:])).contiguous()
            torch_npu.npu_scatter_nd_update_(cache, indices, updates)
            return
        if self.kv_scatter_kind == "kv_compress_epilog":
            torch.ops._C_ascend.kv_compress_epilog(
                kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
                x=x.view(-1, x.shape[-1]),
                slot_mapping=slot_mapping,
                quant_group_size=64,
                quant_mode=2,
                round_scale_flag=True,
                layout=1,
            )
            return
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)


def uses_bf16_sparse_flash_mla(kv_cache_dtype: torch.dtype | None) -> bool:
    return kv_cache_dtype == torch.bfloat16


def add_sparse_attn_extra_kwargs(plan: DsaAttnKvPlan, extra_kwargs: dict[str, Any], **kwargs_to_add) -> None:
    """Merge SparseFlashMla / sharedkv runtime kwargs when the plan accepts them."""
    if plan.applies_sparse_attn_runtime_kwargs:
        extra_kwargs.update(kwargs_to_add)


def build_base_dsa_attn_kv_plan(kv_cache_dtype: torch.dtype | None = None) -> DsaAttnKvPlan:
    """A2/A3: sharedkv + PA_ND + block/offset slots (dtype does not switch path)."""
    return DsaAttnKvPlan(
        attn_kv_dtype=kv_cache_dtype,
        uses_sparse_flash_mla=False,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
        requires_block_offset_slots=True,
        pack_kv_head_dim_extra=False,
        sparse_attn_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={},
        sparse_attn_metadata_extra_kwargs={},
        include_metadata_device=True,
        applies_sparse_attn_runtime_kwargs=True,
        kv_scatter_kind="nd_update_v2",
    )


def build_a5_dsa_attn_kv_plan(kv_cache_dtype: torch.dtype | None = None) -> DsaAttnKvPlan:
    """A5: explicit BF16 → SparseFlashMla; otherwise KV-quant sharedkv (default)."""
    if uses_bf16_sparse_flash_mla(kv_cache_dtype):
        return DsaAttnKvPlan(
            attn_kv_dtype=kv_cache_dtype,
            uses_sparse_flash_mla=True,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            pack_kv_head_dim_extra=False,
            sparse_attn_op=sparse_flash_mla,
            sparse_attn_metadata_op=sparse_flash_mla_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_extra_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
            kv_scatter_kind="nd_update_",
        )
    return DsaAttnKvPlan(
        attn_kv_dtype=kv_cache_dtype,
        uses_sparse_flash_mla=False,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
        requires_block_offset_slots=False,
        pack_kv_head_dim_extra=True,
        sparse_attn_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        # Match main A5: quant metadata takes kv_quant_mode only (no device=).
        sparse_attn_metadata_extra_kwargs={"kv_quant_mode": 1},
        include_metadata_device=False,
        applies_sparse_attn_runtime_kwargs=False,
        kv_scatter_kind="kv_compress_epilog",
    )
