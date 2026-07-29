"""DSA tensor 元数据到 Ascend 自定义算子的薄适配层。

本模块封装两组彼此独立的设备能力：LIDU -> KSC -> SFA-Offload 是 decode
resident 选择、换入和注意力计算流水线；通用 ``KvCacheFullBlockDump`` 负责按
src/dst block id 批量复制 NOPE/ROPE 满块。

这里仅做 tensor ABI 归一化、必要校验、可选 trace 和返回值封装；请求阶段、
DRAM block 预留、resident row 分配、图准入与 layer 时序均由上层负责。正常
热路径不得在本模块隐式创建跨 step 状态或执行 D2H 同步。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch_npu

from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSALightningIndexerUpdateBuffers,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_LIDU_ATTENTION_TOPK = 2048
_A5_LIDU_MAX_BUDGET = 8192
_A5_DSA_OP_LIBRARY: torch.library.Library | None = None


class DSAOffloadSelectionOutput(NamedTuple):
    """LIDU/KSC 完成后交给 SFA-Offload 的逐行 resident 视图。"""

    sparse_indices: torch.Tensor
    tail_info: torch.Tensor


class AscendDSAOpsBackend:
    """Ascend custom-op backend for DSA sparse-cache offload."""

    def __init__(self) -> None:
        # The A2/A3 extension owns these schemas. A5 keeps the same public
        # torch.ops method names and tensor ABI, but binds them to eager
        # PrivateUse1 composite implementations below.
        if self._uses_a5_composite_fallback():
            _register_a5_dsa_torch_ops()

    @staticmethod
    def _uses_a5_composite_fallback() -> bool:
        return get_ascend_device_type() == AscendDeviceType.A5

    @staticmethod
    def _squeeze_cache_head_dim(cache: torch.Tensor | None,
                                name: str) -> torch.Tensor:
        if not torch.is_tensor(cache):
            raise ValueError(f"{name} is required")
        if cache.ndim == 4 and int(cache.shape[2]) == 1:
            return cache.squeeze(2)
        if cache.ndim == 3:
            return cache
        raise ValueError(
            f"{name} must have shape [blocks, block, 1, dim] or "
            f"[blocks, block, dim], got {tuple(cache.shape)}")

    @staticmethod
    def _require_dump_tensor_device(tensor: torch.Tensor, *,
                                    name: str,
                                    device: torch.device) -> None:
        """校验独立满块复制算子的设备地址与连续布局契约。"""
        if tensor.device.type != device.type:
            raise RuntimeError(
                f"KV cache full-block dump requires {name} to be NPU "
                "addressable on the same device family as the source cache. "
                f"Got {name} on {tensor.device}, source cache on {device}.")
        tensor_index = tensor.device.index
        device_index = device.index
        if (tensor_index is not None and device_index is not None
                and int(tensor_index) != int(device_index)):
            raise RuntimeError(
                f"KV cache full-block dump requires {name} on the same NPU "
                f"as the HBM cache. Got {name} on {tensor.device}, HBM cache "
                f"on {device}.")
        if not tensor.is_contiguous():
            raise RuntimeError(
                "KV cache full-block dump uses linear block addressing and "
                f"requires contiguous {name}; got shape={tuple(tensor.shape)}, "
                f"stride={tuple(tensor.stride())}, "
                f"storage_offset={int(tensor.storage_offset())}")

    def dump_full_kv_cache_blocks(
        self,
        *,
        nopek_cache_zone: torch.Tensor,
        ropek_cache_zone: torch.Tensor,
        nopek_dram_arena: torch.Tensor,
        ropek_dram_arena: torch.Tensor,
        src_hbm_block_ids: torch.Tensor,
        dst_dram_block_ids: torch.Tensor,
    ) -> None:
        """Batch-copy this layer's newly completed MLA blocks to hot DRAM.

        This is an independent custom operator. It deliberately shares no ABI
        or state with token selection/materialization; the only ordering
        requirement is that the copy runs after this layer writes MLA cache
        and before a later forward's KSC reads the published DRAM block.

        The DSA row-id tensors are fixed-address graph buffers. Reject malformed
        inputs instead of silently materializing converted tensors in this
        layer-wise hot path. Destination block id -1 is the generic operator's
        no-op sentinel; source and destination block id 0 remain valid.
        """
        hbm_nope_cache = self._squeeze_cache_head_dim(
            nopek_cache_zone, "nopek_cache_zone")
        hbm_rope_cache = self._squeeze_cache_head_dim(
            ropek_cache_zone, "ropek_cache_zone")
        dram_nope_arena = self._squeeze_cache_head_dim(
            nopek_dram_arena, "nopek_dram_arena")
        dram_rope_arena = self._squeeze_cache_head_dim(
            ropek_dram_arena, "ropek_dram_arena")

        device = hbm_nope_cache.device
        for name, tensor in (
                ("ropek_cache_zone", hbm_rope_cache),
                ("nopek_dram_arena", dram_nope_arena),
                ("ropek_dram_arena", dram_rope_arena)):
            self._require_dump_tensor_device(
                tensor,
                name=name,
                device=device,
            )

        for name, block_ids in (
                ("src_hbm_block_ids", src_hbm_block_ids),
                ("dst_dram_block_ids", dst_dram_block_ids)):
            if not torch.is_tensor(block_ids):
                raise TypeError(f"{name} must be a tensor")
            if block_ids.device != device:
                raise RuntimeError(
                    f"{name} must stay on {device}, got {block_ids.device}")
            if block_ids.dtype != torch.int32:
                raise RuntimeError(
                    f"{name} must be int32, got {block_ids.dtype}")
            if block_ids.ndim != 1 or not block_ids.is_contiguous():
                raise RuntimeError(
                    f"{name} must be a contiguous 1-D graph buffer, got "
                    f"shape={tuple(block_ids.shape)}, "
                    f"contiguous={block_ids.is_contiguous()}")
        if int(src_hbm_block_ids.numel()) != int(
                dst_dram_block_ids.numel()):
            raise RuntimeError(
                "KV cache full-block dump source/destination row counts differ: "
                f"src={int(src_hbm_block_ids.numel())}, "
                f"dst={int(dst_dram_block_ids.numel())}")
        if int(src_hbm_block_ids.numel()) == 0:
            return

        if self._uses_a5_composite_fallback():
            valid = (
                (src_hbm_block_ids >= 0)
                & (dst_dram_block_ids >= 0)
            )
            src = src_hbm_block_ids[valid].long()
            dst = dst_dram_block_ids[valid].long()
            dram_nope_arena.index_copy_(
                0,
                dst,
                hbm_nope_cache.index_select(0, src),
            )
            dram_rope_arena.index_copy_(
                0,
                dst,
                hbm_rope_cache.index_select(0, src),
            )
        else:
            torch.ops._C_ascend.kv_cache_full_block_dump(
                hbm_nope_cache,
                hbm_rope_cache,
                dram_nope_arena,
                dram_rope_arena,
                src_hbm_block_ids,
                dst_dram_block_ids,
            )

    @staticmethod
    def _a5_native_indexer_topk(
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(query.shape[0])
        actual_seq_lengths_query = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int32,
            device=query.device,
        )
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=_A5_LIDU_MAX_BUDGET,
            sparse_mode=3,
        )
        return topk_indices

    @staticmethod
    def _clear_lidu_outputs(
        outputs: DSALightningIndexerUpdateBuffers,
    ) -> None:
        outputs.topk_index.fill_(-1)
        outputs.topk_slots.fill_(-1)
        outputs.miss_count.zero_()
        outputs.tail_info.fill_(-1)
        outputs.tail_info[:, 1].zero_()

    def _lightning_indexer_decode_update_a5(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        req_pool_entries: torch.Tensor,
        cache_slots: torch.Tensor,
        row_modes: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        outputs: DSALightningIndexerUpdateBuffers,
    ) -> None:
        """A5 functional implementation of the unchanged LIDU tensor ABI.

        A5's native Lightning Indexer supports top-k up to 8192, while the
        fused A2/A3 update kernel is arch22/arch32-only. This eager fallback
        keeps all request state and output tensors on device and performs the
        resident-map update with tensor operations.
        """
        self._clear_lidu_outputs(outputs)
        block_size = int(key.shape[1])
        tail_counts = (
            (actual_seq_lengths_key - 1).remainder(block_size) + 1
        )
        sparse_rows = row_modes == 2
        candidate_lengths = torch.where(
            sparse_rows,
            actual_seq_lengths_key - tail_counts,
            actual_seq_lengths_key,
        )
        selected = self._a5_native_indexer_topk(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_key=candidate_lengths,
            block_table=block_table,
        )
        if selected.ndim == 2:
            selected = selected.unsqueeze(1)
        selected = selected.to(torch.int32)

        output_indices = outputs.topk_index
        output_slots = outputs.topk_slots
        resident_slot_ids = torch.arange(
            _A5_LIDU_MAX_BUDGET,
            dtype=torch.int32,
            device=query.device,
        )
        for batch_index in range(int(query.shape[0])):
            row_mode = row_modes[batch_index]
            is_dense = row_mode == 1
            is_sparse = row_mode == 2
            pool_index = req_pool_entries[batch_index].long()
            resident_row = cache_slots[pool_index]
            metadata = resident_row[-1]
            is_first_fill = is_sparse & (metadata < 0)

            dense_indices = selected[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ]
            dense_valid = dense_indices >= 0
            output_indices[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ].copy_(torch.where(
                is_dense & dense_valid,
                dense_indices,
                torch.full_like(dense_indices, -1),
            ))
            output_slots[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ].copy_(torch.where(
                is_dense & dense_valid,
                dense_indices,
                torch.full_like(dense_indices, -1),
            ))

            first_fill_indices = selected[
                batch_index,
                0,
                :_A5_LIDU_MAX_BUDGET,
            ]
            target_budget = -metadata
            first_fill_valid = (
                (resident_slot_ids < target_budget)
                & (first_fill_indices >= 0)
                & is_first_fill
            )
            first_fill_out_indices = torch.where(
                first_fill_valid,
                first_fill_indices,
                torch.full_like(first_fill_indices, -1),
            )
            first_fill_out_slots = torch.where(
                first_fill_valid,
                resident_slot_ids,
                torch.full_like(resident_slot_ids, -1),
            )

            steady_indices = selected[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ]
            steady_safe_indices = steady_indices.clamp(
                min=0,
                max=int(resident_row.shape[0]) - 2,
            ).long()
            current_slots = resident_row.index_select(
                0,
                steady_safe_indices,
            )
            steady_valid = steady_indices >= 0
            miss_mask = (
                steady_valid
                & (current_slots < 0)
                & is_sparse
                & ~is_first_fill
            )
            order = torch.argsort(
                miss_mask.to(torch.int32),
                descending=True,
            )
            ordered_indices = steady_indices.index_select(0, order)
            ordered_current_slots = current_slots.index_select(0, order)
            ordered_misses = miss_mask.index_select(0, order)

            slot_ref_counts = torch.zeros(
                _A5_LIDU_MAX_BUDGET,
                dtype=torch.int32,
                device=query.device,
            )
            slot_ref_counts.scatter_add_(
                0,
                current_slots.clamp(
                    min=0,
                    max=_A5_LIDU_MAX_BUDGET - 1,
                ).long(),
                (current_slots >= 0).to(torch.int32),
            )
            budget = metadata.clamp(
                min=0,
                max=_A5_LIDU_MAX_BUDGET,
            )
            available_mask = (
                (resident_slot_ids < budget)
                & (slot_ref_counts == 0)
            )
            available_key = torch.where(
                available_mask,
                resident_slot_ids,
                resident_slot_ids + _A5_LIDU_MAX_BUDGET,
            )
            available_slots = torch.argsort(
                available_key,
            )[:_LIDU_ATTENTION_TOPK].to(torch.int32)
            steady_slots = torch.where(
                ordered_misses,
                available_slots,
                ordered_current_slots,
            )

            steady_active = is_sparse & ~is_first_fill
            steady_output_indices = torch.where(
                steady_active,
                ordered_indices,
                torch.full_like(ordered_indices, -1),
            )
            steady_output_slots = torch.where(
                steady_active,
                steady_slots,
                torch.full_like(steady_slots, -1),
            )
            output_indices[
                batch_index,
                0,
                :_A5_LIDU_MAX_BUDGET,
            ].copy_(torch.where(
                is_first_fill,
                first_fill_out_indices,
                output_indices[
                    batch_index,
                    0,
                    :_A5_LIDU_MAX_BUDGET,
                ],
            ))
            output_slots[
                batch_index,
                0,
                :_A5_LIDU_MAX_BUDGET,
            ].copy_(torch.where(
                is_first_fill,
                first_fill_out_slots,
                output_slots[
                    batch_index,
                    0,
                    :_A5_LIDU_MAX_BUDGET,
                ],
            ))
            output_indices[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ].copy_(torch.where(
                steady_active,
                steady_output_indices,
                output_indices[
                    batch_index,
                    0,
                    :_LIDU_ATTENTION_TOPK,
                ],
            ))
            output_slots[
                batch_index,
                0,
                :_LIDU_ATTENTION_TOPK,
            ].copy_(torch.where(
                steady_active,
                steady_output_slots,
                output_slots[
                    batch_index,
                    0,
                    :_LIDU_ATTENTION_TOPK,
                ],
            ))

            first_fill_tokens = first_fill_indices[
                first_fill_valid
            ].long()
            first_fill_slots = resident_slot_ids[
                first_fill_valid
            ]
            resident_row.index_copy_(
                0,
                first_fill_tokens,
                first_fill_slots,
            )
            resident_row[-1].copy_(torch.where(
                is_first_fill,
                target_budget,
                resident_row[-1],
            ))

            miss_slots = available_slots[ordered_misses]
            token_positions = torch.arange(
                int(resident_row.shape[0]) - 1,
                dtype=torch.int32,
                device=query.device,
            )
            resident_values = resident_row[:-1]
            mapped = (
                (resident_values >= 0)
                & (resident_values < _A5_LIDU_MAX_BUDGET)
            )
            slot_to_token = torch.full(
                (_A5_LIDU_MAX_BUDGET,),
                -1,
                dtype=torch.int32,
                device=query.device,
            )
            slot_to_token[
                resident_values[mapped].long()
            ] = token_positions[mapped]
            evicted_tokens = slot_to_token[
                miss_slots.long()
            ].long()
            valid_evictions = evicted_tokens >= 0
            # An unused resident slot has no old token (slot_to_token=-1).
            # Indexing resident_row[-1] would corrupt its budget metadata.
            # Route those no-op writes to the metadata cell while preserving
            # its current value; valid evictions still clear the old token.
            metadata_index = int(resident_row.shape[0]) - 1
            safe_evicted_tokens = torch.where(
                valid_evictions,
                evicted_tokens,
                torch.full_like(evicted_tokens, metadata_index),
            )
            old_evicted_values = resident_row.index_select(
                0,
                safe_evicted_tokens,
            )
            resident_row.index_copy_(
                0,
                safe_evicted_tokens,
                torch.where(
                    valid_evictions,
                    torch.full_like(old_evicted_values, -1),
                    old_evicted_values,
                ),
            )
            resident_row[
                ordered_indices[ordered_misses].long()
            ] = miss_slots

            miss_count = ordered_misses.sum(dtype=torch.int32)
            outputs.miss_count[batch_index].copy_(torch.where(
                is_first_fill,
                target_budget.to(torch.int32),
                torch.where(
                    steady_active,
                    miss_count,
                    torch.zeros_like(miss_count),
                ),
            ))
            outputs.tail_info[batch_index, 0].copy_(torch.where(
                is_sparse,
                resident_row[-1],
                torch.full_like(resident_row[-1], -1),
            ))
            outputs.tail_info[batch_index, 1].copy_(torch.where(
                is_sparse,
                tail_counts[batch_index],
                torch.zeros_like(tail_counts[batch_index]),
            ))

    def lightning_indexer_decode_update(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        req_pool_entries: torch.Tensor,
        cache_slots: torch.Tensor,
        row_modes: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        outputs: DSALightningIndexerUpdateBuffers,
    ) -> None:
        """Run LIDU with caller-owned graph-stable state and outputs."""
        if self._uses_a5_composite_fallback():
            torch.ops._C_ascend.npu_lightning_indexer_decode_update_out(
                query,
                key,
                weights,
                req_pool_entries,
                cache_slots,
                row_modes,
                actual_seq_lengths_key,
                block_table,
                outputs.topk_index,
                outputs.topk_slots,
                outputs.miss_count,
                outputs.tail_info,
            )
        else:
            torch.ops._C_ascend.npu_lightning_indexer_decode_update_out(
                query,
                key,
                weights,
                req_pool_entries,
                cache_slots,
                row_modes,
                actual_seq_lengths_key,
                block_table,
                outputs.topk_index,
                outputs.topk_slots,
                outputs.miss_count,
                outputs.tail_info,
            )

    def kvcache_scatter_copy(
        self,
        *,
        nopek_cache_zone: torch.Tensor,
        ropek_cache_zone: torch.Tensor,
        nopek_dram_arena: torch.Tensor,
        ropek_dram_arena: torch.Tensor,
        hbm_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        src_token_ids: torch.Tensor,
        dst_slots: torch.Tensor,
        copy_counts: torch.Tensor,
    ) -> None:
        """Materialize only LIDU's miss prefix into resident HBM slots."""
        hbm_rope = self._squeeze_cache_head_dim(
            ropek_cache_zone,
            "ropek_cache_zone",
        )
        hbm_nope = self._squeeze_cache_head_dim(
            nopek_cache_zone,
            "nopek_cache_zone",
        )
        dram_rope = self._squeeze_cache_head_dim(
            ropek_dram_arena,
            "ropek_dram_arena",
        )
        dram_nope = self._squeeze_cache_head_dim(
            nopek_dram_arena,
            "nopek_dram_arena",
        )
        if self._uses_a5_composite_fallback():
            torch.ops._C_ascend.npu_kvcache_scatter_copy(
                hbm_rope,
                hbm_nope,
                dram_rope,
                dram_nope,
                hbm_block_table,
                dram_block_table,
                src_token_ids,
                dst_slots,
                copy_counts,
            )
        else:
            torch.ops._C_ascend.npu_kvcache_scatter_copy(
                hbm_rope,
                hbm_nope,
                dram_rope,
                dram_nope,
                hbm_block_table,
                dram_block_table,
                src_token_ids,
                dst_slots,
                copy_counts,
            )

    @staticmethod
    def _kvcache_scatter_copy_a5(
        hbm_rope: torch.Tensor,
        hbm_nope: torch.Tensor,
        dram_rope: torch.Tensor,
        dram_nope: torch.Tensor,
        hbm_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        src_token_ids: torch.Tensor,
        dst_slots: torch.Tensor,
        copy_counts: torch.Tensor,
    ) -> None:
        block_size = int(hbm_nope.shape[1])
        source = src_token_ids.squeeze(1)
        destination = dst_slots.squeeze(1)
        offsets = torch.arange(
            int(source.shape[1]),
            dtype=torch.int32,
            device=source.device,
        )
        active = offsets.unsqueeze(0) < copy_counts.unsqueeze(1)
        batch_ids = torch.arange(
            int(source.shape[0]),
            dtype=torch.long,
            device=source.device,
        ).unsqueeze(1).expand_as(source)[active]
        source = source[active].long()
        destination = destination[active].long()
        source_blocks = dram_block_table[
            batch_ids,
            source // block_size,
        ].long()
        destination_blocks = hbm_block_table[
            batch_ids,
            destination // block_size,
        ].long()
        source_flat = (
            source_blocks * block_size
            + source.remainder(block_size)
        )
        destination_flat = (
            destination_blocks * block_size
            + destination.remainder(block_size)
        )
        hbm_nope.view(-1, hbm_nope.shape[-1]).index_copy_(
            0,
            destination_flat,
            dram_nope.view(
                -1,
                dram_nope.shape[-1],
            ).index_select(0, source_flat),
        )
        hbm_rope.view(-1, hbm_rope.shape[-1]).index_copy_(
            0,
            destination_flat,
            dram_rope.view(
                -1,
                dram_rope.shape[-1],
            ).index_select(0, source_flat),
        )

    @staticmethod
    def sparse_flash_attention_for_offload(
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        sparse_indices: torch.Tensor,
        tail_info: torch.Tensor,
        scale_value: float,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> torch.Tensor:
        """Run resident top-2048 attention and append the dense tail."""
        if (
            get_ascend_device_type() == AscendDeviceType.A5
        ):
            resident_key = AscendDSAOpsBackend._squeeze_cache_head_dim(
                key,
                "key",
            )
            resident_rope = (
                AscendDSAOpsBackend._squeeze_cache_head_dim(
                    key_rope,
                    "key_rope",
                )
            )
            block_size = int(resident_key.shape[1])
            topk = sparse_indices[
                :,
                0,
                :_LIDU_ATTENTION_TOPK,
            ]
            tail_offsets = torch.arange(
                block_size,
                dtype=torch.int32,
                device=topk.device,
            )
            tail = (
                tail_info[:, 0].unsqueeze(1)
                + tail_offsets.unsqueeze(0)
            )
            tail_valid = (
                tail_offsets.unsqueeze(0)
                < tail_info[:, 1].unsqueeze(1)
            )
            tail = torch.where(
                tail_valid,
                tail,
                torch.full_like(tail, -1),
            )
            selected = torch.cat((topk, tail), dim=-1)
            valid = selected >= 0
            safe_selected = selected.clamp(min=0).long()
            logical_blocks = safe_selected // block_size
            physical_blocks = block_table.gather(
                1,
                logical_blocks,
            ).long()
            flat_slots = (
                physical_blocks * block_size
                + safe_selected.remainder(block_size)
            )
            gathered_key = resident_key.view(
                -1,
                resident_key.shape[-1],
            ).index_select(0, flat_slots.reshape(-1)).view(
                selected.shape[0],
                selected.shape[1],
                resident_key.shape[-1],
            )
            gathered_rope = resident_rope.view(
                -1,
                resident_rope.shape[-1],
            ).index_select(0, flat_slots.reshape(-1)).view(
                selected.shape[0],
                selected.shape[1],
                resident_rope.shape[-1],
            )
            scores = (
                torch.einsum(
                    "bhd,bsd->bhs",
                    query,
                    gathered_key,
                )
                + torch.einsum(
                    "bhd,bsd->bhs",
                    query_rope,
                    gathered_rope,
                )
            ).float()
            scores.mul_(float(scale_value))
            scores.masked_fill_(
                ~valid.unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
            probabilities = torch.softmax(
                scores,
                dim=-1,
            ).to(gathered_key.dtype)
            return torch.einsum(
                "bhs,bsd->bhd",
                probabilities,
                gathered_key,
            ).to(query.dtype)
        return torch.ops._C_ascend.npu_sparse_flash_attention_for_offload(
            query,
            key,
            key,
            sparse_indices,
            tail_info,
            float(scale_value),
            1,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            query_rope,
            key_rope,
            "TND",
            "PA_BSND",
            3,
        )


def _a5_lightning_indexer_decode_update_out(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    topk_index_out: torch.Tensor,
    topk_slots_out: torch.Tensor,
    miss_count_out: torch.Tensor,
    tail_info_out: torch.Tensor,
) -> None:
    outputs = DSALightningIndexerUpdateBuffers(
        topk_index=topk_index_out,
        topk_slots=topk_slots_out,
        miss_count=miss_count_out,
        tail_info=tail_info_out,
    )
    # Call the composite implementation directly; dispatching the public
    # method here would recurse through this PrivateUse1 registration.
    backend = object.__new__(AscendDSAOpsBackend)
    backend._lightning_indexer_decode_update_a5(
        query=query,
        key=key,
        weights=weights,
        req_pool_entries=req_pool_entries,
        cache_slots=cache_slots,
        row_modes=row_modes,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        outputs=outputs,
    )


def _a5_kvcache_scatter_copy(
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    dram_k_rope: torch.Tensor,
    dram_kv_cache: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    src_token_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    AscendDSAOpsBackend._kvcache_scatter_copy_a5(
        hbm_k_rope,
        hbm_kv_cache,
        dram_k_rope,
        dram_kv_cache,
        hbm_block_table,
        dram_block_table,
        src_token_ids,
        dst_slots,
        copy_counts,
    )


def _register_a5_dsa_torch_ops() -> None:
    """Bind A5 composites under the source branch's unchanged torch ABI."""
    global _A5_DSA_OP_LIBRARY
    if _A5_DSA_OP_LIBRARY is not None:
        return

    lib = torch.library.Library("_C_ascend", "FRAGMENT")
    lib.define(
        "npu_lightning_indexer_decode_update_out("
        "Tensor query, Tensor key, Tensor weights, Tensor req_pool_entries, "
        "Tensor(a!) cache_slots, Tensor row_modes, "
        "Tensor actual_seq_lengths_key, Tensor block_table, "
        "Tensor(b!) topk_index_out, Tensor(c!) topk_slots_out, "
        "Tensor(d!) miss_count_out, Tensor(e!) tail_info_out) -> ()"
    )
    lib.impl(
        "npu_lightning_indexer_decode_update_out",
        _a5_lightning_indexer_decode_update_out,
        "PrivateUse1",
    )
    lib.define(
        "npu_kvcache_scatter_copy(Tensor(a!) hbm_k_rope, "
        "Tensor(b!) hbm_kv_cache, Tensor dram_k_rope, "
        "Tensor dram_kv_cache, Tensor hbm_block_table, "
        "Tensor dram_block_table, Tensor src_token_ids, "
        "Tensor dst_slots, Tensor copy_counts) -> ()"
    )
    lib.impl(
        "npu_kvcache_scatter_copy",
        _a5_kvcache_scatter_copy,
        "PrivateUse1",
    )
    _A5_DSA_OP_LIBRARY = lib
