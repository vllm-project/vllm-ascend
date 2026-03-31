# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
import vllm.v1.attention.backends.gdn_attn as gdn_attn
from vllm.v1.utils import CpuGpuBuffer

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18

_IS_PATCHED = False
_ORIGINAL_BUILD = gdn_attn.GDNAttentionMetadataBuilder.build


@dataclass
class GDNChunkedPrefillMetadata:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_chunk64: CpuGpuBuffer
    chunk_offsets_chunk64: CpuGpuBuffer
    update_chunk_offsets_chunk64: CpuGpuBuffer
    final_chunk_indices_chunk64: CpuGpuBuffer
    chunk_indices_large_block: CpuGpuBuffer
    block_indices_cumsum: CpuGpuBuffer


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prepare_chunk_counts_cpu(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _fill_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    cursor = 0
    for seq_idx, num_chunks in enumerate(chunk_counts.tolist()):
        if num_chunks <= 0:
            continue
        out[cursor : cursor + num_chunks, 0].fill_(seq_idx)
        out[cursor : cursor + num_chunks, 1] = torch.arange(
            num_chunks,
            dtype=out.dtype,
        )
        cursor += num_chunks
    return cursor


def _fill_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=out[1 : chunk_counts.numel() + 1],
        )
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[: chunk_counts.numel()])
        out[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


def _get_gdn_num_heads(builder) -> int:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        return hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
    return builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)


def _allocate_chunked_prefill_slot(builder, device: torch.device):
    max_num_batched_tokens = builder.vllm_config.scheduler_config.max_num_batched_tokens
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNChunkedPrefillBufferSlot(
        chunk_indices_chunk64=CpuGpuBuffer(
            max_num_batched_tokens,
            2,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
        chunk_offsets_chunk64=CpuGpuBuffer(
            max_num_seqs + 1,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
        update_chunk_offsets_chunk64=CpuGpuBuffer(
            max_num_seqs + 1,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
        final_chunk_indices_chunk64=CpuGpuBuffer(
            max_num_seqs,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
        chunk_indices_large_block=CpuGpuBuffer(
            max_num_batched_tokens,
            2,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
        block_indices_cumsum=CpuGpuBuffer(
            max_num_batched_tokens,
            2,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
            with_numpy=False,
        ),
    )


def _ensure_chunk_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_chunk_meta_initialized", False):
        return
    builder._ascend_gdn_chunk_meta_initialized = True
    builder._ascend_gdn_chunk_meta_device = device
    builder._ascend_gdn_chunk_size = _GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    gdn_num_heads = _get_gdn_num_heads(builder)
    cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * builder._ascend_gdn_chunk_size))
    builder._ascend_gdn_cumsum_block_size = _next_power_of_2(cumsum_chunks)
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_chunked_prefill_pool = [
            _allocate_chunked_prefill_slot(builder, device),
            _allocate_chunked_prefill_slot(builder, device),
        ]


def _build_non_spec_query_start_loc_cpu(
    builder,
    attn_metadata,
    common_attn_metadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> torch.Tensor | None:
    if attn_metadata.num_prefills <= 0:
        return None

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    if (
        not getattr(builder, "use_spec_decode", False)
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0].sum().item() == 0
    ):
        return query_start_loc_cpu

    spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
    if spec_sequence_masks_cpu.sum().item() == 0:
        return query_start_loc_cpu

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
    non_spec_query_start_loc_cpu = torch.zeros(
        non_spec_query_lens_cpu.numel() + 1,
        dtype=query_start_loc_cpu.dtype,
    )
    torch.cumsum(
        non_spec_query_lens_cpu,
        dim=0,
        out=non_spec_query_start_loc_cpu[1:],
    )
    return non_spec_query_start_loc_cpu


def _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
    chunk_counts_chunk64 = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_chunk_size)
    chunk_counts_large = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_large_block_size)
    chunk_counts_cumsum = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_cumsum_block_size)
    num_seqs = chunk_counts_chunk64.numel()
    chunk_indices_chunk64 = torch.empty((int(chunk_counts_chunk64.sum().item()), 2), dtype=torch.int32)
    chunk_offsets_chunk64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
    update_chunk_offsets_chunk64 = torch.empty((num_seqs + 1,), dtype=torch.int32)
    final_chunk_indices_chunk64 = torch.empty((num_seqs,), dtype=torch.int32)
    chunk_indices_large_block = torch.empty((int(chunk_counts_large.sum().item()), 2), dtype=torch.int32)
    block_indices_cumsum = torch.empty((int(chunk_counts_cumsum.sum().item()), 2), dtype=torch.int32)

    _fill_chunk_indices_cpu(chunk_indices_chunk64, chunk_counts_chunk64)
    _fill_chunk_offsets_cpu(chunk_offsets_chunk64, chunk_counts_chunk64)
    _fill_update_chunk_offsets_cpu(update_chunk_offsets_chunk64, chunk_counts_chunk64)
    _fill_final_chunk_indices_cpu(final_chunk_indices_chunk64, chunk_counts_chunk64)
    _fill_chunk_indices_cpu(chunk_indices_large_block, chunk_counts_large)
    _fill_chunk_indices_cpu(block_indices_cumsum, chunk_counts_cumsum)

    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=chunk_indices_chunk64.to(builder._ascend_gdn_chunk_meta_device),
        chunk_offsets_chunk64=chunk_offsets_chunk64.to(builder._ascend_gdn_chunk_meta_device),
        update_chunk_offsets_chunk64=update_chunk_offsets_chunk64.to(builder._ascend_gdn_chunk_meta_device),
        final_chunk_indices_chunk64=final_chunk_indices_chunk64.to(builder._ascend_gdn_chunk_meta_device),
        chunk_indices_large_block=chunk_indices_large_block.to(builder._ascend_gdn_chunk_meta_device),
        block_indices_cumsum=block_indices_cumsum.to(builder._ascend_gdn_chunk_meta_device),
    )


def _build_non_spec_chunked_prefill_meta(builder, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
    device = builder._ascend_gdn_chunk_meta_device
    if device.type == "cpu":
        return _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu)

    builder._ascend_gdn_chunked_prefill_pool_idx = (builder._ascend_gdn_chunked_prefill_pool_idx + 1) % len(
        builder._ascend_gdn_chunked_prefill_pool
    )
    slot = builder._ascend_gdn_chunked_prefill_pool[builder._ascend_gdn_chunked_prefill_pool_idx]
    chunk_counts_chunk64 = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_chunk_size)
    chunk_counts_large = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_large_block_size)
    chunk_counts_cumsum = _prepare_chunk_counts_cpu(cu_seqlens_cpu, builder._ascend_gdn_cumsum_block_size)
    num_chunk_indices_chunk64 = _fill_chunk_indices_cpu(slot.chunk_indices_chunk64.cpu, chunk_counts_chunk64)
    num_chunk_offsets_chunk64 = _fill_chunk_offsets_cpu(slot.chunk_offsets_chunk64.cpu, chunk_counts_chunk64)
    num_update_chunk_offsets_chunk64 = _fill_update_chunk_offsets_cpu(
        slot.update_chunk_offsets_chunk64.cpu, chunk_counts_chunk64
    )
    num_final_chunk_indices_chunk64 = _fill_final_chunk_indices_cpu(
        slot.final_chunk_indices_chunk64.cpu, chunk_counts_chunk64
    )
    num_chunk_indices_large = _fill_chunk_indices_cpu(slot.chunk_indices_large_block.cpu, chunk_counts_large)
    num_block_indices_cumsum = _fill_chunk_indices_cpu(slot.block_indices_cumsum.cpu, chunk_counts_cumsum)

    chunk_indices_chunk64 = slot.chunk_indices_chunk64.copy_to_gpu(num_chunk_indices_chunk64)
    chunk_offsets_chunk64 = slot.chunk_offsets_chunk64.copy_to_gpu(num_chunk_offsets_chunk64)
    update_chunk_offsets_chunk64 = slot.update_chunk_offsets_chunk64.copy_to_gpu(num_update_chunk_offsets_chunk64)
    final_chunk_indices_chunk64 = slot.final_chunk_indices_chunk64.copy_to_gpu(num_final_chunk_indices_chunk64)
    chunk_indices_large_block = slot.chunk_indices_large_block.copy_to_gpu(num_chunk_indices_large)
    block_indices_cumsum = slot.block_indices_cumsum.copy_to_gpu(num_block_indices_cumsum)
    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=chunk_indices_chunk64,
        chunk_offsets_chunk64=chunk_offsets_chunk64,
        update_chunk_offsets_chunk64=update_chunk_offsets_chunk64,
        final_chunk_indices_chunk64=final_chunk_indices_chunk64,
        chunk_indices_large_block=chunk_indices_large_block,
        block_indices_cumsum=block_indices_cumsum,
        _buffer_slot=slot,
    )


def _patched_build(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
):
    attn_metadata = _ORIGINAL_BUILD(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        fast_build=fast_build,
    )
    attn_metadata.non_spec_chunked_prefill_meta = None
    if attn_metadata.num_prefills <= 0:
        return attn_metadata

    _ensure_chunk_meta_state(self, common_attn_metadata.query_start_loc.device)
    non_spec_query_start_loc_cpu = _build_non_spec_query_start_loc_cpu(
        self,
        attn_metadata,
        common_attn_metadata,
        num_decode_draft_tokens_cpu,
    )
    assert non_spec_query_start_loc_cpu is not None
    attn_metadata.non_spec_chunked_prefill_meta = _build_non_spec_chunked_prefill_meta(
        self, non_spec_query_start_loc_cpu
    )
    return attn_metadata


if not _IS_PATCHED:
    gdn_attn.GDNChunkedPrefillMetadata = GDNChunkedPrefillMetadata
    gdn_attn.GDNAttentionMetadataBuilder.build = _patched_build
    _IS_PATCHED = True
