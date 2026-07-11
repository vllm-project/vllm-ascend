#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

from dataclasses import dataclass

import numpy as np
import torch


def get_cumsum_and_arange(
    num_scheduled_tokens: np.ndarray,
    arange_np: np.ndarray,
    cumsum_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative token counts and flattened per-request arange."""
    cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
    total_num_tokens = cu_num_tokens[-1]
    cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
    arange = arange_np[:total_num_tokens] - cumsums_offsets
    return cu_num_tokens, arange


@dataclass
class PCPCommonLayout:
    num_padded_scheduled_tokens: np.ndarray
    num_pcp_pads: np.ndarray
    pcp_unpad_mask: np.ndarray
    pcp_padded_tokens_length: int
    pcp_tokens: np.ndarray
    positions: np.ndarray
    padded_pos_start_loc: np.ndarray
    all_positions: np.ndarray
    restore_idx: np.ndarray
    pcp_chunk_sizes: np.ndarray
    pcp_chunk_arange: np.ndarray
    pcp_head_chunk_mask: np.ndarray
    num_decode_tokens: int
    pcp_world_size: int

    def get_rank_positions(self, positions_start_loc: int | np.ndarray, rank: int) -> np.ndarray:
        positions = np.zeros(len(self.pcp_head_chunk_mask), dtype=np.int32)
        head_start_loc = positions_start_loc + rank * self.pcp_chunk_sizes
        tail_start_loc = positions_start_loc + (2 * self.pcp_world_size - rank - 1) * self.pcp_chunk_sizes
        positions[self.pcp_head_chunk_mask] = self.pcp_chunk_arange + np.repeat(head_start_loc, self.pcp_chunk_sizes)
        positions[~self.pcp_head_chunk_mask] = (
            self.pcp_chunk_arange[self.num_decode_tokens :]
            + np.repeat(tail_start_loc, self.pcp_chunk_sizes)[self.num_decode_tokens :]
        )
        return positions


@dataclass
class PCPHybridLayout:
    num_padded_scheduled_tokens: np.ndarray
    positions_linear: np.ndarray
    pcp_enter_fa_restore_idx: torch.Tensor
    pcp_fa_padding_restore_idx: torch.Tensor | None
    pcp_exit_fa_scatter_idx: torch.Tensor | None
    pcp_fa_query_idx: torch.Tensor | None
    pcp_padded_tokens_fla: int
    total_pcp_padding_tokens_fla: int
    max_num_tokens_across_pcp: int
    pcp_tokens_padded: np.ndarray
    num_scheduled_tokens_padded: np.ndarray
    total_num_scheduled_tokens: int


def build_fa_padding_restore_idx(
    pcp_unpad_mask: np.ndarray,
    decode_offset: int,
    actual_qkv_len: int,
) -> np.ndarray | None:
    target_len = pcp_unpad_mask.shape[0]
    if actual_qkv_len > target_len:
        raise ValueError(f"actual_qkv_len ({actual_qkv_len}) must not exceed FA padded length ({target_len}).")
    if actual_qkv_len == target_len:
        return None
    if decode_offset > target_len or actual_qkv_len < decode_offset:
        raise ValueError(
            f"Invalid PCP restore layout: decode_offset={decode_offset}, "
            f"actual_qkv_len={actual_qkv_len}, target_len={target_len}."
        )

    restore_idx = np.empty(target_len, dtype=np.int32)
    restore_idx[:decode_offset] = np.arange(decode_offset, dtype=np.int32)

    prefill_unpad_mask = pcp_unpad_mask[decode_offset:]
    prefill_real_tokens = int(prefill_unpad_mask.sum())
    expected_actual_qkv_len = decode_offset + prefill_real_tokens
    if expected_actual_qkv_len != actual_qkv_len:
        raise ValueError(f"PCP unpad mask expects {expected_actual_qkv_len} QKV rows, but got {actual_qkv_len}.")

    prefill_restore_idx = restore_idx[decode_offset:]
    prefill_restore_idx.fill(actual_qkv_len)
    prefill_restore_idx[prefill_unpad_mask] = np.arange(
        decode_offset,
        actual_qkv_len,
        dtype=np.int32,
    )
    return restore_idx


def build_common_pcp_layout(
    num_scheduled_tokens: np.ndarray,
    arange_np: np.ndarray,
    *,
    pcp_world_size: int,
    pcp_world_rank: int,
    num_decode_reqs: int,
    num_decode_tokens: int,
) -> PCPCommonLayout:
    """Build common DualChunkSwap PCP token layout for one PCP rank."""
    num_padded_scheduled_tokens = np.ceil(num_scheduled_tokens / (2 * pcp_world_size)).astype(np.int32) * (
        2 * pcp_world_size
    )
    num_padded_scheduled_tokens[:num_decode_reqs] = num_scheduled_tokens[:num_decode_reqs] * pcp_world_size
    num_pcp_pads = num_padded_scheduled_tokens - num_scheduled_tokens

    cu_padded_tokens, pcp_padded_arange = get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)
    pcp_padded_tokens_length = pcp_padded_arange.shape[0]
    pcp_unpad_mask = pcp_padded_arange < np.repeat(num_scheduled_tokens, num_padded_scheduled_tokens)
    unpad_mask_decode = pcp_unpad_mask[: num_decode_tokens * pcp_world_size]
    unpad_mask_decode = unpad_mask_decode.reshape([-1, pcp_world_size])
    unpad_mask_decode[:, 0] = True
    unpad_mask_decode[:, 1:] = False

    pcp_tokens = num_padded_scheduled_tokens // pcp_world_size
    pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
    pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

    _, pcp_arange = get_cumsum_and_arange(pcp_tokens, arange_np)
    _, pcp_chunk_arange = get_cumsum_and_arange(pcp_chunk_sizes, arange_np)
    pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

    layout = PCPCommonLayout(
        num_padded_scheduled_tokens=num_padded_scheduled_tokens,
        num_pcp_pads=num_pcp_pads,
        pcp_unpad_mask=pcp_unpad_mask,
        pcp_padded_tokens_length=pcp_padded_tokens_length,
        pcp_tokens=pcp_tokens,
        positions=np.empty(0, dtype=np.int32),
        padded_pos_start_loc=np.empty(0, dtype=np.int32),
        all_positions=np.empty(0, dtype=np.int32),
        restore_idx=np.empty(0, dtype=np.int64),
        pcp_chunk_sizes=pcp_chunk_sizes,
        pcp_chunk_arange=pcp_chunk_arange,
        pcp_head_chunk_mask=pcp_head_chunk_mask,
        num_decode_tokens=num_decode_tokens,
        pcp_world_size=pcp_world_size,
    )

    positions = layout.get_rank_positions(0, pcp_world_rank)
    padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
    padded_pos_start_loc[0] = 0
    if num_decode_reqs > 0:
        positions[:num_decode_tokens] = get_cumsum_and_arange(num_scheduled_tokens[:num_decode_reqs], arange_np)[1]

    all_positions = np.concatenate(
        [layout.get_rank_positions(padded_pos_start_loc, rank_i) for rank_i in range(pcp_world_size)]
    )
    layout.positions = positions
    layout.padded_pos_start_loc = padded_pos_start_loc
    layout.all_positions = all_positions
    layout.restore_idx = all_positions.argsort()
    return layout


def build_hybrid_fa_layout(
    num_scheduled_tokens: np.ndarray,
    arange_np: np.ndarray,
    common_layout: PCPCommonLayout,
    *,
    pcp_world_size: int,
    pcp_world_rank: int,
    num_reqs: int,
    num_decode_reqs: int,
    num_decode_tokens: int,
    has_speculative_config: bool,
) -> PCPHybridLayout:
    """Build hybrid linear-attention/full-attention PCP reorder metadata."""
    num_padded_scheduled_tokens = common_layout.num_padded_scheduled_tokens.copy()
    pcp_padded_tokens_fla = 0
    total_pcp_padding_tokens_fla = 0
    max_scheduled_prefill_tokens = 0
    prefill_tokens_allranks = None
    prefill_scheduled_tokens_linear = None

    if num_decode_reqs > 0:
        num_padded_scheduled_tokens[:num_decode_reqs] = num_padded_scheduled_tokens[:num_decode_reqs] // pcp_world_size

    if num_reqs - num_decode_reqs > 0:
        prefill_tokens_tensor = torch.Tensor(num_scheduled_tokens[num_decode_reqs:])
        prefill_tokens_allranks = get_cp_local_seq_lens(prefill_tokens_tensor, pcp_world_size, 1, 1).long().numpy()
        prefill_scheduled_tokens_linear = prefill_tokens_allranks[:, pcp_world_rank, 0]
        num_padded_scheduled_tokens[num_decode_reqs:] = prefill_scheduled_tokens_linear

        prefill_tokens_start_loc = np.zeros((num_reqs - num_decode_reqs, pcp_world_size + 1), dtype=np.int64)
        prefill_tokens_start_loc[:, 1:] = np.cumsum(prefill_tokens_allranks[..., 0], axis=-1)
        prefill_tokens_cu_ranks = prefill_tokens_start_loc[:, pcp_world_rank]
        _, positions_linear = get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)
        positions_linear[num_decode_tokens:] = positions_linear[num_decode_tokens:] + np.repeat(
            prefill_tokens_cu_ranks, prefill_scheduled_tokens_linear
        )

        max_scheduled_prefill_tokens = prefill_tokens_allranks[:, 0, 0].sum()
        num_prefill_tokens = num_scheduled_tokens[num_decode_reqs:].sum()
        total_pcp_padding_tokens_fla = max_scheduled_prefill_tokens * pcp_world_size - num_prefill_tokens
        pcp_padded_tokens_fla += max_scheduled_prefill_tokens - prefill_scheduled_tokens_linear.sum()
    else:
        _, positions_linear = get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)

    max_scheduled_tokens = max_scheduled_prefill_tokens + num_decode_tokens
    enter_fa_prefill_restore_idx = None
    if num_reqs - num_decode_reqs > 0:
        assert prefill_tokens_allranks is not None
        prefill_tokens_allranks = prefill_tokens_allranks[..., 0]
        _, prefill_arange_allranks = get_cumsum_and_arange(prefill_tokens_allranks.flatten(), arange_np)
        _, prefill_rank_offset = get_cumsum_and_arange(
            np.ones(num_reqs - num_decode_reqs, dtype=np.int64) * pcp_world_size, arange_np
        )
        prefill_all_offset = (
            np.repeat(prefill_rank_offset * max_scheduled_tokens, prefill_tokens_allranks.flatten()) + num_decode_tokens
        )

        prefill_local_start_local = np.zeros_like(prefill_tokens_allranks)
        prefill_local_start_local[1:, :] = np.cumsum(prefill_tokens_allranks, axis=0)[:-1, :]
        prefill_local_offset = np.repeat(prefill_local_start_local.flatten(), prefill_tokens_allranks.flatten())
        prefill_all_offset = np.add(prefill_all_offset, prefill_local_offset)
        enter_fa_prefill_restore_idx = np.add(prefill_all_offset, prefill_arange_allranks)

    enter_fa_decode_restore_idx = None
    if num_decode_reqs > 0:
        if has_speculative_config:
            decode_reqs_offset = np.tile(np.arange(num_decode_tokens, dtype=np.int64), pcp_world_size)
            decode_ranks_offset = np.repeat(np.arange(pcp_world_size, dtype=np.int64), num_decode_tokens)
            decode_ranks_offset = decode_ranks_offset * max_scheduled_tokens
        else:
            num_decode_pcp_size = np.ones(num_decode_reqs, dtype=np.int64) * pcp_world_size
            decode_reqs_offset = np.repeat(np.arange(num_decode_reqs, dtype=np.int64), num_decode_pcp_size)
            decode_ranks_offset = get_cumsum_and_arange(num_decode_pcp_size, arange_np)[1] * max_scheduled_tokens
        enter_fa_decode_restore_idx = np.add(decode_reqs_offset, decode_ranks_offset)

    if enter_fa_decode_restore_idx is not None and enter_fa_prefill_restore_idx is not None:
        pcp_enter_fa_restore_idx = torch.from_numpy(
            np.concatenate([enter_fa_decode_restore_idx, enter_fa_prefill_restore_idx])
        )
    elif enter_fa_decode_restore_idx is not None:
        pcp_enter_fa_restore_idx = torch.from_numpy(enter_fa_decode_restore_idx)
    elif enter_fa_prefill_restore_idx is not None:
        pcp_enter_fa_restore_idx = torch.from_numpy(enter_fa_prefill_restore_idx)
    else:
        pcp_enter_fa_restore_idx = torch.empty(0, dtype=torch.int64)

    padding_restore_idx = build_fa_padding_restore_idx(
        common_layout.pcp_unpad_mask[: common_layout.pcp_padded_tokens_length],
        num_decode_tokens * pcp_world_size,
        pcp_enter_fa_restore_idx.shape[0],
    )
    pcp_fa_padding_restore_idx = torch.from_numpy(padding_restore_idx) if padding_restore_idx is not None else None

    pcp_exit_fa_scatter_idx = None
    pcp_fa_query_idx = None
    if num_reqs > num_decode_reqs:
        assert prefill_scheduled_tokens_linear is not None
        all_positions_prefill = [
            common_layout.get_rank_positions(common_layout.padded_pos_start_loc, rank_i)[num_decode_tokens:]
            - num_decode_tokens * pcp_world_size
            for rank_i in range(pcp_world_size)
        ]
        all_positions_prefill_tensor = torch.from_numpy(np.concatenate(all_positions_prefill))
        all_exit_fa_restore_idx = all_positions_prefill_tensor.float().argsort()
        unpad_mask_prefill = common_layout.pcp_unpad_mask[: common_layout.pcp_padded_tokens_length][
            num_decode_tokens * pcp_world_size :
        ]
        ori_tokens_start_loc = np.roll(np.cumsum(num_scheduled_tokens[num_decode_reqs:]), 1)
        ori_tokens_start_loc[0] = 0
        exit_fa_scatter_indices = positions_linear[num_decode_tokens:] + np.repeat(
            ori_tokens_start_loc, prefill_scheduled_tokens_linear
        )

        pcp_exit_fa_scatter_idx = torch.index_select(
            all_exit_fa_restore_idx[unpad_mask_prefill],
            0,
            torch.from_numpy(exit_fa_scatter_indices),
        )
        pcp_fa_query_idx = torch.from_numpy(all_positions_prefill[pcp_world_rank])

    pcp_tokens_padded = common_layout.pcp_tokens[:num_reqs]
    return PCPHybridLayout(
        num_padded_scheduled_tokens=num_padded_scheduled_tokens,
        positions_linear=positions_linear,
        pcp_enter_fa_restore_idx=pcp_enter_fa_restore_idx,
        pcp_fa_padding_restore_idx=pcp_fa_padding_restore_idx,
        pcp_exit_fa_scatter_idx=pcp_exit_fa_scatter_idx,
        pcp_fa_query_idx=pcp_fa_query_idx,
        pcp_padded_tokens_fla=int(pcp_padded_tokens_fla),
        total_pcp_padding_tokens_fla=int(total_pcp_padding_tokens_fla),
        max_num_tokens_across_pcp=int(max_scheduled_tokens),
        pcp_tokens_padded=pcp_tokens_padded,
        num_scheduled_tokens_padded=np.array(pcp_tokens_padded, dtype=np.int32),
        total_num_scheduled_tokens=int(num_padded_scheduled_tokens[:num_reqs].sum()),
    )


def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    pcp_world_size: int = 1,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Return per-request KV lengths for every PCP/DCP rank.

    The output shape is ``[num_requests, pcp_world_size, dcp_world_size]``.
    """
    num_requests = seq_lens.size(0)
    total_world_size = pcp_world_size * dcp_world_size
    seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
    rank_offsets = (
        torch.arange(total_world_size, dtype=seq_lens.dtype, device=seq_lens.device)
        .unsqueeze(0)
        .repeat(num_requests, 1)
    )
    base = seq_lens_tiled // cp_kv_cache_interleave_size // total_world_size * cp_kv_cache_interleave_size
    remainder = seq_lens_tiled - base * total_world_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    return (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
