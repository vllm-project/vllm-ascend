#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/worker.py
#

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class PCPManager:
    """
    Manager for Prefill Context Parallelism (PCP) metadata and buffers.

    This manager encapsulates all PCP-related buffers and logic so that the
    ModelRunner can access them via `self.pcp_manager`.
    """

    num_reqs: int = 0
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        dcp_world_size: int,
        dcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        vllm_config: VllmConfig,
        use_async_scheduling: bool,
        pin_memory: bool = False,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_world_rank = pcp_rank
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_rank
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)
        self.vllm_config = vllm_config
        self.max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_async_scheduling = use_async_scheduling
        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_padded_slot_mapping = torch.full(
            (max_buffer_num_tokens,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        self.pcp_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.total_num_sampled_tokens_pcp = 0
        self.num_pcp_pads_cpu_tensor = torch.zeros((max_num_reqs,), device="cpu", dtype=torch.int64)
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask_cpu_tensor = torch.ones(
            (max_buffer_num_tokens,),
            device="cpu",
            dtype=torch.bool,
        )
        self.num_actual_tokens_pcp_padded = 0
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()
        self.full_indices = list(
            range(
                self.max_num_tokens * self.pcp_world_size * self.dcp_world_size
                + self.pcp_world_size * self.dcp_world_size * self.max_num_reqs
            )
        )
        if self.speculative_config and self.pcp_world_size * self.dcp_world_size > 1:
            self.input_ids_pcp_full = CpuGpuBuffer(
                self.max_num_tokens, dtype=torch.int32, device=device, pin_memory=pin_memory
            )
            self.query_start_loc_pcp_full = CpuGpuBuffer(
                self.max_num_reqs + 1, dtype=torch.int32, device=device, pin_memory=pin_memory
            )
            self.positions_pcp_full = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device="cpu", pin_memory=pin_memory
            )
            self.positions_pcp_full_np = self.positions_pcp_full.numpy()
            self.query_lens_pcp_full = CpuGpuBuffer(
                self.max_num_reqs, dtype=torch.int32, device=device, pin_memory=pin_memory
            )

    def _get_cumsum_and_arange(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def init_batch_info(
        self,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
    ) -> None:
        self.num_reqs = num_reqs
        is_prefill = num_scheduled_tokens[:num_reqs] > self.decode_threshold
        if not any(is_prefill):
            first_prefill = num_reqs
        else:
            first_prefill = is_prefill.argmax()
        self.num_decode_reqs = first_prefill
        self.num_prefill_reqs = num_reqs - self.num_decode_reqs
        self.num_decode_tokens = num_scheduled_tokens[: self.num_decode_reqs].sum()

    def update_tokens_for_pcp(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update token counts and positions for Prefill Context Parallelism (PCP).

        When using Prefill Context Parallelism, each request's prefill sequence is
        split across multiple PCP ranks. The splitting strategy used here is the
        "DualChunkSwap" style: each request's (padded) sequence is split into
        2 * pcp_world_size chunks and ranks are assigned chunks in an interleaved
        head/tail pattern to balance load.

        This function:
        - Computes how many tokens each request should be processed by the current
          PCP rank (pcp_tokens).
        - Computes the flattened positions of those tokens within the local
          padded buffer (pcp_positions).
        - Updates runner state arrays used to restore original order and mask out
          padded tokens after allgather:
            - self.num_pcp_pads_cpu: number of pads added per request
            - self.pcp_unpad_mask_cpu: boolean mask marking real tokens in the
              padded allgather buffer
            - self.pcp_allgather_restore_idx: index array used to restore original
              ordering after per-rank allgather and interleaving.

        Args:
            num_scheduled_tokens: 1D numpy array of length num_reqs containing
                                  the number of new tokens scheduled per request.
            arange_np: 1D numpy array of length max_buffer_num_tokens used for
                       efficient batched arange operations.

        Returns:
            Tuple (pcp_tokens, pcp_positions):
            - pcp_tokens: number of tokens per request that this PCP rank will
                          actually process (after splitting / replication).
            - pcp_positions: flattened positions for those tokens on this rank,
                             used to build the positions buffer for the model.

        Example:
        >>> Assume tokens = [1, 5, 8], pcp_world_size = 2. After _update_tokens_for_pcp.
        >>> pcp_rank = 0 get ([1, 4, 4], [0, 0, 1, 6, 7, 0, 1, 6, 7])
        >>> pcp_rank = 1 get ([1, 4, 4], [0, 2, 3, 4, 5, 2, 3, 4, 5])
        >>> Meanwhile, the following results are same for each pcp rank
        >>> self.num_pcp_pads_cpu
        [1, 3, 0]
        >>> self.pcp_unpad_mask_cpu
        [True, False, True, True, True, True, True, False, False,
        False, True, True, True, True, True, True, True, True]
        >>> self.pcp_allgather_restore_idx
        [0, 9, 1, 2, 10, 11, 12, 13, 3, 4, 5, 6, 14, 15, 16, 17, 7, 8]
        """

        # DualChunkSwap requires alignment to a multiple of (2 * pcp_world_size).
        # We first pad each request's token count up to that multiple.
        num_padded_scheduled_tokens = np.ceil(num_scheduled_tokens / (2 * self.pcp_world_size)).astype(np.int32) * (
            2 * self.pcp_world_size
        )

        # PCP does not split decode requests. For decode requests, we instead
        # duplicate the scheduled tokens across the pcp_world_size ranks.
        num_padded_scheduled_tokens[: self.num_decode_reqs] = (
            num_scheduled_tokens[: self.num_decode_reqs] * self.pcp_world_size
        )

        # Record how many pads were added per request (padded - original).
        self.num_pcp_pads_cpu[: self.num_reqs] = num_padded_scheduled_tokens - num_scheduled_tokens

        # cu_padded_tokens: cumulative sum of padded token counts,
        # pcp_padded_arange: per-request arange flattened for padded tokens.
        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)
        # Build the mask that marks which positions in the padded allgather buffer
        # correspond to real (unpadded) tokens.
        self.pcp_unpad_mask_cpu[: pcp_padded_arange.shape[0]] = pcp_padded_arange < np.repeat(
            num_scheduled_tokens, num_padded_scheduled_tokens
        )
        unpad_mask_decode = self.pcp_unpad_mask_cpu[: self.num_decode_tokens * self.pcp_world_size]
        unpad_mask_decode = unpad_mask_decode.reshape([-1, self.pcp_world_size])
        unpad_mask_decode[:, 0] = True
        unpad_mask_decode[:, 1:] = False
        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size

        # Compute per-request "chunk sizes" for the head/tail splitting.
        # For prefill requests, we further split the pcp_tokens into two chunks
        # (head and tail). For decode requests, the chunk equals pcp_tokens.
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[: self.num_decode_reqs] = pcp_tokens[: self.num_decode_reqs]

        # Build arange-style helpers for pcp tokens and chunk sizes:
        # - pcp_arange gives indices repeated for each token in pcp_tokens
        # - pcp_chunk_arange gives indices repeated for each position inside chunks
        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes, arange_np)

        # Mask that marks whether a position belongs to the head chunk (True)
        # or the tail chunk (False). For decode requests, tail chunk won't exist
        # and is handled specially below.
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(positions_start_loc: int | np.ndarray, rank: int):
            """
            Compute flattened positions for the given rank with a given start
            offset for each request (positions_start_loc).

            - For head chunks: start at positions_start_loc + rank * chunk_size.
            - For tail chunks: start at positions_start_loc + (2*pcp_world_size- rank -
            1) * chunk_size.
            - For decode requests: no tail chunks; their positions are filled from the
              contiguous (unpadded) `tokens` arange instead (handled after).
            """
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = positions_start_loc + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            # Fill head positions using chunk arange offset by head_start_loc.
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(head_start_loc, pcp_chunk_sizes)
            # Fill tail positions. Note decode requests do not have tail chunks,
            # so the tail filling is only for prefill positions.
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[self.num_decode_tokens :]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[self.num_decode_tokens :]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_world_rank)
        # Decode tokens are duplicated only after AG. But their positions are
        # same without prefill context parallel.
        if self.num_decode_reqs > 0:
            positions[: self.num_decode_tokens] = self._get_cumsum_and_arange(
                num_scheduled_tokens[: self.num_decode_reqs], arange_np
            )[1]

        # Build the restore index used after allgather.
        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions_lst = [
            get_current_rank_positions(padded_pos_start_loc, rank_i) for rank_i in range(self.pcp_world_size)
        ]
        all_positions = np.concatenate(all_positions_lst)
        self.pcp_allgather_restore_idx.np[: all_positions.shape[0]] = all_positions.argsort()
        self.pcp_allgather_restore_idx.copy_to_gpu(all_positions.shape[0])

        self.pcp_tokens[: self.num_reqs] = pcp_tokens[: self.num_reqs]
        self.total_num_sampled_tokens_pcp = pcp_tokens[: self.num_reqs].sum()
        return (
            pcp_tokens[: self.num_reqs],
            positions,
        )

    def get_logits_indices(self, cu_num_tokens: np.ndarray):
        return torch.from_numpy(cu_num_tokens) * self.pcp_world_size - self.num_pcp_pads_cpu_tensor[: self.num_reqs] - 1

    def get_padded_slot_mapping(self, num_tokens: int, num_tokens_padded: int, slot_mapping: torch.Tensor):
        # After pcp allgather and restore, there are padded tokens in kv,
        # so we need pad slotmapping for alignment.
        pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[: num_tokens_padded * self.pcp_world_size]

        cp_unpad_mask = self.pcp_unpad_mask_cpu_tensor[: num_tokens * self.pcp_world_size]
        pcp_padded_slot_mapping.fill_(-1)
        pcp_padded_slot_mapping[: num_tokens * self.pcp_world_size][cp_unpad_mask] = slot_mapping
        return pcp_padded_slot_mapping

    def get_restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ):
        # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
        # ignores the padding from CUDA Graph.
        from vllm.distributed.parallel_state import get_pcp_group

        hidden_states = get_pcp_group().all_gather(
            hidden_states[: self.num_actual_tokens_pcp_padded // self.pcp_world_size],
            0,
        )
        restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
        return torch.index_select(
            hidden_states,
            0,
            restore_idx,
        )

    def generate_pcp_mtp_input(
        self,
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: dict[str, int],
        with_prefill: bool = True,
        input_batch=None,
        arange_np=None,
        req_indices=None,
        positions_np=None,
        cu_num_tokens=None,
        draft_token_ids=None,
        scheduler_output=None,
        num_spec_tokens=None,
    ):
        """
        While pcp > 1, model inputs (input_ids, position, etc.) are split across pcp group,
        but mtp need to shift original input_ids before pcp splitting,
        so we record original input_ids here.
        """
        total_num_scheduled_tokens_pcp_full = total_num_scheduled_tokens
        num_scheduled_tokens_pcp_full = np.empty(self.num_reqs, dtype=np.int32)
        for i, req_id in enumerate(input_batch.req_ids):
            num_scheduled_tokens_pcp_full[i] = num_scheduled_tokens[req_id]
        self.query_lens_pcp_full.cpu[: self.num_reqs] = torch.from_numpy(num_scheduled_tokens_pcp_full)
        req_indices_pcp_full = np.repeat(arange_np[: self.num_reqs], num_scheduled_tokens_pcp_full)
        cu_num_tokens_pcp_full = np.cumsum(num_scheduled_tokens_pcp_full)
        self.query_start_loc_pcp_full.np[0] = 0
        self.query_start_loc_pcp_full.np[1 : self.num_reqs + 1] = cu_num_tokens_pcp_full
        self.query_start_loc_pcp_full.np[self.num_reqs + 1 :].fill(-1)
        cumsums_offsets_pcp_full = np.repeat(
            cu_num_tokens_pcp_full - num_scheduled_tokens_pcp_full, num_scheduled_tokens_pcp_full
        )
        arange_pcp_full = arange_np[:total_num_scheduled_tokens_pcp_full] - cumsums_offsets_pcp_full
        positions_pcp_full_np = self.positions_pcp_full_np[:total_num_scheduled_tokens_pcp_full]
        np.add(input_batch.num_computed_tokens_cpu[req_indices_pcp_full], arange_pcp_full, out=positions_pcp_full_np)
        token_indices_pcp_full = positions_pcp_full_np + req_indices_pcp_full * input_batch.token_ids_cpu.shape[1]
        torch.index_select(
            input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices_pcp_full),
            out=self.input_ids_pcp_full.cpu[:total_num_scheduled_tokens_pcp_full],
        )
        if self.use_async_scheduling:
            self._update_input_ids_pcp_full_ids(
                input_batch,
                draft_token_ids,
                scheduler_output,
                total_num_scheduled_tokens,
                cu_num_tokens_pcp_full,
                num_spec_tokens,
            )
        self.query_lens_pcp_full.copy_to_gpu()
        self.query_start_loc_pcp_full.copy_to_gpu()
        self.input_ids_pcp_full.copy_to_gpu(total_num_scheduled_tokens_pcp_full)
        self.cu_num_tokens_pcp_full = cu_num_tokens_pcp_full
        # For mtpx, pre-allocate mtp slot_mapping here
        if self.decode_threshold > 2 and not with_prefill:
            num_tokens_ori = sum(list(num_scheduled_tokens.values()))
            num_tokens_mtp = num_tokens_ori + self.num_reqs * (self.decode_threshold - 2)
            num_tokens_mtp_pad = num_tokens_mtp * self.pcp_world_size
            req_indices_split = np.array_split(req_indices, cu_num_tokens)[: self.num_reqs]
            positions_split = np.array_split(positions_np, cu_num_tokens)[: self.num_reqs]
            for req_idx in range(self.num_reqs):
                ori_req_indice = req_indices_split[req_idx]
                ori_position = positions_split[req_idx]
                req_indices_split[req_idx] = np.append(
                    ori_req_indice, np.repeat(ori_req_indice[-1], self.decode_threshold - 2)
                )
                positions_split[req_idx] = np.append(
                    ori_position, np.arange(ori_position[-1] + 1, ori_position[-1] + self.decode_threshold - 1)
                )
            req_indices_mtp = np.concatenate(req_indices_split)
            positions_mtp = np.concatenate(positions_split)
            input_batch.block_table.compute_slot_mapping(req_indices_mtp, positions_mtp)
            mtp_slot_ori = input_batch.block_table.block_tables[0].slot_mapping.cpu[:num_tokens_mtp]
            unpad_mask = np.repeat(False, num_tokens_mtp_pad)
            unpad_mask[:: self.pcp_world_size] = True
            mtp_slot_pad = torch.full([num_tokens_mtp_pad], -1, dtype=torch.int32)
            mtp_slot_pad[unpad_mask] = mtp_slot_ori
            self.mtp_slot_pad = mtp_slot_pad.to(self.device, non_blocking=True)

    def _update_input_ids_pcp_full_ids(
        self,
        input_batch,
        draft_token_ids,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
        num_spec_tokens: int,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if input_batch.prev_sampled_token_ids is None or input_batch.prev_req_id_to_index is None:
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = input_batch.prev_req_id_to_index
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(range(flattened_index - draft_len + 1, flattened_index + 1))
                start = prev_index * num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
        num_commmon_tokens = len(sample_flattened_indices)

        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(sample_flattened_indices, dtype=torch.int64)
        prev_common_req_indices_tensor = torch.tensor(prev_common_req_indices, dtype=torch.int64)
        self.input_ids_pcp_full.cpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0].cpu(),
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(spec_flattened_indices, dtype=torch.int64)
        prev_draft_token_indices_tensor = torch.tensor(prev_draft_token_indices, dtype=torch.int64)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = draft_token_ids.to(dtype=torch.int32)

        self.input_ids_pcp_full.cpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor].cpu(),
        )

    def _get_cp_local_seq_lens(
        self,
        seq_lens: torch.Tensor,
        pcp_world_size: int = 1,
        dcp_world_size: int = 1,
        cp_kv_cache_interleave_size: int = 1,
    ) -> torch.Tensor:
        """While using pcp or dcp, kv_cache size stored on each rank may be different,
        use this function to calculate split decode seq_lens of each (p/d)cp rank.
        """
        num_requests = seq_lens.size(0)
        total_world_size = pcp_world_size * dcp_world_size
        seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
        rank_offsets = torch.arange(total_world_size, dtype=torch.int32).unsqueeze(0).repeat(num_requests, 1)
        base = seq_lens_tiled // cp_kv_cache_interleave_size // total_world_size * cp_kv_cache_interleave_size
        remainder = seq_lens_tiled - base * total_world_size
        remainder = torch.clip(
            remainder - rank_offsets * cp_kv_cache_interleave_size,
            0,
            cp_kv_cache_interleave_size,
        )
        dcp_local_seq_lens = (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
        return dcp_local_seq_lens

    def generate_pcp_metadata(
        self,
        total_num_scheduled_tokens: int,
        query_lens: torch.Tensor,
        input_batch: "NPUInputBatch",
        num_scheduled_tokens: np.ndarray | None,
        block_table_tensor: torch.Tensor,
        num_reqs_padded: int,
        num_reqs: int,
    ):
        from vllm_ascend.attention.utils import AscendPrefillContextParallelMetadata

        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_world_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        if self.pcp_world_size * self.dcp_world_size > 1:
            assert num_scheduled_tokens is not None
            decode_context_lens = (
                input_batch.num_computed_tokens_cpu[: self.num_decode_reqs]
                + num_scheduled_tokens[: self.num_decode_reqs]
            )
            prefill_context_lens = input_batch.num_computed_tokens_cpu[self.num_decode_reqs : self.num_reqs]
            context_lens = np.concatenate([decode_context_lens, prefill_context_lens])
            num_computed_tokens_of_pcp_dcp = torch.zeros(
                [self.num_reqs * self.decode_threshold, self.pcp_world_size, self.dcp_world_size],
                dtype=torch.int32,
            )
            # For pcp + spec decode, we flatten seq_lens
            # to avoid irregular attn_mask shape.
            # Same as block_table, we flatten decode seq_lens to query_lens,
            # and keep prefill seq_lens unchanged.
            for decode_idx in range(self.decode_threshold):
                num_computed_tokens_of_pcp_dcp[self.decode_threshold - 1 - decode_idx :: self.decode_threshold] = (
                    self._get_cp_local_seq_lens(
                        torch.tensor(context_lens) - decode_idx,
                        self.pcp_world_size,
                        self.dcp_world_size,
                        self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    )
                )
            ori_query_lens_cpu = None
            if self.decode_threshold > 1:
                num_computed_tokens_of_pcp_dcp_list = []
                if self.num_decode_reqs:
                    num_decodes_flatten = query_lens[: self.num_decode_reqs].sum().item()
                    if query_lens[: self.num_decode_reqs].min().item() == self.decode_threshold:
                        decode_flatten_idx = list(range(num_decodes_flatten))
                    else:
                        decode_flatten_idx = []
                        for req_id in range(self.num_decode_reqs):
                            offset = (req_id + 1) * self.decode_threshold
                            decode_flatten_idx += list(range(offset - query_lens[req_id], offset))
                    num_computed_tokens_of_pcp_dcp_list.append(num_computed_tokens_of_pcp_dcp[decode_flatten_idx])
                if self.num_prefill_reqs:
                    num_computed_tokens_of_pcp_dcp_list.append(
                        num_computed_tokens_of_pcp_dcp[
                            (self.num_decode_reqs + 1) * self.decode_threshold - 1 :: self.decode_threshold
                        ]
                    )
                num_computed_tokens_of_pcp_dcp = torch.cat(num_computed_tokens_of_pcp_dcp_list, dim=0)

                # For pcp + spec decode, we flatten block_table
                # to avoid irregular attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens_cpu = self.query_lens_pcp_full.cpu[:num_reqs_padded]
                ori_query_lens = self.query_lens_pcp_full.gpu[:num_reqs_padded]
                num_prefill_reqs = self.num_prefill_reqs
                num_decode_reqs = self.num_decode_reqs
                num_decode_reqs_flatten = ori_query_lens_cpu[:num_decode_reqs].sum().item()
                block_table_tensor[num_decode_reqs_flatten : num_decode_reqs_flatten + num_prefill_reqs].copy_(
                    block_table_tensor[num_decode_reqs : num_decode_reqs + num_prefill_reqs].clone()
                )
                block_table_tensor[:num_decode_reqs_flatten].copy_(
                    block_table_tensor[:num_decode_reqs].repeat_interleave(ori_query_lens[:num_decode_reqs], dim=0)
                )
                block_table_tensor = block_table_tensor[: num_decode_reqs_flatten + num_prefill_reqs]
                if num_reqs_padded > num_reqs:
                    pad_size = num_reqs_padded - num_reqs
                    ori_query_lens_cpu[-pad_size:] = torch.full([pad_size], ori_query_lens_cpu[-pad_size - 1].item())

            long_seq_metadata = AscendPrefillContextParallelMetadata(
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.numpy(),
            )
            if ori_query_lens_cpu is not None:
                long_seq_metadata.query_lens_pcp_full_cpu = ori_query_lens_cpu
                long_seq_metadata.max_query_len_pcp_full = ori_query_lens_cpu.max().item()
            if self.pcp_world_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                split_with_q_head_nomask_idx_reqs = []
                split_kv_with_q_tail_nomask_idx_reqs = []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                q_req_offset = 0
                kv_req_offset = 0
                q_head_chunk_id = self.pcp_world_rank
                q_tail_chunk_id = self.pcp_world_size * 2 - 1 - self.pcp_world_rank
                for i, seq_len in enumerate(query_lens):
                    if i < self.num_decode_reqs:
                        continue
                    chunk_len = seq_len // 2
                    chunk_seqlens.append(chunk_len)
                    q_head_idx.extend(list(range(q_req_offset, q_req_offset + chunk_len)))
                    kv_with_q_head_nomask_idx.extend(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id))
                    )
                    kv_with_q_head_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_head_chunk_id,
                                kv_req_offset + chunk_len * (q_head_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)
                    split_with_q_head_nomask_idx_reqs.append(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_head_chunk_id))
                    )
                    q_tail_idx.extend(list(range(q_req_offset + chunk_len, q_req_offset + chunk_len * 2)))
                    kv_with_q_tail_nomask_idx.extend(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id))
                    )
                    kv_with_q_tail_mask_idx.extend(
                        list(
                            range(
                                kv_req_offset + chunk_len * q_tail_chunk_id,
                                kv_req_offset + chunk_len * (q_tail_chunk_id + 1),
                            )
                        )
                    )
                    kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)
                    split_kv_with_q_tail_nomask_idx_reqs.append(
                        list(range(kv_req_offset, kv_req_offset + chunk_len * q_tail_chunk_id))
                    )
                    q_req_offset += seq_len
                    kv_req_offset += seq_len * self.pcp_world_size

                q_head_idx_tensor = self._list_to_tensor(q_head_idx, self.device)
                q_tail_idx_tensor = self._list_to_tensor(q_tail_idx, self.device)
                self.q_head_idx_tensor = q_head_idx_tensor
                self.q_tail_idx_tensor = q_tail_idx_tensor

                q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
                q_full_idx = q_full_idx.to(torch.float32).argsort().to(torch.int32)
                self.q_full_idx = q_full_idx

                self.kv_idx_names = {
                    "kv_with_q_head_nomask_idx_tensor": kv_with_q_head_nomask_idx,
                    "kv_with_q_head_mask_idx_tensor": kv_with_q_head_mask_idx,
                    "kv_with_q_tail_nomask_idx_tensor": kv_with_q_tail_nomask_idx,
                    "kv_with_q_tail_mask_idx_tensor": kv_with_q_tail_mask_idx,
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = self._list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_mask_seqlens = torch.tensor([chunk_seqlens, chunk_seqlens], dtype=torch.int32)
                head_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_head_nomask_seqlens], dtype=torch.int32
                )
                tail_attn_nomask_seqlens = torch.tensor(
                    [chunk_seqlens, kv_with_q_tail_nomask_seqlens], dtype=torch.int32
                )
                if self.vllm_config.model_config.use_mla:
                    (
                        split_q_head_nomask_idx_tensor_list,
                        split_q_tail_nomask_idx_tensor_list,
                        head_attn_nomask_seqlens_list,
                        tail_attn_nomask_seqlens_list,
                    ) = self._split_nomask_idx_tensor_list(
                        split_with_q_head_nomask_idx_reqs,
                        split_kv_with_q_tail_nomask_idx_reqs,
                        head_attn_nomask_seqlens,
                        chunk_seqlens,
                    )

                self.extra_long_seq_kwargs = {
                    "attn_mask_seqlens": attn_mask_seqlens,
                    "head_attn_nomask_seqlens": head_attn_nomask_seqlens,
                    "tail_attn_nomask_seqlens": tail_attn_nomask_seqlens,
                }
                long_seq_metadata.pcp_allgather_restore_idx = self.pcp_allgather_restore_idx.gpu[
                    :num_actual_tokens_pcp_padded
                ]
                long_seq_metadata.q_head_idx_tensor = self.q_head_idx_tensor
                long_seq_metadata.q_tail_idx_tensor = self.q_tail_idx_tensor
                long_seq_metadata.q_full_idx = self.q_full_idx
                long_seq_metadata.kv_with_q_head_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_head_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_head_mask_idx_tensor = self.kv_idx_names["kv_with_q_head_mask_idx_tensor"]
                long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = self.kv_idx_names[
                    "kv_with_q_tail_nomask_idx_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_mask_idx_tensor = self.kv_idx_names["kv_with_q_tail_mask_idx_tensor"]
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs["attn_mask_seqlens"]
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs["head_attn_nomask_seqlens"]
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs["tail_attn_nomask_seqlens"]
                if self.vllm_config.model_config.use_mla:
                    long_seq_metadata.kv_with_q_head_nomask_idx_tensor = split_q_head_nomask_idx_tensor_list
                    long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = split_q_tail_nomask_idx_tensor_list
                    long_seq_metadata.head_attn_nomask_seqlens = head_attn_nomask_seqlens_list
                    long_seq_metadata.tail_attn_nomask_seqlens = tail_attn_nomask_seqlens_list

        self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata, block_table_tensor

    def _list_to_tensor(self, lst, device, dtype=torch.int32):
        tensor_npu = torch.zeros(len(lst), dtype=dtype, device=device)
        tensor_npu.copy_(torch.tensor(lst, dtype=dtype), non_blocking=True)
        return tensor_npu

    def _split_nomask_idx_tensor_list(
        self,
        split_with_q_head_nomask_idx_reqs,
        split_kv_with_q_tail_nomask_idx_reqs,
        head_attn_nomask_seqlens,
        chunk_seqlens,
    ):
        split_q_head_nomask_idx_tensor_list, split_q_tail_nomask_idx_tensor_list = [], []
        head_attn_nomask_seqlens_list, tail_attn_nomask_seqlens_list = [], []
        if split_with_q_head_nomask_idx_reqs:
            # In long-sequence scenarios, the computational cost and latency
            # of the _npu_ring_mla operator are not proportional, so we split
            # long sequences into shorter ones to improve performance.
            split_size = 16 * 1024
            if self.pcp_world_rank == 0:
                split_q_head_nomask_idx_list = [self.kv_idx_names["kv_with_q_head_nomask_idx_tensor"]]
            else:
                split_q_head_nomask_idx_list, split_q_head_nomask_lens_list = self._split_multi_batch_kv_idx(
                    split_with_q_head_nomask_idx_reqs, split_size
                )
            split_q_tail_nomask_idx_list, split_q_tail_nomask_lens_list = self._split_multi_batch_kv_idx(
                split_kv_with_q_tail_nomask_idx_reqs, split_size
            )

            for q_head_nomask_idx in split_q_head_nomask_idx_list:
                split_q_head_nomask_idx_tensor_list.append(self._list_to_tensor(q_head_nomask_idx, self.device))

            for q_tail_nomask_idx in split_q_tail_nomask_idx_list:
                split_q_tail_nomask_idx_tensor_list.append(self._list_to_tensor(q_tail_nomask_idx, self.device))

            if self.pcp_world_rank == 0:
                head_attn_nomask_seqlens_list = [head_attn_nomask_seqlens]
            else:
                for q_head_nomask_lens in split_q_head_nomask_lens_list:
                    head_attn_nomask_seqlens_list.append(
                        torch.tensor([chunk_seqlens, q_head_nomask_lens], dtype=torch.int32)
                    )
            for q_tail_nomask_lens in split_q_tail_nomask_lens_list:
                tail_attn_nomask_seqlens_list.append(
                    torch.tensor([chunk_seqlens, q_tail_nomask_lens], dtype=torch.int32)
                )
        return (
            split_q_head_nomask_idx_tensor_list,
            split_q_tail_nomask_idx_tensor_list,
            head_attn_nomask_seqlens_list,
            tail_attn_nomask_seqlens_list,
        )

    def _split_multi_batch_kv_idx(
        self,
        kv_nomask_idx_multi_batch,
        split_size,
    ):
        batch_lengths = [len(batch) for batch in kv_nomask_idx_multi_batch]
        max_batch_length = max(batch_lengths) if batch_lengths else 0
        time = (max_batch_length + split_size - 1) // split_size
        split_kv_idx_3d = []
        split_kv_len_2d = []
        merged_split_kv_idx_3d = []

        for single_batch in kv_nomask_idx_multi_batch:
            current_batch_split = []
            current_batch_len = []
            for t in range(time):
                start = t * split_size
                current_segment = single_batch[start : start + split_size]
                current_batch_split.append(current_segment)
                current_batch_len.append(len(current_segment))

            split_kv_idx_3d.append(current_batch_split)
            split_kv_len_2d.append(current_batch_len)

        for time_idx in range(time):
            current_time_merged = []
            for batch in split_kv_idx_3d:
                current_time_merged.extend(batch[time_idx])
            merged_split_kv_idx_3d.append(current_time_merged)

        def reshape_kv_len_to_time_first(split_kv_len_2d):
            if not split_kv_len_2d or not split_kv_len_2d[0]:
                return []
            return [
                [batch_len[time_idx] for batch_len in split_kv_len_2d] for time_idx in range(len(split_kv_len_2d[0]))
            ]

        merged_split_kv_len_2d = reshape_kv_len_to_time_first(split_kv_len_2d)
        return merged_split_kv_idx_3d, merged_split_kv_len_2d
