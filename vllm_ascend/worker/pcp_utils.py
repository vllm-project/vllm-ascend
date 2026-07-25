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

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.spec_decode.utils import correct_optimistic_seq_lens_cpu
from vllm_ascend.utils import is_pd_decode_recompute_scheduler_enabled
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass(frozen=True)
class PCPSpecDecodeMTPInputs:
    """Device-side PCP state needed by proposer MTP draft steps."""

    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    slot_indices: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass(frozen=True)
class PCPSpecDecodeFirstPassInputs:
    """PCP-adjusted inputs for the first speculative draft pass."""

    num_tokens: int
    input_ids: torch.Tensor
    target_positions: torch.Tensor
    target_hidden_states: torch.Tensor
    token_indices_to_sample: torch.Tensor
    long_seq_args: tuple[torch.Tensor | None, torch.Tensor | None] | None


@dataclass(frozen=True)
class PCPAsyncSpecDecodeRebuildResult:
    """Status returned after trying to rebuild async spec decode CP inputs."""

    rebuilt: bool
    positions_ready_on_device: bool


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
    decode_req_mask: np.ndarray | None = None

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
        use_sparse: bool = False,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_world_rank = pcp_rank
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_rank
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)
        self.pcp_spec_token_offsets = torch.arange(
            max(self.decode_threshold - 1, 1),
            dtype=torch.int64,
            device=device,
        )
        self.pcp_req_offsets = torch.arange(
            max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.pcp_rank_offsets = torch.arange(
            pcp_world_size,
            dtype=torch.int64,
            device=device,
        )
        self.mtp_slot_pad: torch.Tensor | None = None
        self.vllm_config = vllm_config
        self.pd_decode_recompute_scheduler_enabled = is_pd_decode_recompute_scheduler_enabled(vllm_config)
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
        self.pcp_exit_fa_scatter_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.sample_slot_mapping = torch.full(
            (max_buffer_num_tokens,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        self.pcp_padded_slot_mapping_list: list = []  # reinitialized in initialize_slot_mapping
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
        self.use_sparse = use_sparse
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
        self.pcp_fa_query_idx = torch.zeros(
            self.max_num_tokens + 2 * self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.pcp_enter_fa_restore_idx = torch.zeros(
            self.max_num_tokens + 2 * self.pcp_world_size * self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.pcp_fa_padding_restore_idx = torch.zeros(
            self.max_num_tokens * self.pcp_world_size + 2 * self.pcp_world_size * self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.pcp_use_hybrid_attn = self.vllm_config.model_config.hf_config.model_type in (
            "qwen3_next",
            "qwen3_5",
            "qwen3_5_moe",
        )
        self.dcp_mtp_attn_mask = CpuGpuBuffer(
            (max_num_reqs, self.decode_threshold, vllm_config.model_config.max_model_len),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )

        self.pcp_pads_logits_hybrid_attn = torch.ones(self.max_num_reqs, dtype=torch.int32) * (self.pcp_world_size - 1)
        self.pcp_padded_tokens_fla = 0
        self.pcp_padded_tokens_length = 0
        self.num_scheduled_tokens_padded: np.ndarray | None = None
        self.max_num_tokens_across_pcp = 0
        self.total_pcp_padding_tokens_fla = 0
        self.pcp_tokens_padded = None
        self.total_num_scheduled_tokens = 0
        self._local_num_scheduled_tokens: np.ndarray | None = None
        self._local_total_num_scheduled_tokens: int | None = None

        # Full pre-PCP token layout used to rebuild draft slot mapping
        # after async scheduling corrects num_computed_tokens.
        self.async_rebuild_req_indices_full = None
        self.async_rebuild_cu_num_tokens_full = None
        self.async_rebuild_num_tokens_full = 0

        logger.debug(
            "PCP initialized: pcp_world_size=%s, pcp_rank=%s, "
            "dcp_world_size=%s, dcp_rank=%s, "
            "use_sparse=%s, use_async_scheduling=%s, hybrid_attn=%s",
            self.pcp_world_size,
            self.pcp_world_rank,
            self.dcp_world_size,
            self.dcp_world_rank,
            self.use_sparse,
            self.use_async_scheduling,
            self.pcp_use_hybrid_attn,
        )

    @staticmethod
    def _build_fa_padding_restore_idx(
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

    def classify_decode_request_mask(
        self,
        num_scheduled_tokens: np.ndarray | torch.Tensor,
        num_computed_tokens: np.ndarray | torch.Tensor,
        num_prompt_tokens: np.ndarray | torch.Tensor,
        decode_threshold: int,
    ) -> np.ndarray:
        """Return a per-request mask for true decode requests.

        Matches vLLM ``reorder_batch_to_split_decodes_and_prefills``:
        decode = has context, scheduled tokens <= threshold, and prompt finished.
        """

        has_context = num_computed_tokens > 0
        is_below_threshold = num_scheduled_tokens <= decode_threshold
        done_prefilling = num_computed_tokens >= num_prompt_tokens
        if self.pd_decode_recompute_scheduler_enabled:
            # PD D + RecomputeScheduler: KV recv leaves num_computed at N-1.
            done_prefilling = done_prefilling | (num_computed_tokens == num_prompt_tokens - 1)
        return has_context & is_below_threshold & done_prefilling

    def init_batch_info(
        self,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        num_computed_tokens: np.ndarray,
        num_prompt_tokens: np.ndarray,
    ) -> None:
        self.num_reqs = num_reqs
        scheduled = num_scheduled_tokens[:num_reqs]
        self.decode_req_mask = self.classify_decode_request_mask(
            scheduled,
            num_computed_tokens[:num_reqs],
            num_prompt_tokens[:num_reqs],
            self.decode_threshold,
        )
        self.num_decode_reqs = int(self.decode_req_mask.sum())
        self.num_prefill_reqs = num_reqs - self.num_decode_reqs
        self.num_decode_tokens = int(scheduled[: self.num_decode_reqs].sum())
        self.num_scheduled_tokens_padded = num_scheduled_tokens  # for graph compiling in hybrid_attn

        self.query_lens_pcp_full.cpu[: self.num_reqs] = torch.from_numpy(num_scheduled_tokens)
        self.query_lens_pcp_full.cpu[self.num_reqs :].fill_(0)
        self.query_lens_pcp_full.copy_to_gpu()

    def adjust_cu_num_scheduled_tokens_for_pcp(
        self,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray,
    ) -> np.ndarray:
        # Re-align cu_num_scheduled_tokens under PCP hybrid attention so the
        # caller can build correct logits_indices for PCP. Prefill requests
        # need to be padded up to a multiple of (pcp_world_size * 2) tokens,
        # while decode requests are simply multiplied by pcp_world_size and
        # offset by the per-req pcp pads.
        if self.num_prefill_reqs <= 0:
            return cu_num_scheduled_tokens

        prefill_lens = self.pcp_tokens[self.num_decode_reqs : self.num_decode_reqs + self.num_prefill_reqs]
        pads = copy.deepcopy(num_pcp_pads)
        pads[self.num_decode_reqs :] = np.cumsum(pads[self.num_decode_reqs :])
        base = int(cu_num_scheduled_tokens[self.num_decode_reqs - 1]) if self.num_decode_reqs > 0 else 0
        prefill_cu = [base + s for s in accumulate(prefill_lens)]

        cu_num_scheduled_tokens = cu_num_scheduled_tokens.copy()
        cu_num_scheduled_tokens[self.num_decode_reqs :] = prefill_cu
        cu_num_scheduled_tokens[self.num_decode_reqs :] = (
            cu_num_scheduled_tokens[self.num_decode_reqs :] * self.pcp_world_size - pads[self.num_decode_reqs :]
        )
        return cu_num_scheduled_tokens

    def cache_local_schedule_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        total_num_scheduled_tokens: int,
    ) -> None:
        # Copy to decouple from mutable batch arrays.
        self._local_num_scheduled_tokens = num_scheduled_tokens[:num_reqs].copy()
        self._local_total_num_scheduled_tokens = int(total_num_scheduled_tokens)

    def get_local_schedule_layout(
        self,
    ) -> tuple[np.ndarray | None, int | None]:
        return self._local_num_scheduled_tokens, self._local_total_num_scheduled_tokens

    def fill_prompt_embeds_for_pcp(
        self,
        req_embeds: torch.Tensor,
        req_positions_np: np.ndarray,
        dst_slice: torch.Tensor,
    ) -> None:
        valid_mask_np = req_positions_np < req_embeds.shape[0]
        if not valid_mask_np.any():
            return

        if valid_mask_np.all():
            torch.index_select(
                req_embeds,
                0,
                torch.from_numpy(req_positions_np.astype(np.int64)),
                out=dst_slice,
            )
            return

        src_positions = torch.from_numpy(req_positions_np[valid_mask_np].astype(np.int64))
        dst_positions = torch.from_numpy(np.nonzero(valid_mask_np)[0].astype(np.int64))
        dst_slice.index_copy_(0, dst_positions, req_embeds.index_select(0, src_positions))

    def build_local_mm_schedule(
        self,
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray,
        encoder_cache: dict[str, torch.Tensor],
    ) -> tuple[dict[str, list[int]], set[str]]:
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        needed_mm_hashes: set[str] = set()

        req_start_idx = 0
        for req_idx, req_id in enumerate(req_ids):
            if req_idx >= local_num_scheduled_tokens.shape[0]:
                break

            num_sched = int(local_num_scheduled_tokens[req_idx])
            if num_sched <= 0:
                req_start_idx += num_sched
                continue

            req_positions = positions_np[req_start_idx : req_start_idx + num_sched]
            req_state = requests[req_id]
            mm_input_ids = list[int]()

            for mm_input_id, mm_feature in enumerate(req_state.mm_features):
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                end_pos = start_pos + pos_info.length
                mm_hash = mm_feature.identifier

                local_mask = (req_positions >= start_pos) & (req_positions < end_pos)
                if not local_mask.any():
                    continue

                local_indices = np.nonzero(local_mask)[0]
                rel_positions = req_positions[local_indices] - start_pos
                is_embed = pos_info.is_embed
                if is_embed is not None:
                    is_embed_np = is_embed.cpu().numpy()
                    if not is_embed_np[rel_positions].any():
                        continue

                needed_mm_hashes.add(mm_hash)
                if mm_hash not in encoder_cache:
                    mm_input_ids.append(mm_input_id)

            if mm_input_ids:
                scheduled_encoder_inputs[req_id] = mm_input_ids

            req_start_idx += num_sched

        return scheduled_encoder_inputs, needed_mm_hashes

    def gather_mm_embeddings_for_pcp(
        self,
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray,
        shift_computed_tokens: int,
        encoder_cache: dict[str, torch.Tensor],
        is_mm_embed: torch.Tensor,
        model: Any,
        is_multimodal_pruning_enabled: bool,
        uses_mrope: bool,
        warning_once: Callable[..., Any] | None = None,
    ) -> tuple[list[torch.Tensor], bool, bool]:
        mm_embeds = list[torch.Tensor]()
        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_idx, req_id in enumerate(req_ids):
            num_sched = int(local_num_scheduled_tokens[req_idx])
            req_positions = positions_np[req_start_idx : req_start_idx + num_sched]
            if shift_computed_tokens:
                req_positions = req_positions + shift_computed_tokens
            req_state = requests[req_id]
            req_taken_mask = np.zeros(num_sched, dtype=np.bool_)
            mm_embeds_req: list[torch.Tensor] = []
            req_mm_local_indices: list[np.ndarray] = []

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                end_pos = start_pos + pos_info.length
                mm_hash = mm_feature.identifier

                local_mask = (req_positions >= start_pos) & (req_positions < end_pos)
                if not local_mask.any():
                    continue

                local_indices = np.nonzero(local_mask)[0]
                rel_positions = req_positions[local_indices] - start_pos

                is_embed = pos_info.is_embed
                if is_embed is not None:
                    is_embed_np = is_embed.cpu().numpy()
                    keep_mask = is_embed_np[rel_positions]
                    if not keep_mask.any():
                        continue
                    local_indices = local_indices[keep_mask]
                    rel_positions = rel_positions[keep_mask]
                    embed_index_map = np.cumsum(is_embed_np.astype(np.int64)) - 1
                    embed_indices = embed_index_map[rel_positions]
                else:
                    embed_indices = rel_positions

                # OR semantics for overlapping mm features: keep first writer.
                keep_new = ~req_taken_mask[local_indices]
                if not keep_new.any():
                    continue
                local_indices = local_indices[keep_new]
                embed_indices = embed_indices[keep_new]
                req_taken_mask[local_indices] = True

                encoder_output = encoder_cache.get(mm_hash)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
                embed_index_tensor = torch.from_numpy(embed_indices.astype(np.int64)).to(
                    device=encoder_output.device,
                    non_blocking=True,
                )
                mm_embeds_item = torch.index_select(encoder_output, 0, embed_index_tensor)
                mm_embeds_req.append(mm_embeds_item)
                req_mm_local_indices.append(local_indices.astype(np.int64, copy=False))
                is_mm_embed[req_start_idx + local_indices] = True

            if is_multimodal_pruning_enabled and uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = model.recompute_mrope_positions(
                    input_ids=req_state.prompt_token_ids,
                    multimodal_embeddings=mm_embeds_req,
                    mrope_positions=req_state.mrope_positions,
                    num_computed_tokens=req_state.num_computed_tokens,
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            # Keep multimodal embedding order aligned with is_mm_embed scanning order.
            # Under PCP, request positions may be non-monotonic; concatenating by
            # feature order can misalign embeddings with boolean mask traversal.
            if len(mm_embeds_req) > 1:
                total_local_idx = sum(x.size for x in req_mm_local_indices)
                total_embed_rows = sum(x.shape[0] for x in mm_embeds_req)
                if total_local_idx == total_embed_rows and total_local_idx > 0:
                    local_idx_cat = np.concatenate(req_mm_local_indices, axis=0)
                    embed_cat = torch.cat(mm_embeds_req, dim=0)
                    order = np.argsort(local_idx_cat, kind="stable")
                    order_t = torch.from_numpy(order.astype(np.int64)).to(
                        device=embed_cat.device,
                        non_blocking=True,
                    )
                    mm_embeds_req = [embed_cat.index_select(0, order_t)]
                elif warning_once is not None:
                    warning_once(
                        "PCP MM reorder skipped due to size mismatch: local_idx=%d, embed_rows=%d",
                        total_local_idx,
                        total_embed_rows,
                    )

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_sched

        return mm_embeds, should_sync_mrope_positions, should_sync_xdrope_positions

    def maybe_localize_scheduler_output_for_mm_preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids: list[str],
        requests: dict[str, Any],
        positions_np: np.ndarray,
        local_num_scheduled_tokens: np.ndarray | None,
        local_total_num_scheduled_tokens: int | None,
        encoder_cache: dict[str, torch.Tensor],
    ) -> dict[str, Any] | None:
        need_localize = (
            local_total_num_scheduled_tokens is not None
            and local_total_num_scheduled_tokens != scheduler_output.total_num_scheduled_tokens
        )
        if not need_localize and local_num_scheduled_tokens is not None:
            for req_idx, req_id in enumerate(req_ids):
                if req_idx >= local_num_scheduled_tokens.shape[0]:
                    break
                global_sched = scheduler_output.num_scheduled_tokens.get(req_id)
                if global_sched is None or int(global_sched) != int(local_num_scheduled_tokens[req_idx]):
                    need_localize = True
                    break

        if not need_localize:
            return None

        restore_state: dict[str, Any] = {
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "num_scheduled_tokens": scheduler_output.num_scheduled_tokens,
            "scheduled_encoder_inputs": scheduler_output.scheduled_encoder_inputs,
            "free_encoder_mm_hashes": scheduler_output.free_encoder_mm_hashes,
        }

        if local_total_num_scheduled_tokens is not None:
            scheduler_output.total_num_scheduled_tokens = local_total_num_scheduled_tokens

        if local_num_scheduled_tokens is None:
            return restore_state

        num_sched_by_req = dict(scheduler_output.num_scheduled_tokens)
        for req_idx, req_id in enumerate(req_ids):
            if req_idx >= local_num_scheduled_tokens.shape[0]:
                break
            num_sched_by_req[req_id] = int(local_num_scheduled_tokens[req_idx])
        scheduler_output.num_scheduled_tokens = num_sched_by_req

        (
            scheduler_output.scheduled_encoder_inputs,
            local_needed_mm_hashes,
        ) = self.build_local_mm_schedule(
            req_ids=req_ids,
            requests=requests,
            positions_np=positions_np,
            local_num_scheduled_tokens=local_num_scheduled_tokens,
            encoder_cache=encoder_cache,
        )

        # Under PCP, global free list can be earlier than local consumption.
        # Keep MM hashes for all active requests.
        active_mm_hashes = {
            mm_feature.identifier for req_state in requests.values() for mm_feature in req_state.mm_features
        }
        keep_hashes = active_mm_hashes | local_needed_mm_hashes
        scheduler_output.free_encoder_mm_hashes = [
            mm_hash for mm_hash in scheduler_output.free_encoder_mm_hashes if mm_hash not in keep_hashes
        ]

        return restore_state

    def restore_scheduler_output_after_mm_preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        restore_state: dict[str, Any] | None,
    ) -> None:
        if restore_state is None:
            return

        scheduler_output.total_num_scheduled_tokens = restore_state["total_num_scheduled_tokens"]
        scheduler_output.num_scheduled_tokens = restore_state["num_scheduled_tokens"]
        scheduler_output.scheduled_encoder_inputs = restore_state["scheduled_encoder_inputs"]
        scheduler_output.free_encoder_mm_hashes = restore_state["free_encoder_mm_hashes"]

    def initialize_slot_mapping(self) -> None:
        """
        Hyrbid-attention models, such as qwen3_next, have plural kv_cache_groups, which may lead to
        problems like overwriting last group's pcp_padded_slot_mapping, since they share the same
        address. Therefore we need as many pcp_padded_slot_mappings as kv_cache_groups.
        """
        pcp_padded_slot_mapping = torch.full(
            (self.sample_slot_mapping.shape[0],),
            fill_value=-1,
            dtype=torch.int32,
            device=self.sample_slot_mapping.device,
        )
        self.pcp_padded_slot_mapping_list.append(pcp_padded_slot_mapping)

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
                          For hybrid-attention model: number of unpadded tokens
                          per requests
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
        self.pcp_padded_tokens_length = pcp_padded_arange.shape[0]
        # Build the mask that marks which positions in the padded allgather buffer
        # correspond to real (unpadded) tokens.
        self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length] = pcp_padded_arange < np.repeat(
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
        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0

        # Decode tokens are duplicated only after AG. But their positions are
        # same without prefill context parallel.
        if self.num_decode_reqs > 0:
            positions[: self.num_decode_tokens] = self._get_cumsum_and_arange(
                num_scheduled_tokens[: self.num_decode_reqs], arange_np
            )[1]

        # Build the restore index used after allgather.
        all_positions_lst = [
            get_current_rank_positions(padded_pos_start_loc, rank_i) for rank_i in range(self.pcp_world_size)
        ]
        all_positions = np.concatenate(all_positions_lst)
        self.pcp_allgather_restore_idx.np[: all_positions.shape[0]] = all_positions.argsort()
        self.pcp_allgather_restore_idx.copy_to_gpu(all_positions.shape[0])

        self.pcp_tokens[: self.num_reqs] = pcp_tokens[: self.num_reqs]
        self.total_num_sampled_tokens_pcp = pcp_tokens[: self.num_reqs].sum()

        if self.pcp_use_hybrid_attn:
            max_scheduled_prefill_tokens = 0
            self.pcp_padded_tokens_fla = 0
            if self.num_decode_reqs > 0:
                num_padded_scheduled_tokens[: self.num_decode_reqs] = (
                    num_padded_scheduled_tokens[: self.num_decode_reqs] // self.pcp_world_size
                )
            self.total_pcp_padding_tokens_fla = 0
            # have prefills
            if self.num_reqs - self.num_decode_reqs > 0:
                prefill_tokens_tensor = torch.Tensor(num_scheduled_tokens[self.num_decode_reqs :])
                # [num_prefill_reqs, pcp_world_size, 1] [[3,2]] [[2,2,2,1],[2,1,1,1]]
                num_prefill_tokens_allranks = (
                    self._get_cp_local_seq_lens(prefill_tokens_tensor, self.pcp_world_size, 1, 1).long().numpy()
                )
                # [3] [2]  |  [2,2] [2,1] [2,1] [1,1]
                num_prefill_scheduled_tokens_linear = num_prefill_tokens_allranks[:, self.pcp_world_rank, 0]
                num_padded_scheduled_tokens[self.num_decode_reqs :] = num_prefill_scheduled_tokens_linear
                # [[3,5]] | [[0,0,0,0,0],[0,0,0,0,0]]
                num_prefill_tokens_start_loc = np.zeros(
                    (self.num_reqs - self.num_decode_reqs, self.pcp_world_size + 1), dtype=np.int64
                )
                # [[0,3,5]] | [[0,2,4,6,7],[0,2,3,4,5]]
                num_prefill_tokens_start_loc[:, 1:] = np.cumsum(num_prefill_tokens_allranks[..., 0], axis=-1)
                # [0] [3] | [0,0] [2,2] [4,3] [6,4] [7,5]
                num_prefill_tokens_cu_ranks = num_prefill_tokens_start_loc[:, self.pcp_world_rank]
                # [0,1,2] [0,1] | [0,1,0,1] [0,1,0] [0,1,0] [0,0]
                # -> [0,1,2] [3,4] | [0,1,0,1] [2,3,2] [4,5,3] [6,4]
                _, positions_linear = self._get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)
                positions_linear[self.num_decode_tokens :] = positions_linear[self.num_decode_tokens :] + np.repeat(
                    num_prefill_tokens_cu_ranks, num_prefill_scheduled_tokens_linear
                )

                max_scheduled_prefill_tokens = num_prefill_tokens_allranks[:, 0, 0].sum()
                num_prefill_tokens = num_scheduled_tokens[self.num_decode_reqs :].sum()
                self.total_pcp_padding_tokens_fla = (
                    max_scheduled_prefill_tokens * self.pcp_world_size - num_prefill_tokens
                )
                self.pcp_padded_tokens_fla += max_scheduled_prefill_tokens - num_prefill_scheduled_tokens_linear.sum()

            max_scheduled_tokens = max_scheduled_prefill_tokens + self.num_decode_tokens
            enter_fa_prefill_restore_idx = None
            if self.num_reqs - self.num_decode_reqs > 0:
                # prefill reorder idx
                # [[3,2]] [[2,2,2,1],[2,2,1,1],[1,1,1,1]]
                num_prefill_tokens_allranks = num_prefill_tokens_allranks[..., 0]
                # [0,1,2,0,1] [0,1,0,1,0,1,0,|0,1,0,1,0,0]
                _, prefill_arange_allranks = self._get_cumsum_and_arange(
                    num_prefill_tokens_allranks.flatten(), arange_np
                )
                # [0,1] [0,1,2,3,0,1,2,3]
                _, prefill_rank_offset = self._get_cumsum_and_arange(
                    np.ones(self.num_reqs - self.num_decode_reqs, dtype=np.int64) * self.pcp_world_size, arange_np
                )
                # [0,0,0,3,3] [0,M,2M,3M,0,M,2M,3M] -> [0,0,M,M,2M,2M,3M,0,0,M,M,2M,3M] + D
                prefill_all_offset = (
                    np.repeat(prefill_rank_offset * max_scheduled_tokens, num_prefill_tokens_allranks.flatten())
                    + self.num_decode_tokens
                )

                # [0,0,0,0,|2,2,2,1,|4,4,3,2] -> [0,0,0,0,0,0,0,|2,2,2,2,2,1,|4,4,3,2]
                # [[0,0]] -> [0,0,0,0,0]
                prefill_local_start_local = np.zeros_like(num_prefill_tokens_allranks)
                prefill_local_start_local[1:, :] = np.cumsum(num_prefill_tokens_allranks, axis=0)[:-1, :]
                prefill_local_offset = np.repeat(
                    prefill_local_start_local.flatten(), num_prefill_tokens_allranks.flatten()
                )
                prefill_all_offset = np.add(prefill_all_offset, prefill_local_offset)
                # [0,1,2,3,4]  [0,1,M,M+1,2M,2M+1,3M,0,1,M,M+1,2M,3M]
                enter_fa_prefill_restore_idx = np.add(prefill_all_offset, prefill_arange_allranks)
            else:
                _, positions_linear = self._get_cumsum_and_arange(num_padded_scheduled_tokens, arange_np)

            # decode reorder idx
            enter_fa_decode_restore_idx = None
            if self.num_decode_reqs > 0:
                if self.pcp_use_hybrid_attn and self.speculative_config:
                    # hybrid attn model has different position assignment for decode tokens.
                    decode_reqs_offset = np.tile(np.arange(self.num_decode_tokens, dtype=np.int64), self.pcp_world_size)
                    decode_ranks_offset = (
                        np.repeat(np.arange(self.pcp_world_size, dtype=np.int64), self.num_decode_tokens)
                        * max_scheduled_tokens
                    )
                else:
                    num_decode_pcp_size = np.ones(self.num_decode_reqs, dtype=np.int64) * self.pcp_world_size
                    decode_reqs_offset = np.repeat(np.arange(self.num_decode_reqs, dtype=np.int64), num_decode_pcp_size)
                    decode_ranks_offset = (
                        self._get_cumsum_and_arange(num_decode_pcp_size, arange_np)[1] * max_scheduled_tokens
                    )
                enter_fa_decode_restore_idx = np.add(decode_reqs_offset, decode_ranks_offset)

            if enter_fa_decode_restore_idx is not None and enter_fa_prefill_restore_idx is not None:
                pcp_enter_fa_restore_idx = torch.from_numpy(
                    np.concatenate([enter_fa_decode_restore_idx, enter_fa_prefill_restore_idx])
                )
            elif enter_fa_decode_restore_idx is not None:
                pcp_enter_fa_restore_idx = torch.from_numpy(enter_fa_decode_restore_idx)

            elif enter_fa_prefill_restore_idx is not None:
                pcp_enter_fa_restore_idx = torch.from_numpy(enter_fa_prefill_restore_idx)
            self.pcp_enter_fa_restore_idx[: pcp_enter_fa_restore_idx.shape[0]].copy_(
                pcp_enter_fa_restore_idx.long(), non_blocking=True
            )
            pcp_unpad_mask = self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length]
            pcp_fa_padding_restore_idx = self._build_fa_padding_restore_idx(
                pcp_unpad_mask,
                self.num_decode_tokens * self.pcp_world_size,
                pcp_enter_fa_restore_idx.shape[0],
            )
            if pcp_fa_padding_restore_idx is not None:
                self.pcp_fa_padding_restore_idx[: pcp_fa_padding_restore_idx.shape[0]].copy_(
                    torch.from_numpy(pcp_fa_padding_restore_idx),
                    non_blocking=True,
                )

            if self.num_reqs > self.num_decode_reqs:
                all_positions_prefill = [
                    get_current_rank_positions(padded_pos_start_loc, rank_i)[self.num_decode_tokens :]
                    - self.num_decode_tokens * self.pcp_world_size
                    for rank_i in range(self.pcp_world_size)
                ]
                all_positions_prefill_tensor = torch.from_numpy(np.concatenate(all_positions_prefill))
                all_exit_fa_restore_idx = all_positions_prefill_tensor.float().argsort()
                unpad_mask_prefill = self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length][
                    self.num_decode_tokens * self.pcp_world_size :
                ]
                # [0] | [0,7]
                ori_tokens_start_loc = np.roll(np.cumsum(num_scheduled_tokens[self.num_decode_reqs :]), 1)
                ori_tokens_start_loc[0] = 0
                # [0,1,2] [3,4] | [0,1,7,8] [2,3,9] [4,5,10] [6,11]
                exit_fa_scatter_indices = positions_linear[self.num_decode_tokens :] + np.repeat(
                    ori_tokens_start_loc, num_prefill_scheduled_tokens_linear
                )

                exit_fa_scatter_idx = torch.index_select(
                    all_exit_fa_restore_idx[unpad_mask_prefill], 0, torch.from_numpy(exit_fa_scatter_indices)
                )
                self.pcp_exit_fa_scatter_idx.gpu[: exit_fa_scatter_idx.shape[0]].copy_(
                    exit_fa_scatter_idx.long(), non_blocking=True
                )

                positions_prefill = all_positions_prefill[self.pcp_world_rank]
                pcp_fa_query_idx_tensor = torch.from_numpy(positions_prefill)
                self.pcp_fa_query_idx[: pcp_fa_query_idx_tensor.shape[0]].copy_(
                    pcp_fa_query_idx_tensor.long(), non_blocking=True
                )
            self.pcp_tokens[: self.num_reqs] = pcp_tokens[: self.num_reqs]
            self.total_num_sampled_tokens_pcp = num_scheduled_tokens[: self.num_reqs].sum()
            self.max_num_tokens_across_pcp = max_scheduled_tokens
            self.pcp_tokens_padded = pcp_tokens[: self.num_reqs]
            self.num_scheduled_tokens_padded = np.array(self.pcp_tokens_padded, dtype=np.int32)
            self.total_num_scheduled_tokens = num_padded_scheduled_tokens[: self.num_reqs].sum()
            return num_padded_scheduled_tokens, positions_linear
        return pcp_tokens[: self.num_reqs], positions

    def get_logits_indices(
        self,
        cu_num_tokens: np.ndarray,
        num_reqs: int,
        tokens_original: list[int] | None = None,
    ):
        if not self.pcp_use_hybrid_attn or tokens_original is None:
            logits_indices = (
                torch.from_numpy(cu_num_tokens) * self.pcp_world_size
                - self.num_pcp_pads_cpu_tensor[: self.num_reqs]
                - 1
            )
        else:
            tokens_original_tensor = torch.tensor(tokens_original, dtype=torch.int32)
            assert self.decode_req_mask is not None
            num_decode_reqs = int(self.decode_req_mask.sum())
            decode_pads = self.pcp_pads_logits_hybrid_attn[:num_decode_reqs]
            pad_len = tokens_original_tensor.shape[0] - num_decode_reqs
            tokens_logits = tokens_original_tensor + F.pad(decode_pads, (0, pad_len), value=0)
            logits_indices = torch.cumsum(tokens_logits, dim=0) - 1
        return logits_indices

    def get_padded_slot_mapping(
        self,
        num_tokens: int,
        num_tokens_padded: int,
        slot_mapping: torch.Tensor,
        kv_cache_group_id: int,
    ):
        # After pcp allgather and restore, there are padded tokens in kv,
        # so we need pad slotmapping for alignment.
        pcp_padded_slot_mapping = self.pcp_padded_slot_mapping_list[kv_cache_group_id]
        if self.pcp_use_hybrid_attn:
            assert self.num_scheduled_tokens_padded is not None
            num_tokens = self.num_scheduled_tokens_padded.sum()
        if not self.pcp_use_hybrid_attn or self.total_num_sampled_tokens_pcp != num_tokens_padded:
            pcp_padded_slot_mapping = pcp_padded_slot_mapping[: num_tokens_padded * self.pcp_world_size]
        else:
            pcp_padded_slot_mapping = pcp_padded_slot_mapping[: num_tokens * self.pcp_world_size]
        cp_unpad_mask = self.pcp_unpad_mask_cpu_tensor[: num_tokens * self.pcp_world_size]
        pcp_padded_slot_mapping.fill_(-1)
        pcp_padded_slot_mapping[: num_tokens * self.pcp_world_size][cp_unpad_mask] = slot_mapping
        return pcp_padded_slot_mapping

    def get_restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
        num_input_tokens: int | None = None,
    ) -> torch.Tensor:
        """Gather PCP hidden states and restore the original token order.

        ``num_input_tokens`` is explicit for spec decode, where draft graph
        padding can differ from the main-model PCP scheduled-token length.
        Main-model callers omit it and use the PCP metadata length.
        """
        from vllm.distributed.parallel_state import get_pcp_group

        if not self.pcp_use_hybrid_attn:
            local_num_tokens = (
                num_input_tokens
                if num_input_tokens is not None
                else self.num_actual_tokens_pcp_padded // self.pcp_world_size
            )
            hidden_states = get_pcp_group().all_gather(
                hidden_states[:local_num_tokens],
                0,
            )
            restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
            return torch.index_select(
                hidden_states,
                0,
                restore_idx,
            )
        else:
            if num_input_tokens is not None:
                hidden_states = hidden_states[:num_input_tokens]
            if hidden_states.shape[0] == self.total_num_scheduled_tokens and self.pcp_padded_tokens_fla > 0:
                hidden_states = F.pad(
                    hidden_states, pad=(0, 0, 0, self.pcp_padded_tokens_fla), mode="constant", value=0
                )
            hidden_states = (
                hidden_states[: self.max_num_tokens_across_pcp] if self.max_num_tokens_across_pcp > 0 else hidden_states
            )
            hidden_states = get_pcp_group().all_gather(hidden_states.contiguous(), dim=0)
            restore_idx = self.pcp_enter_fa_restore_idx[: hidden_states.shape[0] - self.total_pcp_padding_tokens_fla]
            return torch.index_select(hidden_states, 0, restore_idx)

    def mask_spec_decode_restore_idx_for_graph(
        self,
        pcp_allgather_restore_idx: torch.Tensor,
    ) -> None:
        """Mask graph-only PCP restore slots used by padded draft batches."""
        index = torch.arange(
            pcp_allgather_restore_idx.shape[0],
            dtype=torch.int64,
            device=pcp_allgather_restore_idx.device,
        )
        mask = (index % (self.pcp_world_size * self.decode_threshold)) >= self.decode_threshold
        pcp_allgather_restore_idx[mask] = 0
        restore_len = pcp_allgather_restore_idx.shape[0]
        self.pcp_allgather_restore_idx.gpu[:restore_len].copy_(pcp_allgather_restore_idx)
        self.pcp_allgather_restore_idx.gpu[restore_len:].fill_(0)

    def get_spec_decode_decode_hidden_states(
        self,
        target_hidden_states_d_padded: torch.Tensor,
        num_decode_reqs: int,
        num_decode_tokens: int | None = None,
    ) -> torch.Tensor:
        """Remove PCP decode padding from target hidden states for proposer input."""
        if num_decode_tokens is None:
            num_decode_tokens = self.num_decode_tokens
        if num_decode_tokens == 0:
            return target_hidden_states_d_padded
        if self.pcp_use_hybrid_attn:
            return target_hidden_states_d_padded[:num_decode_tokens]

        query_start_loc = self.query_start_loc_pcp_full.gpu[: num_decode_reqs + 1]
        decode_req_starts = query_start_loc[:num_decode_reqs].to(torch.int64)
        decode_query_lens = (query_start_loc[1 : num_decode_reqs + 1] - query_start_loc[:num_decode_reqs]).to(
            torch.int64
        )
        decode_padded_starts = decode_req_starts * self.pcp_world_size
        decode_req_starts_per_token = torch.repeat_interleave(
            decode_req_starts,
            decode_query_lens,
            output_size=num_decode_tokens,
        )
        decode_padded_starts_per_token = torch.repeat_interleave(
            decode_padded_starts,
            decode_query_lens,
            output_size=num_decode_tokens,
        )
        decode_offsets = (
            torch.arange(
                num_decode_tokens,
                dtype=torch.int64,
                device=target_hidden_states_d_padded.device,
            )
            - decode_req_starts_per_token
        )
        decode_hidden_state_indices = decode_padded_starts_per_token + decode_offsets
        return target_hidden_states_d_padded[decode_hidden_state_indices]

    def prepare_spec_decode_first_pass_inputs(
        self,
        input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        common_attn_metadata: Any,
        long_seq_metadata: Any | None,
        req_scheduled_tokens: dict[str, int] | None,
        req_ids: list[str],
        logits_indices: torch.Tensor,
        num_tokens: int,
        num_prefill_reqs: int,
        num_decode_reqs: int,
        uses_mrope: bool,
    ) -> PCPSpecDecodeFirstPassInputs:
        """Prepare CP-adjusted proposer inputs for the first draft pass."""
        long_seq_args: tuple[torch.Tensor | None, torch.Tensor | None] | None = None
        if self.pcp_world_size * self.dcp_world_size <= 1:
            return PCPSpecDecodeFirstPassInputs(
                num_tokens=num_tokens,
                input_ids=input_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                long_seq_args=long_seq_args,
            )

        assert long_seq_metadata is not None
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        ori_token_indices_to_sample = token_indices_to_sample.clone()
        query_lens_d = self.query_lens_pcp_full.cpu[:num_decode_reqs]
        long_seq_args = (query_lens_d, ori_token_indices_to_sample)

        if self.pcp_world_size <= 1:
            return PCPSpecDecodeFirstPassInputs(
                num_tokens=num_tokens,
                input_ids=input_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                long_seq_args=long_seq_args,
            )

        num_tokens_d = self.num_decode_tokens
        num_tokens_d_padded = num_tokens_d * self.pcp_world_size
        input_ids_d = input_ids[:num_tokens_d]
        input_ids_p = input_ids[num_tokens_d:num_tokens]
        target_hidden_states_d_padded = target_hidden_states[:num_tokens_d_padded]
        if num_tokens_d:
            target_hidden_states_d = self.get_spec_decode_decode_hidden_states(
                target_hidden_states_d_padded,
                num_decode_reqs,
                num_tokens_d,
            )
        else:
            target_hidden_states_d = target_hidden_states_d_padded
        target_hidden_states_p = target_hidden_states[num_tokens_d_padded:]

        req_scheduled_tokens_p: dict[str, int] = {}
        if num_prefill_reqs:
            assert req_scheduled_tokens is not None
            num_reqs = num_decode_reqs + num_prefill_reqs
            for i, req_id in enumerate(req_ids[:num_reqs]):
                if i >= num_decode_reqs:
                    req_scheduled_tokens_p[req_id] = req_scheduled_tokens[req_id]

        (
            num_tokens_p,
            input_ids_p,
            target_hidden_states_p,
            max_query_len_p,
            seq_lens_p,
            cu_num_tokens_p,
        ) = self._split_spec_decode_pcp_prefill_input(
            req_scheduled_tokens_p,
            input_ids_p,
            target_hidden_states_p,
        )
        num_tokens = num_tokens_d + num_tokens_p
        if uses_mrope:
            target_positions = target_positions[:, :num_tokens]
        else:
            target_positions = target_positions[:num_tokens]
        input_ids = torch.cat([input_ids_d, input_ids_p], dim=0)
        target_hidden_states = torch.cat([target_hidden_states_d, target_hidden_states_p], dim=0)

        if num_decode_reqs:
            token_indices_to_sample[:num_decode_reqs] = logits_indices[token_indices_to_sample[:num_decode_reqs]]
        if num_prefill_reqs:
            token_indices_to_sample[-num_prefill_reqs:] = logits_indices[-num_prefill_reqs:]
            common_attn_metadata.num_actual_tokens = num_tokens
            common_attn_metadata.max_query_len = max(self.decode_threshold, max_query_len_p)
            common_attn_metadata.seq_lens[-num_prefill_reqs:] = seq_lens_p
            if common_attn_metadata.seq_lens_cpu is not None:
                common_attn_metadata.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
            query_start_loc_p = cu_num_tokens_p[1:] + common_attn_metadata.query_start_loc_cpu[num_decode_reqs].item()
            common_attn_metadata.query_start_loc[-num_prefill_reqs:] = query_start_loc_p
            common_attn_metadata.query_start_loc_cpu[-num_prefill_reqs:] = query_start_loc_p

        return PCPSpecDecodeFirstPassInputs(
            num_tokens=num_tokens,
            input_ids=input_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            long_seq_args=long_seq_args,
        )

    def _split_spec_decode_pcp_prefill_input(
        self,
        req_scheduled_tokens: dict[str, int],
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Split prefill input_ids and target_hidden_states in the PCP group.

        The target hidden states already include PCP padding; this method
        selects only the local PCP rank's prefill tokens and returns the
        attention metadata fields affected by that split.
        """
        if len(req_scheduled_tokens) == 0:
            return (
                0,
                input_ids.new_zeros((0,)),
                target_hidden_states.new_zeros((0, target_hidden_states.size(1))),
                0,
                torch.zeros((0,), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )

        if self.pcp_use_hybrid_attn:
            return self._split_spec_decode_pcp_prefill_input_hybrid(
                req_scheduled_tokens,
                input_ids,
                target_hidden_states,
            )

        def _pcp_pad_and_split(num_tokens: int) -> tuple[list[int], int, int]:
            num_pcp_padded_scheduled_tokens = cdiv(num_tokens, 2 * self.pcp_world_size) * 2 * self.pcp_world_size
            pcp_pad = num_pcp_padded_scheduled_tokens - num_tokens
            chunk_size = num_pcp_padded_scheduled_tokens // (2 * self.pcp_world_size)

            req_position_cp: list[int] = []
            req_position_cp.extend(
                self.full_indices[self.pcp_world_rank * chunk_size : (self.pcp_world_rank + 1) * chunk_size]
            )
            req_position_cp.extend(
                self.full_indices[
                    num_pcp_padded_scheduled_tokens
                    - (self.pcp_world_rank + 1) * chunk_size : num_pcp_padded_scheduled_tokens
                    - self.pcp_world_rank * chunk_size
                ]
            )

            return req_position_cp, num_pcp_padded_scheduled_tokens, pcp_pad

        num_pcp_scheduled_tokens = []
        ori_start_index = 0
        pad_start_index = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            req_position_pcp, num_pcp_padded_scheduled_tokens, num_pcp_pad = _pcp_pad_and_split(ori_num_tokens)
            actual_num_tokens = len(req_position_pcp)
            num_pcp_scheduled_tokens.append(actual_num_tokens)
            pad_input_ids = F.pad(
                input_ids[ori_start_index : ori_start_index + ori_num_tokens],
                (0, num_pcp_pad),
            )
            ori_start_index += ori_num_tokens
            pcp_chunk_indices = [pad_start_index + pos for pos in req_position_pcp]
            pcp_split_input_ids = pad_input_ids[req_position_pcp]
            pcp_split_hidden_states = target_hidden_states[pcp_chunk_indices]
            pcp_split_input_ids_list.append(pcp_split_input_ids)
            pcp_split_hidden_states_list.append(pcp_split_hidden_states)
            pad_start_index += num_pcp_padded_scheduled_tokens
        num_tokens = sum(num_pcp_scheduled_tokens)
        input_ids = torch.cat(pcp_split_input_ids_list)
        target_hidden_states = torch.cat(pcp_split_hidden_states_list, dim=0)
        max_query_len = max(num_pcp_scheduled_tokens)
        seq_lens = torch.tensor(num_pcp_scheduled_tokens, dtype=torch.int32)
        cu_num_tokens = torch.tensor(np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens

    def _split_spec_decode_pcp_prefill_input_hybrid(
        self,
        req_scheduled_tokens: dict[str, int],
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Linear-split prefill inputs for hybrid-attention PCP models."""
        num_pcp_scheduled_tokens = []
        global_offset = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            padded_tokens = cdiv(ori_num_tokens, 2 * self.pcp_world_size) * 2 * self.pcp_world_size
            pcp_tokens = padded_tokens // self.pcp_world_size
            num_pads = padded_tokens - ori_num_tokens
            rank_start = self.pcp_world_rank * pcp_tokens
            num_pcp_scheduled_tokens.append(pcp_tokens)

            req_input_ids = input_ids[global_offset : global_offset + ori_num_tokens]
            if num_pads > 0:
                req_input_ids = F.pad(req_input_ids, (0, num_pads))
            pcp_split_input_ids_list.append(req_input_ids[rank_start : rank_start + pcp_tokens])

            req_hidden = target_hidden_states[global_offset : global_offset + ori_num_tokens]
            if num_pads > 0:
                req_hidden = F.pad(req_hidden, (0, 0, 0, num_pads))
            pcp_split_hidden_states_list.append(req_hidden[rank_start : rank_start + pcp_tokens])
            global_offset += ori_num_tokens
        num_tokens = sum(num_pcp_scheduled_tokens)
        input_ids = torch.cat(pcp_split_input_ids_list)
        target_hidden_states = torch.cat(pcp_split_hidden_states_list, dim=0)
        max_query_len = max(num_pcp_scheduled_tokens)
        seq_lens = torch.tensor(num_pcp_scheduled_tokens, dtype=torch.int32)
        cu_num_tokens = torch.tensor(np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens

    def _get_spec_decode_mtp_slot_inputs(
        self,
        ori_token_indices_to_sample: torch.Tensor,
        num_reqs: int,
        num_speculative_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build device-side CP slot indices for MTP draft requests."""
        assert self.mtp_slot_pad is not None
        query_start_loc = self.query_start_loc_pcp_full.gpu[: num_reqs + 1]
        req_starts = query_start_loc[:num_reqs].to(torch.int64)
        cu_num_tokens = query_start_loc[1 : num_reqs + 1].to(torch.int64)
        query_lens = cu_num_tokens - req_starts
        num_reject_tokens = cu_num_tokens - ori_token_indices_to_sample.to(torch.int64) - 1
        num_accept_tokens = query_lens - num_reject_tokens
        slot_idx_base = (
            req_starts * self.pcp_world_size
            + self.pcp_req_offsets[:num_reqs] * (num_speculative_tokens - 1) * self.pcp_world_size
            + (num_accept_tokens - 1) * self.pcp_world_size
        )
        slot_indices = (slot_idx_base[:, None] + self.pcp_rank_offsets[: self.pcp_world_size]).reshape(-1)
        return slot_indices, self.mtp_slot_pad

    def prepare_spec_decode_mtp_drafting_inputs(
        self,
        common_attn_metadata: Any,
        attn_metadata: Any,
        ori_token_indices_to_sample: torch.Tensor | None,
        batch_size: int,
        num_decode_reqs: int,
        is_prefill_batch: bool,
        num_speculative_tokens: int,
    ) -> PCPSpecDecodeMTPInputs | None:
        """Prepare CP MTP metadata for decode and DCP-prefill batches."""
        is_decode_only_batch = num_decode_reqs > 0 and not is_prefill_batch
        is_dcp_prefill_batch = self.pcp_world_size == 1 and self.dcp_world_size > 1 and is_prefill_batch
        if num_speculative_tokens <= 1 or not (is_decode_only_batch or is_dcp_prefill_batch):
            return None

        assert ori_token_indices_to_sample is not None
        num_reqs = batch_size if is_dcp_prefill_batch else num_decode_reqs
        slot_indices, slot_mapping = self._get_spec_decode_mtp_slot_inputs(
            ori_token_indices_to_sample,
            num_reqs,
            num_speculative_tokens,
        )

        seq_lens = getattr(attn_metadata, "seq_lens", None)
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens is None:
            assert seq_lens_cpu is not None
            seq_lens = seq_lens_cpu
        seq_lens = seq_lens[:batch_size].clone()
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:batch_size].clone()

        common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor[:batch_size]
        return PCPSpecDecodeMTPInputs(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_indices=slot_indices,
            slot_mapping=slot_mapping,
        )

    def rebuild_async_spec_decode_inputs(
        self,
        *,
        use_async_spec_decode: bool,
        valid_sampled_token_count_gpu: torch.Tensor | None,
        prev_req_id_to_index: Any,
        prev_positions_gpu: torch.Tensor | None,
        with_prefill: bool,
        enable_prompt_embeds: bool,
        has_req_prompt_embeds: bool,
        supports_mm_inputs: bool,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        req_indices: np.ndarray,
        req_indices_gpu: torch.Tensor,
        position_pcp: np.ndarray | None,
        query_pos_gpu: torch.Tensor,
        query_pos_np: np.ndarray,
        positions: torch.Tensor,
        positions_np: np.ndarray,
        num_computed_tokens: torch.Tensor,
        num_computed_tokens_cpu: np.ndarray,
        prev_positions_np: np.ndarray,
        prev_num_draft_tokens_np: np.ndarray,
        valid_sampled_token_count_event: Any | None,
        valid_sampled_token_count_cpu: torch.Tensor | None,
        input_batch: NPUInputBatch,
        input_ids: CpuGpuBuffer,
        scheduler_output: "SchedulerOutput",
        arange_np: np.ndarray,
        cu_num_tokens: np.ndarray,
        draft_token_ids: torch.Tensor | None,
        num_spec_tokens: int,
        prepare_input_ids: Callable[["SchedulerOutput", int, int, np.ndarray], None],
    ) -> PCPAsyncSpecDecodeRebuildResult:
        """Rebuild CP/spec inputs after async accepted-token correction."""
        should_rebuild = (
            self.pcp_world_size * self.dcp_world_size > 1
            and use_async_spec_decode
            and valid_sampled_token_count_gpu is not None
            and bool(prev_req_id_to_index)
            and self.num_decode_reqs > 0
        )
        if not should_rebuild:
            return PCPAsyncSpecDecodeRebuildResult(
                rebuilt=False,
                positions_ready_on_device=False,
            )

        can_rebuild_on_device = (
            prev_positions_gpu is not None
            and not with_prefill
            and not enable_prompt_embeds
            and not has_req_prompt_embeds
            and not supports_mm_inputs
        )
        if can_rebuild_on_device:
            if self.pcp_world_size > 1:
                assert position_pcp is not None
                position_offsets_gpu = (
                    torch.from_numpy(position_pcp[:total_num_scheduled_tokens])
                    .pin_memory()
                    .to(
                        dtype=torch.int64,
                        device=self.device,
                        non_blocking=True,
                    )
                )
            else:
                position_offsets_gpu = query_pos_gpu[:total_num_scheduled_tokens].to(torch.int64)
            positions_gpu = num_computed_tokens[req_indices_gpu].to(torch.int64) + position_offsets_gpu
            positions[:total_num_scheduled_tokens].copy_(positions_gpu)

            num_tokens_full = self.async_rebuild_num_tokens_full
            query_start_loc_full = self.query_start_loc_pcp_full.gpu[: num_reqs + 1]
            query_lens_full = (query_start_loc_full[1:] - query_start_loc_full[:-1]).to(torch.int64)
            req_indices_full_gpu = torch.repeat_interleave(
                self.pcp_req_offsets[:num_reqs],
                query_lens_full,
                output_size=num_tokens_full,
            )
            token_offsets_full = torch.arange(
                num_tokens_full,
                dtype=torch.int64,
                device=self.device,
            )
            positions_full_gpu = (
                num_computed_tokens[req_indices_full_gpu].to(torch.int64)
                + token_offsets_full
                - query_start_loc_full[req_indices_full_gpu].to(torch.int64)
            )

            if self.pcp_world_size > 1:
                input_batch.block_table.compute_slot_mapping(
                    num_reqs,
                    query_start_loc_full,
                    positions_full_gpu,
                )

            extra_tokens = self.decode_threshold - 2
            if extra_tokens > 0 and not with_prefill:
                mtp_lens = query_lens_full + extra_tokens
                num_tokens_mtp = num_tokens_full + num_reqs * extra_tokens
                req_indices_mtp = torch.repeat_interleave(
                    self.pcp_req_offsets[:num_reqs],
                    mtp_lens,
                    output_size=num_tokens_mtp,
                )
                mtp_start_loc = torch.empty(
                    num_reqs + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                mtp_start_loc[0].fill_(0)
                mtp_start_loc[1:] = torch.cumsum(mtp_lens, dim=0)
                mtp_offsets = torch.arange(
                    num_tokens_mtp,
                    dtype=torch.int64,
                    device=self.device,
                )
                positions_mtp = (
                    num_computed_tokens[req_indices_mtp].to(torch.int64) + mtp_offsets - mtp_start_loc[req_indices_mtp]
                )
                input_batch.block_table.compute_slot_mapping_draft(
                    req_indices_mtp,
                    positions_mtp,
                )
                mtp_slot_ori = input_batch.block_table.block_tables[0].slot_mapping.gpu[:num_tokens_mtp]
                num_tokens_mtp_pad = num_tokens_mtp * self.pcp_world_size
                if self.mtp_slot_pad is None or self.mtp_slot_pad.numel() < num_tokens_mtp_pad:
                    self.mtp_slot_pad = torch.empty(
                        num_tokens_mtp_pad,
                        dtype=torch.int32,
                        device=self.device,
                    )
                mtp_slot_pad = self.mtp_slot_pad[:num_tokens_mtp_pad]
                mtp_slot_pad.fill_(-1)
                mtp_slot_pad[:: self.pcp_world_size].copy_(mtp_slot_ori)

            return PCPAsyncSpecDecodeRebuildResult(
                rebuilt=True,
                positions_ready_on_device=True,
            )

        base_num_computed_tokens_np = num_computed_tokens_cpu[:num_reqs].copy()
        assert valid_sampled_token_count_event is not None
        assert valid_sampled_token_count_cpu is not None
        valid_sampled_token_count_event.synchronize()
        correct_optimistic_seq_lens_cpu(
            base_num_computed_tokens_np,
            prev_positions_np,
            prev_num_draft_tokens_np,
            valid_sampled_token_count_cpu.numpy(),
            num_reqs,
        )

        if self.pcp_world_size > 1:
            assert position_pcp is not None
            position_offsets = position_pcp
        else:
            position_offsets = query_pos_np
        np.add(
            base_num_computed_tokens_np[req_indices],
            position_offsets[:total_num_scheduled_tokens],
            out=positions_np,
        )

        token_indices = positions_np[:total_num_scheduled_tokens] + req_indices * input_batch.token_ids_cpu.shape[1]
        torch.index_select(
            input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=input_ids.cpu[:total_num_scheduled_tokens],
        )
        input_ids.copy_to_gpu(total_num_scheduled_tokens)
        prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        req_indices_full = self.async_rebuild_req_indices_full
        cu_num_tokens_full = self.async_rebuild_cu_num_tokens_full
        num_tokens_full = self.async_rebuild_num_tokens_full
        assert req_indices_full is not None
        assert cu_num_tokens_full is not None

        token_counts = np.diff(np.concatenate(([0], cu_num_tokens_full)))
        token_starts = np.repeat(cu_num_tokens_full - token_counts, token_counts)
        query_pos = arange_np[:num_tokens_full] - token_starts

        positions_full = np.empty(num_tokens_full, dtype=np.int64)
        np.add(
            base_num_computed_tokens_np[req_indices_full],
            query_pos,
            out=positions_full,
        )

        if self.pcp_world_size > 1:
            pre_pcp_query_start_loc = torch.zeros(
                num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            pre_pcp_query_start_loc[1 : num_reqs + 1] = torch.from_numpy(cu_num_tokens_full).to(
                dtype=torch.int32, device=self.device
            )

            input_batch.block_table.compute_slot_mapping(
                num_reqs,
                pre_pcp_query_start_loc,
                torch.from_numpy(positions_full).to(self.device),
            )

        self.generate_pcp_mtp_input(
            num_tokens_full,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            input_batch,
            arange_np,
            req_indices_full,
            positions_full,
            cu_num_tokens_full,
            draft_token_ids,
            scheduler_output,
            num_spec_tokens,
            precomputed_positions_np=positions_full,
            prev_positions=prev_positions_gpu,
        )

        return PCPAsyncSpecDecodeRebuildResult(
            rebuilt=True,
            positions_ready_on_device=False,
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
        precomputed_positions_np=None,
        prev_positions: torch.Tensor | None = None,
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
        if precomputed_positions_np is None:
            np.add(
                input_batch.num_computed_tokens_cpu[req_indices_pcp_full],
                arange_pcp_full,
                out=positions_pcp_full_np,
            )
        else:
            np.copyto(
                positions_pcp_full_np,
                precomputed_positions_np[:total_num_scheduled_tokens_pcp_full],
            )
        token_indices_pcp_full = positions_pcp_full_np + req_indices_pcp_full * input_batch.token_ids_cpu.shape[1]
        torch.index_select(
            input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices_pcp_full),
            out=self.input_ids_pcp_full.cpu[:total_num_scheduled_tokens_pcp_full],
        )
        self.input_ids_pcp_full.copy_to_gpu(total_num_scheduled_tokens_pcp_full)
        self.query_start_loc_pcp_full.copy_to_gpu()
        if self.use_async_scheduling:
            self._update_input_ids_pcp_full_ids(
                input_batch,
                draft_token_ids,
                scheduler_output,
                total_num_scheduled_tokens,
                cu_num_tokens_pcp_full,
                num_spec_tokens,
                prev_positions,
            )
        self.cu_num_tokens_pcp_full = cu_num_tokens_pcp_full

        if self.use_async_scheduling and precomputed_positions_np is None:
            # Save full pre-CP layout so async scheduling can rebuild
            # speculative inputs with corrected num_computed_tokens.
            self.async_rebuild_req_indices_full = req_indices.copy()
            self.async_rebuild_cu_num_tokens_full = cu_num_tokens.copy()
            self.async_rebuild_num_tokens_full = total_num_scheduled_tokens

        # For mtpx, pre-allocate mtp slot_mapping here
        needs_dcp_prefill_slots = self.pcp_world_size == 1 and self.dcp_world_size > 1 and with_prefill
        if self.decode_threshold > 2 and (not with_prefill or needs_dcp_prefill_slots):
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
            input_batch.block_table.compute_slot_mapping_draft(req_indices_mtp, positions_mtp)
            mtp_slot_ori = input_batch.block_table.block_tables[0].slot_mapping.cpu[:num_tokens_mtp]
            unpad_mask = np.repeat(False, num_tokens_mtp_pad)
            unpad_mask[:: self.pcp_world_size] = True
            self.mtp_slot_pad = torch.full([num_tokens_mtp_pad], -1, dtype=torch.int32, pin_memory=True)
            self.mtp_slot_pad[unpad_mask] = mtp_slot_ori
            self.mtp_slot_pad = self.mtp_slot_pad.to(self.device, non_blocking=True)

    def _update_input_ids_pcp_full_ids(
        self,
        input_batch,
        draft_token_ids,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
        num_spec_tokens: int,
        prev_positions: torch.Tensor | None = None,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if input_batch.prev_sampled_token_ids is None or input_batch.prev_req_id_to_index is None:
            return

        if prev_positions is not None:
            num_reqs = self.num_reqs
            query_start_loc = self.query_start_loc_pcp_full.gpu[: num_reqs + 1]
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            is_decode_req = self.pcp_req_offsets[:num_reqs] < self.num_decode_reqs
            draft_lens = torch.where(
                is_decode_req,
                torch.clamp(query_lens - 1, min=0),
                torch.zeros_like(query_lens),
            )

            sample_indices = (query_start_loc[1:] - 1 - draft_lens).to(torch.int64)
            prev_positions = prev_positions[:num_reqs].to(torch.int64)
            common_mask = prev_positions >= 0
            safe_prev_positions = prev_positions.clamp(min=0)

            sampled_src = input_batch.prev_sampled_token_ids[safe_prev_positions, 0]
            sampled_src = sampled_src.to(dtype=self.input_ids_pcp_full.gpu.dtype)
            sampled_src = torch.where(
                common_mask,
                sampled_src,
                self.input_ids_pcp_full.gpu[sample_indices],
            )
            self.input_ids_pcp_full.gpu.scatter_(
                dim=0,
                index=sample_indices,
                src=sampled_src,
            )

            if draft_token_ids is None or not num_spec_tokens:
                return

            assert isinstance(draft_token_ids, torch.Tensor)
            if num_spec_tokens > self.pcp_spec_token_offsets.numel():
                spec_offsets = torch.arange(
                    num_spec_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                spec_offsets = self.pcp_spec_token_offsets[:num_spec_tokens]
            spec_offsets = spec_offsets.unsqueeze(0)
            draft_lens = torch.clamp(
                draft_lens.to(torch.int64),
                max=num_spec_tokens,
            )
            spec_mask = common_mask.unsqueeze(1) & (spec_offsets < draft_lens.unsqueeze(1))
            sample_indices_2d = sample_indices.unsqueeze(1)
            spec_dst = sample_indices_2d + 1 + spec_offsets
            safe_dst = torch.where(
                spec_mask,
                spec_dst,
                sample_indices_2d.expand(-1, num_spec_tokens),
            )
            spec_src_indices = safe_prev_positions.unsqueeze(1) * num_spec_tokens + spec_offsets

            draft_token_ids = draft_token_ids.to(dtype=torch.int32)
            spec_src = draft_token_ids.flatten()[spec_src_indices]
            spec_src = torch.where(
                spec_mask,
                spec_src,
                self.input_ids_pcp_full.gpu[safe_dst],
            )
            self.input_ids_pcp_full.gpu.scatter_(
                dim=0,
                index=safe_dst.reshape(-1),
                src=spec_src.reshape(-1),
            )
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
        num_common_tokens = len(sample_flattened_indices)

        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(sample_flattened_indices, dtype=torch.int64, device=self.device)
        prev_common_req_indices_tensor = torch.tensor(prev_common_req_indices, dtype=torch.int64, device=self.device)
        self.input_ids_pcp_full.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(spec_flattened_indices, dtype=torch.int64, device=self.device)
        prev_draft_token_indices_tensor = torch.tensor(prev_draft_token_indices, dtype=torch.int64, device=self.device)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = draft_token_ids.to(dtype=torch.int32)

        self.input_ids_pcp_full.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
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
        rank_offsets = (
            torch.arange(
                total_world_size,
                dtype=seq_lens.dtype,
                device=seq_lens.device,
            )
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
        dcp_local_seq_lens = (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
        return dcp_local_seq_lens

    @staticmethod
    def _is_mla_kv_cache_spec(kv_cache_spec: Any) -> bool:
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

        return isinstance(kv_cache_spec, AscendMLAAttentionSpec)

    @staticmethod
    def _is_sfa_dcp_metadata_builder(attn_metadata_builder: Any | None) -> bool:
        if attn_metadata_builder is None:
            return False
        from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder

        return isinstance(attn_metadata_builder, AscendSFADCPMetadataBuilder)

    def update_spec_decode_drafting_cp_metadata(
        self,
        attn_metadata: Any,
        kv_cache_spec: Any,
        seq_lens: torch.Tensor,
        draft_index: int,
        seq_lens_cpu: torch.Tensor | None = None,
        attn_metadata_builder: Any | None = None,
    ) -> None:
        """Update per-draft-step CP seq-len metadata after metadata build."""
        is_mla = self._is_mla_kv_cache_spec(kv_cache_spec)
        is_sfa_dcp = self._is_sfa_dcp_metadata_builder(attn_metadata_builder)
        seq_lens_for_cp = seq_lens
        if not is_mla and seq_lens_cpu is not None:
            seq_lens_for_cp = seq_lens_cpu

        num_computed_tokens_of_pcp_dcp = self._get_cp_local_seq_lens(
            seq_lens_for_cp + draft_index + 1,
            self.pcp_world_size,
            self.dcp_world_size,
            self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
        )
        cp_seq_len = num_computed_tokens_of_pcp_dcp[:, self.pcp_world_rank, self.dcp_world_rank]

        if is_sfa_dcp:
            dcp_context = attn_metadata.dcp_context
            assert dcp_context is not None
            dcp_seq_lens = dcp_context.seq_lens
            sfa_cp_seq_len = cp_seq_len.to(
                device=dcp_seq_lens.device,
                dtype=dcp_seq_lens.dtype,
                non_blocking=True,
            )
            dcp_seq_lens[: sfa_cp_seq_len.shape[0]].copy_(sfa_cp_seq_len, non_blocking=True)
            dcp_seq_lens[sfa_cp_seq_len.shape[0] :].fill_(0)
        elif is_mla:
            attn_metadata.decode.cp_seq_len = cp_seq_len
        else:
            attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp.numpy()

    def generate_pcp_metadata(
        self,
        total_num_scheduled_tokens: int,
        query_lens: torch.Tensor,
        input_batch: "NPUInputBatch",
        num_scheduled_tokens: np.ndarray | None,
        block_table_tensor: torch.Tensor,
        num_reqs_padded: int,
        num_reqs: int,
        fixed_decode_seq_lens_cpu: np.ndarray | None = None,
    ):
        from vllm_ascend.attention.utils import AscendPrefillContextParallelMetadata

        if self.pcp_world_size > 1 and self.pcp_use_hybrid_attn:
            assert self.num_scheduled_tokens_padded is not None
            total_num_scheduled_tokens = self.num_scheduled_tokens_padded.sum()
        num_actual_tokens_pcp_padded = total_num_scheduled_tokens * self.pcp_world_size
        self.num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded
        long_seq_metadata = None
        ori_query_lens_cpu = self.query_lens_pcp_full.cpu[:num_reqs_padded]
        if self.pcp_world_size * self.dcp_world_size > 1:
            assert num_scheduled_tokens is not None
            if fixed_decode_seq_lens_cpu is not None:
                decode_context_lens = fixed_decode_seq_lens_cpu[: self.num_decode_reqs]
            else:
                decode_context_lens = (
                    input_batch.num_computed_tokens_cpu[: self.num_decode_reqs]
                    + num_scheduled_tokens[: self.num_decode_reqs]
                )
            prefill_context_lens = input_batch.num_computed_tokens_cpu[self.num_decode_reqs : self.num_reqs]
            context_lens = np.concatenate([decode_context_lens, prefill_context_lens])

            num_computed_tokens_of_pcp_dcp = self._get_cp_local_seq_lens(
                torch.tensor(context_lens),
                self.pcp_world_size,
                self.dcp_world_size,
                self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[PCP][DFX] num_computed_tokens_of_pcp_dcp=%s",
                    num_computed_tokens_of_pcp_dcp.tolist(),
                )

            pcp_unpad_mask = self.pcp_unpad_mask_cpu[: self.pcp_padded_tokens_length]
            long_seq_metadata = AscendPrefillContextParallelMetadata(
                pcp_use_hybrid_attn=self.pcp_use_hybrid_attn,
                num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp.numpy(),
                pcp_unpad_mask=torch.from_numpy(pcp_unpad_mask),
                pcp_padded_tokens_fla=self.pcp_padded_tokens_fla,
                query_lens_pcp_full_cpu=ori_query_lens_cpu,
                max_query_len_pcp_full=ori_query_lens_cpu.max().item(),
            )
            if self.pcp_world_size > 1:
                q_head_idx, q_tail_idx = [], []
                kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
                kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
                kv_tail_proj_idx: list[int] = []
                kv_with_q_head_attn_idx_in_tail, kv_with_q_tail_attn_idx_in_tail = [], []
                split_with_q_head_nomask_idx_reqs = []
                split_kv_with_q_tail_nomask_idx_reqs = []
                chunk_seqlens = []
                kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
                head_actual_seq_lengths_kv, tail_actual_seq_lengths_kv = [], []
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
                    tail_proj_offset = len(kv_tail_proj_idx)
                    tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
                    kv_tail_proj_idx.extend(list(range(kv_req_offset, kv_req_offset + tail_proj_len)))
                    kv_with_q_head_attn_idx_in_tail.extend(
                        list(range(tail_proj_offset, tail_proj_offset + chunk_len * (q_head_chunk_id + 1)))
                    )
                    kv_with_q_tail_attn_idx_in_tail.extend(
                        list(range(tail_proj_offset, tail_proj_offset + tail_proj_len))
                    )
                    head_actual_seq_lengths_kv.append(len(kv_with_q_head_attn_idx_in_tail))
                    tail_actual_seq_lengths_kv.append(len(kv_with_q_tail_attn_idx_in_tail))
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
                    "kv_tail_proj_idx_tensor": kv_tail_proj_idx,
                    "kv_with_q_head_attn_idx_in_tail_tensor": kv_with_q_head_attn_idx_in_tail,
                    "kv_with_q_tail_attn_idx_in_tail_tensor": kv_with_q_tail_attn_idx_in_tail,
                }
                for key, value in self.kv_idx_names.items():
                    tensor_npu = self._list_to_tensor(value, self.device)
                    self.kv_idx_names[key] = tensor_npu

                attn_chunk_seqlens = torch.tensor(chunk_seqlens, dtype=torch.int32)
                attn_mask_seqlens = torch.cumsum(torch.tensor(chunk_seqlens, dtype=torch.int32), dim=0).tolist()
                head_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_head_nomask_seqlens, dtype=torch.int32), dim=0
                ).tolist()
                tail_attn_nomask_seqlens = torch.cumsum(
                    torch.tensor(kv_with_q_tail_nomask_seqlens, dtype=torch.int32), dim=0
                ).tolist()

                self.extra_long_seq_kwargs = {
                    "attn_mask_seqlens": attn_mask_seqlens,
                    "head_attn_nomask_seqlens": head_attn_nomask_seqlens,
                    "tail_attn_nomask_seqlens": tail_attn_nomask_seqlens,
                    "head_actual_seq_lengths_kv": head_actual_seq_lengths_kv,
                    "tail_actual_seq_lengths_kv": tail_actual_seq_lengths_kv,
                }
                long_seq_metadata.pcp_allgather_restore_idx = self.pcp_allgather_restore_idx.gpu[
                    :num_actual_tokens_pcp_padded
                ]
                if self.pcp_use_hybrid_attn:
                    long_seq_metadata.pcp_exit_fa_scatter_idx = self.pcp_exit_fa_scatter_idx.gpu[
                        : num_scheduled_tokens.sum() - self.num_decode_tokens
                    ]
                    long_seq_metadata.pcp_fa_query_idx = self.pcp_fa_query_idx[
                        : num_actual_tokens_pcp_padded // self.pcp_world_size - self.num_decode_tokens
                    ]
                    actual_qkv_len = int(pcp_unpad_mask.sum()) + self.num_decode_tokens * (self.pcp_world_size - 1)
                    long_seq_metadata.pcp_enter_fa_restore_idx = self.pcp_enter_fa_restore_idx[:actual_qkv_len]
                    if actual_qkv_len < num_actual_tokens_pcp_padded:
                        long_seq_metadata.pcp_fa_padding_restore_idx = self.pcp_fa_padding_restore_idx[
                            :num_actual_tokens_pcp_padded
                        ]
                    else:
                        long_seq_metadata.pcp_fa_padding_restore_idx = None
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[PCP][DFX] long_seq_metadata reorder idx: "
                            "pcp_allgather_restore_idx=%s, "
                            "pcp_exit_fa_scatter_idx=%s, "
                            "pcp_enter_fa_restore_idx=%s, "
                            "pcp_fa_padding_restore_idx=%s",
                            long_seq_metadata.pcp_allgather_restore_idx.detach().cpu().tolist(),
                            long_seq_metadata.pcp_exit_fa_scatter_idx.detach().cpu().tolist(),
                            long_seq_metadata.pcp_enter_fa_restore_idx.detach().cpu().tolist(),
                            long_seq_metadata.pcp_fa_padding_restore_idx.detach().cpu().tolist()
                            if long_seq_metadata.pcp_fa_padding_restore_idx is not None
                            else None,
                        )
                    long_seq_metadata.max_num_tokens_across_pcp = self.max_num_tokens_across_pcp
                    long_seq_metadata.total_num_scheduled_tokens = self.total_num_scheduled_tokens
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
                long_seq_metadata.kv_tail_proj_idx_tensor = self.kv_idx_names["kv_tail_proj_idx_tensor"]
                long_seq_metadata.kv_with_q_head_attn_idx_in_tail_tensor = self.kv_idx_names[
                    "kv_with_q_head_attn_idx_in_tail_tensor"
                ]
                long_seq_metadata.kv_with_q_tail_attn_idx_in_tail_tensor = self.kv_idx_names[
                    "kv_with_q_tail_attn_idx_in_tail_tensor"
                ]
                long_seq_metadata.attn_mask_seqlens = self.extra_long_seq_kwargs["attn_mask_seqlens"]
                long_seq_metadata.head_attn_nomask_seqlens = self.extra_long_seq_kwargs["head_attn_nomask_seqlens"]
                long_seq_metadata.tail_attn_nomask_seqlens = self.extra_long_seq_kwargs["tail_attn_nomask_seqlens"]
                long_seq_metadata.head_actual_seq_lengths_kv = self.extra_long_seq_kwargs["head_actual_seq_lengths_kv"]
                long_seq_metadata.tail_actual_seq_lengths_kv = self.extra_long_seq_kwargs["tail_actual_seq_lengths_kv"]
                long_seq_metadata.attn_chunk_seqlens = attn_chunk_seqlens

            # Generate MTP attention masks for decode requests when cp_size > 1
            # with speculative decoding.
            if (
                self.dcp_world_size * self.pcp_world_size > 1
                and self.speculative_config
                and num_scheduled_tokens is not None
            ):
                # Generate the mask contents for the real decode requests.
                if self.num_decode_reqs > 0:
                    decode_num_scheduled_tokens = num_scheduled_tokens[: self.num_decode_reqs]
                    if fixed_decode_seq_lens_cpu is not None:
                        decode_num_computed_tokens = (
                            fixed_decode_seq_lens_cpu[: self.num_decode_reqs] - decode_num_scheduled_tokens
                        ).tolist()
                    else:
                        decode_num_computed_tokens = input_batch.num_computed_tokens_cpu[
                            : self.num_decode_reqs
                        ].tolist()

                    dcp_mtp_attn_mask = self.generate_mtp_attention_mask_for_decode(
                        decode_num_computed_tokens, decode_num_scheduled_tokens
                    )
                    if dcp_mtp_attn_mask is not None:
                        self.dcp_mtp_attn_mask.np[: self.num_decode_reqs] = dcp_mtp_attn_mask
                        self.dcp_mtp_attn_mask.copy_to_gpu(self.num_decode_reqs)
                # Always expose the (stable, pre-allocated) MTP mask buffer
                # for cp>1 + speculative decode, even when num_decode_reqs == 0.
                mask_n = self.num_decode_reqs if self.num_decode_reqs > 0 else num_reqs
                long_seq_metadata.dcp_mtp_attn_mask = self.dcp_mtp_attn_mask.gpu[:mask_n]
            else:
                long_seq_metadata.dcp_mtp_attn_mask = None

        self.long_seq_metadata = long_seq_metadata
        return long_seq_metadata, block_table_tensor

    def _list_to_tensor(self, lst, device, dtype=torch.int32):
        tensor_npu = torch.zeros(len(lst), dtype=dtype, device=device)
        tensor_npu.copy_(torch.tensor(lst, dtype=dtype), non_blocking=True)
        return tensor_npu

    def remap_mrope_positions_for_pcp(
        self,
        positions_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        input_batch: "NPUInputBatch",
        requests: dict[str, Any],
        mrope_positions: CpuGpuBuffer,
    ):
        """Remap mrope_positions after PCP split.

        _calc_mrope_positions fills mrope_positions using the original
        (pre-PCP-split) sequential token ordering from scheduler_output.
        After PCP splits tokens across ranks, each rank only processes a
        subset of tokens (head+tail chunks), so we must remap mrope_positions
        to match the PCP-local token ordering.

        positions_np already contains the correct absolute position for each
        token on this PCP rank (computed by update_tokens_for_pcp). We use
        these positions to gather the correct mrope_positions from
        req.mrope_positions (for prompt tokens) or compute them on-the-fly
        (for completion/decode tokens).
        """
        mrope_pos_ptr = 0
        for index, req_id in enumerate(input_batch.req_ids):
            req = requests[req_id]
            num_sched = int(num_scheduled_tokens[index])
            local_positions = positions_np[mrope_pos_ptr : mrope_pos_ptr + num_sched]

            if req.mrope_positions is not None and req.mrope_positions.shape[1] > 0:
                num_prompt_tokens = length_from_prompt_token_ids_or_embeds(req.prompt_token_ids, req.prompt_embeds)
                max_mrope_idx = req.mrope_positions.shape[1]

                # Build the mrope_positions for this request's PCP-local
                # tokens. For each token, gather from req.mrope_positions
                # using its absolute position from positions_np.
                mrope_dst = np.empty((3, num_sched), dtype=np.int64)

                # Prompt tokens: positions within prompt range,
                # gather from pre-computed req.mrope_positions.
                prompt_mask = local_positions < min(num_prompt_tokens, max_mrope_idx)
                if prompt_mask.any():
                    prompt_indices = local_positions[prompt_mask].astype(np.int64)
                    prompt_indices = np.clip(prompt_indices, 0, max_mrope_idx - 1)
                    mrope_dst[:, prompt_mask] = req.mrope_positions[:, torch.from_numpy(prompt_indices)].numpy()

                # Completion/decode tokens: all 3 dims use the same
                # position.
                completion_mask = local_positions >= num_prompt_tokens
                if completion_mask.any():
                    # For completion tokens, use mrope_position_delta to
                    # compute the correct position, same as
                    # get_next_input_positions_tensor.
                    if req.mrope_position_delta is not None:
                        comp_positions = local_positions[completion_mask] + req.mrope_position_delta
                    else:
                        comp_positions = local_positions[completion_mask]
                    mrope_dst[:, completion_mask] = comp_positions[np.newaxis, :]

                # Padding tokens beyond req.mrope_positions shape:
                # use the last valid mrope position.
                padding_mask = (~prompt_mask) & (~completion_mask)
                if padding_mask.any():
                    last_idx = max_mrope_idx - 1
                    mrope_dst[:, padding_mask] = req.mrope_positions[:, last_idx : last_idx + 1].numpy()

                mrope_positions.cpu[:, mrope_pos_ptr : mrope_pos_ptr + num_sched] = torch.from_numpy(mrope_dst)
            else:
                # No mrope_positions available:
                # all 3 dims equal the 1D position.
                mrope_positions.cpu[:, mrope_pos_ptr : mrope_pos_ptr + num_sched] = torch.from_numpy(
                    local_positions[np.newaxis, :].astype(np.int64)
                )

            mrope_pos_ptr += num_sched

    def generate_mtp_attention_mask_for_decode(
        self,
        decode_num_computed_tokens: list[int],
        decode_num_scheduled_tokens: np.ndarray,
    ) -> list[torch.Tensor | None]:
        """
        Generate MTP attention masks for decode requests in PCP mode.

        This function handles the case where decode requests with MTP (speculative decoding)
        need attention masks computed based on the local sequence after load balancing.

        New MTP token allocation logic (using position % cp_size):
        - History tokens are already split via DualChunkSwap
        - MTP tokens are allocated based on (history_len + mtp_idx) % cp_size
        - Each rank only computes mask for tokens assigned to itself

        Example:
            - pcp=1, dcp=2 (cp_size=2)
            - history_len=5: [a,b,c,d,e] split via DualChunkSwap
              - cp0: [a,b,c] (positions 0,1,2) -> 3 tokens
              - cp1: [d,e] (positions 3,4) -> 2 tokens
            - num_scheduled_tokens=4: [f,g,h,i] (positions 5,6,7,8)
            - MTP allocation by position % cp_size:
              - f: pos 5 % 2 = 1 -> rank1
              - g: pos 6 % 2 = 0 -> rank0
              - h: pos 7 % 2 = 1 -> rank1
              - i: pos 8 % 2 = 0 -> rank0
            - Final:
              - rank0: [a,b,c,g,i] positions [0,1,2,6,8] -> mask shape 4x5
              - rank1: [d,e,f,h] positions [3,4,5,7] -> mask shape 4x4

        Args:
            decode_num_computed_tokens: List of global history lengths for decode requests
            decode_num_scheduled_tokens: Array of scheduled token counts for decode requests
        """
        cp_rank = self.pcp_world_rank * self.dcp_world_size + self.dcp_world_rank
        cp_size = self.pcp_world_size * self.dcp_world_size
        assert cp_size > 1, "cp_size must be greater than 1"

        interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size

        q_lens = torch.tensor(decode_num_scheduled_tokens[: self.num_decode_reqs], dtype=torch.int32)
        global_histories = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
        total_lens = global_histories + q_lens
        context_lens = total_lens - q_lens

        # Interleave-aware per-rank KV length:
        # base = L // I // W * I, remainder = L - base * W,
        # local = base + clip(remainder - rank * I, 0, I)
        base_k = total_lens // interleave_size // cp_size * interleave_size
        remainder_k = total_lens - base_k * cp_size
        k_lens = base_k + torch.clamp(remainder_k - cp_rank * interleave_size, 0, interleave_size)
        valid = k_lens > 0

        if not valid.any():
            return self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]

        k_lens = torch.where(valid, k_lens, torch.zeros_like(k_lens))

        mtp_attn_mask = self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]
        mtp_attn_mask.zero_()

        num_valid = valid.sum().item()
        if num_valid == 0:
            return mtp_attn_mask

        max_q = int(q_lens[valid].max().item())
        max_k = int(k_lens[valid].max().item())

        # Generate indices up to max dimensions
        q_indices = torch.arange(max_q, dtype=torch.int32)
        k_indices = torch.arange(max_k, dtype=torch.int32)

        valid_q = valid[:, None] & (q_indices[None, :] < q_lens[:, None])
        valid_k = valid[:, None] & (k_indices[None, :] < k_lens[:, None])

        # Interleave-aware k_upper: for query token at global position P,
        # count local KV tokens with global pos <= P (inclusive causal), then
        # convert to an inclusive local index. Using P (exclusive) here would
        # drop the query's own KV when it lives on this rank and can also make
        # k_upper < 0, which disables masking for that row.
        positions = context_lens[:, None] + q_indices[None, :]  # [num_decode_reqs, max_q]
        inclusive_positions = positions + 1
        base_q = inclusive_positions // interleave_size // cp_size * interleave_size
        remainder_q = inclusive_positions - base_q * cp_size
        local_q = base_q + torch.clamp(remainder_q - cp_rank * interleave_size, 0, interleave_size)
        k_upper = local_q - 1  # inclusive upper KV index

        k_upper_expanded = k_upper[:, :, None]  # [num_decode_reqs, max_q, 1]
        k_idx_expanded = k_indices[None, None, :]  # [1, 1, max_k]
        full_mask = (k_idx_expanded > k_upper_expanded) & (k_upper_expanded >= 0)

        valid_mask_3d = valid_q[:, :, None] & valid_k[:, None, :]
        full_mask = full_mask & valid_mask_3d

        mtp_attn_mask[: self.num_decode_reqs, :max_q, :max_k] = full_mask

        return mtp_attn_mask
