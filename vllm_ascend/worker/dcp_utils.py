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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.attention.context_parallel.common_cp import (
    get_dcp_local_seq_lens,
)
from vllm_ascend.spec_decode.utils import correct_optimistic_seq_lens_cpu
from vllm_ascend.utils import is_pd_decode_recompute_scheduler_enabled
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass(frozen=True)
class DCPSpecDecodeMTPInputs:
    """Device-side DCP state needed by proposer MTP draft steps."""

    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    slot_indices: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass(frozen=True)
class DCPSpecDecodeFirstPassInputs:
    """DCP metadata attached to the first speculative draft pass."""

    num_tokens: int
    input_ids: torch.Tensor
    target_positions: torch.Tensor
    target_hidden_states: torch.Tensor
    token_indices_to_sample: torch.Tensor
    long_seq_args: tuple[torch.Tensor | None, torch.Tensor | None] | None


@dataclass(frozen=True)
class DCPAsyncSpecDecodeRebuildResult:
    """Status returned after rebuilding async speculative inputs."""

    rebuilt: bool
    positions_ready_on_device: bool


class DCPManager:
    """Manage Decode Context Parallel metadata and reusable buffers."""

    num_reqs: int = 0
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    decode_req_mask: np.ndarray | None = None

    def __init__(
        self,
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
        del max_buffer_num_tokens
        self.dcp_world_size = dcp_world_size
        self.dcp_world_rank = dcp_rank
        self.vllm_config = vllm_config
        self.device = device
        self.use_async_scheduling = use_async_scheduling
        self.use_sparse = use_sparse
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)
        self.pd_decode_recompute_scheduler_enabled = is_pd_decode_recompute_scheduler_enabled(vllm_config)
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = max_num_reqs
        self.req_offsets = torch.arange(max_num_reqs, dtype=torch.int64, device=device)
        self.query_lens_full = CpuGpuBuffer(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.query_start_loc_full = CpuGpuBuffer(
            max_num_reqs + 1,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.mtp_slot_mapping: torch.Tensor | None = None
        self.dcp_mtp_attn_mask = CpuGpuBuffer(
            (
                max_num_reqs,
                self.decode_threshold,
                vllm_config.model_config.max_model_len,
            ),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.async_rebuild_req_indices: np.ndarray | None = None
        self.async_rebuild_cu_num_tokens: np.ndarray | None = None
        self.async_rebuild_num_tokens = 0
        self.long_seq_metadata: Any | None = None

    def classify_decode_request_mask(
        self,
        num_scheduled_tokens: np.ndarray | torch.Tensor,
        num_computed_tokens: np.ndarray | torch.Tensor,
        num_prompt_tokens: np.ndarray | torch.Tensor,
        decode_threshold: int,
    ) -> np.ndarray:
        """Match vLLM's decode/prefill batch classification."""
        has_context = num_computed_tokens > 0
        is_below_threshold = num_scheduled_tokens <= decode_threshold
        done_prefilling = num_computed_tokens >= num_prompt_tokens
        if self.pd_decode_recompute_scheduler_enabled:
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
        self.query_lens_full.cpu[:num_reqs] = torch.from_numpy(scheduled)
        self.query_lens_full.cpu[num_reqs:].fill_(0)
        self.query_lens_full.copy_to_gpu()

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
    ) -> DCPSpecDecodeFirstPassInputs:
        del req_scheduled_tokens, req_ids, logits_indices, num_prefill_reqs, uses_mrope
        assert long_seq_metadata is not None
        common_attn_metadata.context_parallel_metadata = long_seq_metadata
        original_sample_indices = token_indices_to_sample.clone()
        decode_query_lens = self.query_lens_full.cpu[:num_decode_reqs]
        return DCPSpecDecodeFirstPassInputs(
            num_tokens=num_tokens,
            input_ids=input_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            long_seq_args=(decode_query_lens, original_sample_indices),
        )

    def _get_spec_decode_mtp_slot_inputs(
        self,
        original_sample_indices: torch.Tensor,
        num_reqs: int,
        num_speculative_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.mtp_slot_mapping is not None
        query_start_loc = self.query_start_loc_full.gpu[: num_reqs + 1]
        req_starts = query_start_loc[:num_reqs].to(torch.int64)
        cu_num_tokens = query_start_loc[1 : num_reqs + 1].to(torch.int64)
        query_lens = cu_num_tokens - req_starts
        num_reject_tokens = cu_num_tokens - original_sample_indices.to(torch.int64) - 1
        num_accept_tokens = query_lens - num_reject_tokens
        slot_indices = req_starts + self.req_offsets[:num_reqs] * (num_speculative_tokens - 1) + num_accept_tokens - 1
        return slot_indices, self.mtp_slot_mapping

    def prepare_spec_decode_mtp_drafting_inputs(
        self,
        common_attn_metadata: Any,
        attn_metadata: Any,
        ori_token_indices_to_sample: torch.Tensor | None,
        batch_size: int,
        num_decode_reqs: int,
        is_prefill_batch: bool,
        num_speculative_tokens: int,
    ) -> DCPSpecDecodeMTPInputs | None:
        is_decode_only_batch = num_decode_reqs > 0 and not is_prefill_batch
        if num_speculative_tokens <= 1 or not (is_decode_only_batch or is_prefill_batch):
            return None
        assert ori_token_indices_to_sample is not None
        num_reqs = batch_size if is_prefill_batch else num_decode_reqs
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
        return DCPSpecDecodeMTPInputs(
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
    ) -> DCPAsyncSpecDecodeRebuildResult:
        del draft_token_ids
        should_rebuild = (
            use_async_spec_decode
            and valid_sampled_token_count_gpu is not None
            and bool(prev_req_id_to_index)
            and self.num_decode_reqs > 0
        )
        if not should_rebuild:
            return DCPAsyncSpecDecodeRebuildResult(False, False)

        can_rebuild_on_device = (
            prev_positions_gpu is not None
            and not with_prefill
            and not enable_prompt_embeds
            and not has_req_prompt_embeds
            and not supports_mm_inputs
        )
        if can_rebuild_on_device:
            position_offsets = query_pos_gpu[:total_num_scheduled_tokens].to(torch.int64)
            positions_gpu = num_computed_tokens[req_indices_gpu].to(torch.int64) + position_offsets
            positions[:total_num_scheduled_tokens].copy_(positions_gpu)

            extra_tokens = self.decode_threshold - 2
            if extra_tokens > 0 and not with_prefill:
                query_start_loc = self.query_start_loc_full.gpu[: num_reqs + 1]
                query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
                mtp_lens = query_lens + extra_tokens
                num_tokens_mtp = self.async_rebuild_num_tokens + num_reqs * extra_tokens
                req_indices_mtp = torch.repeat_interleave(
                    self.req_offsets[:num_reqs],
                    mtp_lens,
                    output_size=num_tokens_mtp,
                )
                mtp_start_loc = torch.empty(num_reqs + 1, dtype=torch.int64, device=self.device)
                mtp_start_loc[0] = 0
                mtp_start_loc[1:] = torch.cumsum(mtp_lens, dim=0)
                mtp_offsets = torch.arange(num_tokens_mtp, dtype=torch.int64, device=self.device)
                positions_mtp = (
                    num_computed_tokens[req_indices_mtp].to(torch.int64) + mtp_offsets - mtp_start_loc[req_indices_mtp]
                )
                input_batch.block_table.compute_slot_mapping_draft(req_indices_mtp, positions_mtp)
                slot_mapping = input_batch.block_table.block_tables[0].slot_mapping.gpu[:num_tokens_mtp]
                self.mtp_slot_mapping = slot_mapping.clone()
            return DCPAsyncSpecDecodeRebuildResult(True, True)

        base_num_computed_tokens = num_computed_tokens_cpu[:num_reqs].copy()
        assert valid_sampled_token_count_event is not None
        assert valid_sampled_token_count_cpu is not None
        valid_sampled_token_count_event.synchronize()
        correct_optimistic_seq_lens_cpu(
            base_num_computed_tokens,
            prev_positions_np,
            prev_num_draft_tokens_np,
            valid_sampled_token_count_cpu.numpy(),
            num_reqs,
        )
        np.add(
            base_num_computed_tokens[req_indices],
            query_pos_np[:total_num_scheduled_tokens],
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

        full_req_indices = self.async_rebuild_req_indices
        full_cu_num_tokens = self.async_rebuild_cu_num_tokens
        assert full_req_indices is not None
        assert full_cu_num_tokens is not None
        token_counts = np.diff(np.concatenate(([0], full_cu_num_tokens)))
        token_starts = np.repeat(full_cu_num_tokens - token_counts, token_counts)
        query_positions = arange_np[: self.async_rebuild_num_tokens] - token_starts
        full_positions = np.empty(self.async_rebuild_num_tokens, dtype=np.int64)
        np.add(
            base_num_computed_tokens[full_req_indices],
            query_positions,
            out=full_positions,
        )
        self.generate_dcp_mtp_input(
            self.async_rebuild_num_tokens,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            input_batch,
            arange_np,
            full_req_indices,
            full_positions,
            full_cu_num_tokens,
            scheduler_output=scheduler_output,
            num_spec_tokens=num_spec_tokens,
            precomputed_positions_np=full_positions,
            prev_positions=prev_positions_gpu,
        )
        return DCPAsyncSpecDecodeRebuildResult(True, False)

    def generate_dcp_mtp_input(
        self,
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: dict[str, int],
        with_prefill: bool = True,
        input_batch: NPUInputBatch | None = None,
        arange_np: np.ndarray | None = None,
        req_indices: np.ndarray | None = None,
        positions_np: np.ndarray | None = None,
        cu_num_tokens: np.ndarray | None = None,
        draft_token_ids: torch.Tensor | None = None,
        scheduler_output: "SchedulerOutput | None" = None,
        num_spec_tokens: int | None = None,
        precomputed_positions_np: np.ndarray | None = None,
        prev_positions: torch.Tensor | None = None,
    ) -> None:
        del (
            total_num_scheduled_tokens,
            arange_np,
            draft_token_ids,
            scheduler_output,
            num_spec_tokens,
            precomputed_positions_np,
            prev_positions,
        )
        assert input_batch is not None
        assert req_indices is not None
        assert positions_np is not None
        assert cu_num_tokens is not None
        scheduled = np.fromiter(
            (num_scheduled_tokens[req_id] for req_id in input_batch.req_ids),
            dtype=np.int32,
            count=self.num_reqs,
        )
        cumulative = np.cumsum(scheduled)
        self.query_start_loc_full.np[0] = 0
        self.query_start_loc_full.np[1 : self.num_reqs + 1] = cumulative
        self.query_start_loc_full.np[self.num_reqs + 1 :].fill(-1)
        self.query_start_loc_full.copy_to_gpu()

        if self.use_async_scheduling:
            self.async_rebuild_req_indices = req_indices.copy()
            self.async_rebuild_cu_num_tokens = cu_num_tokens.copy()
            self.async_rebuild_num_tokens = int(cumulative[-1])

        if self.decode_threshold <= 2:
            return
        extra_tokens = self.decode_threshold - 2
        req_indices_split = np.array_split(req_indices, cu_num_tokens)[: self.num_reqs]
        positions_split = np.array_split(positions_np, cu_num_tokens)[: self.num_reqs]
        for req_idx in range(self.num_reqs):
            if req_indices_split[req_idx].size == 0:
                continue
            req_indices_split[req_idx] = np.append(
                req_indices_split[req_idx],
                np.repeat(req_indices_split[req_idx][-1], extra_tokens),
            )
            positions_split[req_idx] = np.append(
                positions_split[req_idx],
                np.arange(
                    positions_split[req_idx][-1] + 1,
                    positions_split[req_idx][-1] + extra_tokens + 1,
                ),
            )
        req_indices_mtp = np.concatenate(req_indices_split)
        positions_mtp = np.concatenate(positions_split)
        input_batch.block_table.compute_slot_mapping_draft(req_indices_mtp, positions_mtp)
        num_tokens_mtp = req_indices_mtp.shape[0]
        slot_mapping = input_batch.block_table.block_tables[0].slot_mapping.cpu[:num_tokens_mtp]
        self.mtp_slot_mapping = slot_mapping.pin_memory().to(self.device, non_blocking=True)

    def _get_dcp_local_seq_lens(
        self,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return each request's interleave-aware KV length on every DCP rank."""
        return get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_world_size,
            self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
        )

    @staticmethod
    def _is_mla_kv_cache_spec(kv_cache_spec: Any) -> bool:
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

        return isinstance(kv_cache_spec, AscendMLAAttentionSpec)

    @staticmethod
    def _is_sfa_dcp_metadata_builder(
        attn_metadata_builder: Any | None,
    ) -> bool:
        if attn_metadata_builder is None:
            return False
        from vllm_ascend.attention.context_parallel.sfa_cp import (
            AscendSFADCPMetadataBuilder,
        )

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
        is_mla = self._is_mla_kv_cache_spec(kv_cache_spec)
        is_sfa_dcp = self._is_sfa_dcp_metadata_builder(attn_metadata_builder)
        seq_lens_for_dcp = seq_lens
        if not is_mla and seq_lens_cpu is not None:
            seq_lens_for_dcp = seq_lens_cpu
        local_seq_lens = self._get_dcp_local_seq_lens(seq_lens_for_dcp + draft_index + 1)
        rank_seq_lens = local_seq_lens[:, self.dcp_world_rank]

        if is_sfa_dcp:
            dcp_context = attn_metadata.dcp_context
            assert dcp_context is not None
            target = dcp_context.seq_lens
            rank_seq_lens = rank_seq_lens.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=True,
            )
            target[: rank_seq_lens.shape[0]].copy_(rank_seq_lens, non_blocking=True)
            target[rank_seq_lens.shape[0] :].fill_(0)
        elif is_mla:
            attn_metadata.decode.cp_seq_len = rank_seq_lens
        elif attn_metadata.decode_meta is not None:
            attn_metadata.decode_meta.num_computed_tokens_of_dcp = local_seq_lens.numpy()

    def generate_dcp_metadata(
        self,
        total_num_scheduled_tokens: int,
        query_lens: torch.Tensor,
        input_batch: NPUInputBatch,
        num_scheduled_tokens: np.ndarray | None,
        block_table_tensor: torch.Tensor,
        num_reqs_padded: int,
        num_reqs: int,
        fixed_decode_seq_lens_cpu: np.ndarray | None = None,
    ) -> tuple[Any, torch.Tensor]:
        del total_num_scheduled_tokens, query_lens
        from vllm_ascend.attention.utils import AscendDCPMetadata

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
        local_seq_lens = self._get_dcp_local_seq_lens(torch.tensor(context_lens))
        query_lens_cpu = self.query_lens_full.cpu[:num_reqs_padded]
        metadata = AscendDCPMetadata(
            num_computed_tokens_of_dcp=local_seq_lens.numpy(),
            query_lens_cpu=query_lens_cpu,
            max_query_len=(int(query_lens_cpu[:num_reqs].max().item()) if num_reqs else 0),
        )

        if self.speculative_config:
            if self.num_decode_reqs > 0:
                decode_scheduled = num_scheduled_tokens[: self.num_decode_reqs]
                if fixed_decode_seq_lens_cpu is not None:
                    decode_computed = (fixed_decode_seq_lens_cpu[: self.num_decode_reqs] - decode_scheduled).tolist()
                else:
                    decode_computed = input_batch.num_computed_tokens_cpu[: self.num_decode_reqs].tolist()
                mask = self.generate_mtp_attention_mask_for_decode(decode_computed, decode_scheduled)
                self.dcp_mtp_attn_mask.np[: self.num_decode_reqs] = mask
                self.dcp_mtp_attn_mask.copy_to_gpu(self.num_decode_reqs)
            mask_count = self.num_decode_reqs if self.num_decode_reqs > 0 else num_reqs
            metadata.dcp_mtp_attn_mask = self.dcp_mtp_attn_mask.gpu[:mask_count]

        self.long_seq_metadata = metadata
        return metadata, block_table_tensor

    def generate_mtp_attention_mask_for_decode(
        self,
        decode_num_computed_tokens: list[int],
        decode_num_scheduled_tokens: np.ndarray,
    ) -> torch.Tensor:
        """Build interleave-aware causal masks for DCP speculative decode."""
        interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        q_lens = torch.tensor(
            decode_num_scheduled_tokens[: self.num_decode_reqs],
            dtype=torch.int32,
        )
        histories = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
        total_lens = histories + q_lens
        k_lens = get_dcp_local_seq_lens(
            total_lens,
            self.dcp_world_size,
            interleave_size,
        )[:, self.dcp_world_rank]
        valid = k_lens > 0
        output = self.dcp_mtp_attn_mask.cpu[: self.num_decode_reqs]
        output.zero_()
        if not valid.any():
            return output

        max_q = int(q_lens[valid].max().item())
        max_k = int(k_lens[valid].max().item())
        q_indices = torch.arange(max_q, dtype=torch.int32)
        k_indices = torch.arange(max_k, dtype=torch.int32)
        valid_q = valid[:, None] & (q_indices[None, :] < q_lens[:, None])
        valid_k = valid[:, None] & (k_indices[None, :] < k_lens[:, None])
        positions = histories[:, None] + q_indices[None, :]
        inclusive_positions = positions + 1
        local_q = get_dcp_local_seq_lens(
            inclusive_positions,
            self.dcp_world_size,
            interleave_size,
        )[..., self.dcp_world_rank]
        upper = local_q - 1
        full_mask = (
            (k_indices[None, None, :] > upper[:, :, None])
            & (upper[:, :, None] >= 0)
            & valid_q[:, :, None]
            & valid_k[:, None, :]
        )
        output[: self.num_decode_reqs, :max_q, :max_k] = full_mask
        return output
