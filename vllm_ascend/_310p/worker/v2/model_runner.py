# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

# Register 310P kernel implementations only when the V2 runner is imported.
from vllm_ascend._310p.worker.v2 import kernel_registry as kernel_registry
from vllm_ascend._310p.worker.v2.aclgraph import ModelAclGraphManager310
from vllm_ascend._310p.worker.v2.kv_block_zeroer import AscendKVBlockZeroer310V2
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend._310p.worker.v2.states import Ascend310PRequestState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class NPUModelRunner310V2(NPUModelRunner):
    """First-release Model Runner V2 implementation for Ascend 310P."""

    aclgraph_manager_cls = ModelAclGraphManager310
    request_state_cls = Ascend310PRequestState

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_first_release_config(vllm_config)
        super().__init__(vllm_config, device)
        self.sampler = Ascend310PGreedySampler()
        self.input_ids_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int32, device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64, device="cpu")
        self.next_prefill_tokens_cpu = torch.zeros(1, self.max_num_reqs, dtype=torch.int32, device="cpu")

    @staticmethod
    def _validate_first_release_config(vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        unsupported_parallel = {
            "pipeline_parallel_size": getattr(parallel_config, "pipeline_parallel_size", 1),
            "data_parallel_size": getattr(parallel_config, "data_parallel_size", 1),
            "decode_context_parallel_size": getattr(parallel_config, "decode_context_parallel_size", 1),
            "prefill_context_parallel_size": getattr(parallel_config, "prefill_context_parallel_size", 1),
        }
        enabled = [name for name, size in unsupported_parallel.items() if size != 1]
        if enabled:
            raise NotImplementedError(
                "310P Model Runner V2 first release only supports tensor parallelism; "
                f"unsupported parallel settings: {', '.join(enabled)}."
            )

        if vllm_config.speculative_config is not None:
            raise NotImplementedError("Speculative decoding is deferred to the second 310P Model Runner V2 release.")
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("Prefix caching is deferred to the second 310P Model Runner V2 release.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("LoRA is outside the 310P Model Runner V2 V1-alignment scope.")
        if getattr(parallel_config, "enable_expert_parallel", False):
            raise NotImplementedError("Expert parallelism is outside the 310P Model Runner V2 first-release scope.")
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("KV transfer is outside the 310P Model Runner V2 first-release scope.")
        if getattr(vllm_config.scheduler_config, "async_scheduling", False):
            raise NotImplementedError("Async scheduling is outside the 310P Model Runner V2 first-release scope.")
        if getattr(vllm_config.model_config, "enable_sleep_mode", False):
            raise NotImplementedError("Sleep mode is outside the 310P Model Runner V2 first-release scope.")

    def _prepare_prefill_inputs(
        self,
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
        *,
        idx_mapping_np,
        query_start_loc_np,
    ) -> None:
        del idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens
        self.input_ids_cpu[: input_ids.shape[0]].zero_()
        self.next_prefill_tokens_cpu.zero_()
        for batch_idx, req_idx in enumerate(idx_mapping_np):
            num_computed = int(self.req_states.num_computed_tokens_np[req_idx])
            req_prefill_len = int(self.req_states.prefill_len.np[req_idx])
            if num_computed >= req_prefill_len:
                continue
            start = int(query_start_loc_np[batch_idx])
            end = int(query_start_loc_np[batch_idx + 1])
            self.input_ids_cpu[start:end] = self.req_states.all_token_ids.cpu[
                req_idx, num_computed : num_computed + end - start
            ]
            next_position = num_computed + end - start
            if next_position < req_prefill_len:
                self.next_prefill_tokens_cpu[0, req_idx] = self.req_states.all_token_ids.cpu[req_idx, next_position]

        input_ids.copy_(self.input_ids_cpu[: input_ids.shape[0]], non_blocking=True)
        next_prefill_tokens.copy_(self.next_prefill_tokens_cpu, non_blocking=True)

    def _prepare_pos_seq_lens(
        self,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        positions,
        seq_lens,
        *,
        idx_mapping_np,
        query_start_loc_np,
        num_scheduled_tokens,
    ) -> None:
        del idx_mapping, query_start_loc, num_computed_tokens
        self.input_buffers.seq_lens_cpu.zero_()
        self.positions_cpu[: positions.shape[0]].zero_()
        for batch_idx, (req_idx, num_tokens) in enumerate(zip(idx_mapping_np, num_scheduled_tokens)):
            num_computed = int(self.req_states.num_computed_tokens_np[req_idx])
            start = int(query_start_loc_np[batch_idx])
            end = start + int(num_tokens)
            self.positions_cpu[start:end] = torch.arange(num_computed, num_computed + num_tokens)
            self.input_buffers.seq_lens_cpu[batch_idx] = num_computed + num_tokens

        positions.copy_(self.positions_cpu[: positions.shape[0]], non_blocking=True)
        seq_lens.copy_(self.input_buffers.seq_lens_cpu, non_blocking=True)

    def _combine_sampled_and_draft_tokens(
        self,
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
        num_bonus_tokens,
        *,
        idx_mapping_np,
        query_start_loc_np,
        seq_lens_np,
        prefill_len_np,
    ):
        del query_start_loc, seq_lens, prefill_len, draft_tokens, cu_num_logits, num_bonus_tokens
        if num_logits != len(idx_mapping_np):
            raise NotImplementedError("310P Model Runner V2 first release does not support draft tokens.")

        logits_indices_np = np.empty(num_logits, dtype=np.int64)
        for batch_idx, req_idx in enumerate(idx_mapping_np):
            query_end = int(query_start_loc_np[batch_idx + 1])
            logits_indices_np[batch_idx] = query_end - 1
            if seq_lens_np[batch_idx] > prefill_len_np[batch_idx]:
                input_ids[query_end - 1 : query_end].copy_(last_sampled_tokens[req_idx])
        return async_copy_to_gpu(logits_indices_np, device=self.device)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Prepare attention metadata exclusively from existing CPU mirrors."""
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping_np,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )

        positions_np = np.empty(input_batch.num_tokens, dtype=np.int64)
        for batch_idx, (start_position, num_scheduled_tokens) in enumerate(
            zip(input_batch.num_computed_tokens_np, input_batch.num_scheduled_tokens)
        ):
            start = int(input_batch.query_start_loc_np[batch_idx])
            end = start + int(num_scheduled_tokens)
            positions_np[start:end] = np.arange(start_position, start_position + num_scheduled_tokens, dtype=np.int64)

        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping_np,
            input_batch.query_start_loc_np,
            positions_np,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: AscendInputBatch,
        grammar_output: GrammarOutput | None,
    ):
        if grammar_output is not None:
            raise NotImplementedError("Structured output postprocessing is deferred to the second 310P V2 release.")

        logits = self.model.compute_logits(hidden_states[input_batch.logits_indices])
        sampler_output = self.sampler(logits, input_batch)
        can_sample_np = input_batch.seq_lens_np[: input_batch.num_reqs] >= input_batch.prefill_len_np
        num_sampled = async_copy_to_gpu(can_sample_np.astype(np.int32), device=self.device)
        num_rejected = torch.zeros_like(num_sampled)
        return sampler_output, num_sampled, num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        del num_rejected
        nonnegative = idx_mapping >= 0
        valid_indices = idx_mapping.masked_select(nonnegative)
        sampled = sampled_tokens[:, 0].masked_select(nonnegative).to(self.req_states.last_sampled_tokens.dtype)
        valid_num_sampled = num_sampled.masked_select(nonnegative)
        has_sample = valid_num_sampled > 0

        token_positions = self.req_states.total_len.gpu[valid_indices].to(torch.int64)
        old_tokens = self.req_states.all_token_ids.gpu[valid_indices, token_positions]
        tokens_to_store = torch.where(has_sample, sampled.to(torch.int32), old_tokens)
        self.req_states.all_token_ids.gpu.index_put_((valid_indices, token_positions), tokens_to_store)

        old_last = self.req_states.last_sampled_tokens[valid_indices, 0]
        last_to_store = torch.where(has_sample, sampled, old_last)
        self.req_states.last_sampled_tokens.index_copy_(0, valid_indices, last_to_store.unsqueeze(-1))
        self.req_states.total_len.gpu.index_add_(0, valid_indices, valid_num_sampled)

        if query_start_loc is not None:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            computed_delta = query_lens.masked_select(nonnegative)
            self.req_states.num_computed_tokens.gpu.index_add_(
                0,
                valid_indices,
                computed_delta.to(self.req_states.num_computed_tokens.gpu.dtype),
            )

        self.model_state.postprocess_state(idx_mapping, num_sampled)
        self._copy_num_computed_tokens_to_cpu()

    def postprocess_num_computed_tokens(self, input_batch: AscendInputBatch) -> None:
        query_lens = input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]
        self.req_states.num_computed_tokens.gpu.index_add_(
            0,
            input_batch.idx_mapping,
            query_lens.to(self.req_states.num_computed_tokens.gpu.dtype),
        )
        self._copy_num_computed_tokens_to_cpu()

    def _init_kv_zero_meta(self) -> None:
        self.kv_block_zeroer = AscendKVBlockZeroer310V2(self.device, self.pin_memory)
        self.kv_block_zeroer.init_meta(
            attn_groups_iter=(group for groups in self.attn_groups for group in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=getattr(self, "runner_only_attn_layers", set()),
            static_forward_context=self.compilation_config.static_forward_context,
        )
