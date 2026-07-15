# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.llm_base_proposer import greedy_sample
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample


class AscendDSparkProposer(AscendDflashProposer):
    """DeepSeek V4 DSpark block proposer.

    The draft model uses the target model's selected hidden layers as context
    and emits one DSpark draft block in a single model forward.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        draft_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
        if target_layer_ids:
            self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size() * len(target_layer_ids)
            self.hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device)
            self._dflash_hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device)
        self.method = "dflash"
        self.parallel_drafting = True
        self.extra_slots_per_request = self.num_speculative_tokens
        self.net_num_new_slots_per_request = self.num_speculative_tokens
        self.needs_extra_input_slots = True
        self.parallel_drafting_token_id = getattr(
            draft_hf_config,
            "ptd_token_id",
            getattr(draft_hf_config, "dspark_noise_token_id", 0),
        )
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self.positions = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self._slot_mapping_buffer = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self._request_slots_buffer = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self._context_request_slots_buffer = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        self._dspark_sampling_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        self._dspark_idx_mapping_buffer = torch.arange(self.max_batch_size, dtype=torch.int32, device=device)
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, self.num_speculative_tokens), dtype=torch.int64, device=device)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self._dspark_max_request_slots = max(1, int(getattr(scheduler_config, "max_num_seqs", self.max_batch_size) or self.max_batch_size))
        self._dspark_req_id_to_slot: dict[str, int] = {}
        self._dspark_free_slots = list(range(self._dspark_max_request_slots))
        self._dspark_slots_to_reset: list[int] = []
        self.arange_dspark = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)
        self._dspark_last_draft_logits: torch.Tensor | None = None
        self._dspark_last_draft_probs: torch.Tensor | None = None
        self._dspark_last_confidence: torch.Tensor | None = None
        self._dspark_last_req_ids: list[str] | None = None
        self._dspark_probabilistic = vllm_config.speculative_config.draft_sample_method == "probabilistic"
        self.use_cuda_graph = self.use_cuda_graph and not self._dspark_probabilistic
        self._dspark_confidence_threshold = float(getattr(draft_hf_config, "dspark_confidence_threshold", 0.0) or 0.0)
        if not 0.0 <= self._dspark_confidence_threshold <= 1.0:
            raise ValueError(f"dspark_confidence_threshold must be in [0.0, 1.0], got {self._dspark_confidence_threshold}")
        self._runnable = self._run_dspark_draft

    def load_model(self, model: torch.nn.Module) -> None:
        super().load_model(model)
        self._runnable = self._run_dspark_draft
        if (
            self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and self.use_cuda_graph
        ):
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_dspark_draft,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

    def take_last_draft_logits(self) -> torch.Tensor | None:
        draft_logits = self._dspark_last_draft_logits
        self._dspark_last_draft_logits = None
        return draft_logits

    def take_last_draft_probs(self, num_draft_tokens: list[int] | None = None) -> torch.Tensor | None:
        draft_probs = self._dspark_last_draft_probs
        self._dspark_last_draft_probs = None
        self._dspark_last_req_ids = None
        if draft_probs is None or num_draft_tokens is None:
            return draft_probs
        rows = []
        for req_idx, num_tokens in enumerate(num_draft_tokens):
            if num_tokens > 0:
                rows.append(draft_probs[req_idx, :num_tokens])
        if not rows:
            return None
        return torch.cat(rows, dim=0).contiguous()

    def _dspark_confident_prefix_length(self, confidence: torch.Tensor | None, max_tokens: int) -> int:
        if max_tokens == 0 or self._dspark_confidence_threshold <= 0.0:
            return max_tokens
        if confidence is None:
            return 0
        confidence = confidence.float().reshape(confidence.shape[0], -1)[:, :max_tokens]
        below_threshold = confidence.sigmoid() < self._dspark_confidence_threshold
        if not bool(below_threshold.any().item()):
            return max_tokens
        first_low = below_threshold.to(torch.int64).argmax(dim=1)
        full_len = torch.full_like(first_low, max_tokens)
        prefix_len = torch.where(below_threshold.any(dim=1), first_low, full_len)
        return int(prefix_len.min().item())

    def take_last_draft_probs_by_req_id(self) -> tuple[torch.Tensor | None, list[str] | None]:
        draft_probs = self._dspark_last_draft_probs
        req_ids = self._dspark_last_req_ids
        self._dspark_last_draft_probs = None
        self._dspark_last_req_ids = None
        return draft_probs, req_ids

    def _current_req_ids(self, batch_size: int) -> list[str] | None:
        if self.runner is None or not hasattr(self.runner, "input_batch"):
            return None
        req_ids = getattr(self.runner.input_batch, "req_ids", None)
        if req_ids is None:
            return None
        return list(req_ids[:batch_size])

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        del kv_cache_config
        kernel_block_size = kernel_block_sizes
        while isinstance(kernel_block_size, list):
            kernel_block_size = kernel_block_size[0] if kernel_block_size else None
        if kernel_block_size is not None:
            self.kernel_block_size = int(kernel_block_size)
            self.block_size = int(kernel_block_size)
        self._draft_attn_layer_names = set()
        self.attn_layer_names = []
        self.piece_all_attn_layer_name = [[] for _ in range(self.num_speculative_tokens)]
        self.draft_attn_groups = []
        self.kv_cache_gid = 0

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        del dummy_compute_logits, kwargs
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        block_size = self.num_speculative_tokens
        if num_reqs <= 0:
            num_reqs = max(1, min(self.max_batch_size, (num_tokens + block_size - 1) // block_size))
        num_query_total = min(num_reqs * block_size, self.max_query_tokens)
        num_context = min(num_tokens, self.max_num_tokens)

        num_input_tokens, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(
            num_query_total, is_draft_model=True
        )
        num_input_tokens = min(num_input_tokens, num_query_total)
        if num_tokens_across_dp is not None:
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            if num_tokens_across_dp[dp_rank] != num_input_tokens:
                num_tokens_across_dp = num_tokens_across_dp.clone()
                num_tokens_across_dp[dp_rank] = num_input_tokens
        self.input_ids[:num_input_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[:num_input_tokens].copy_(self.arange_dspark[:num_input_tokens])
        self._request_slots_buffer[:num_input_tokens].zero_()
        self._slot_mapping_buffer[:num_input_tokens].copy_(self.arange_dspark[:num_input_tokens])

        context_positions = self._context_positions_buffer[:num_context]
        context_positions.copy_(self.arange_dspark[:num_context])
        context_request_slots = self._context_request_slots_buffer[:num_context]
        context_request_slots.zero_()
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context].copy_(self.hidden_states[:num_context])
        if num_input_tokens % block_size != 0:
            raise ValueError(
                f"DSpark graph capture requires a multiple of block_size tokens, got "
                f"{num_input_tokens} tokens for block_size={block_size}"
            )
        graph_batch_size = num_input_tokens // block_size
        num_sample = graph_batch_size * block_size
        self.token_indices_to_sample[:num_sample].copy_(self.arange_dspark[:num_sample])

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            self._prepare_dspark_context_cache()
            if is_profile:
                forward_context = get_forward_context()
                if forward_context is not None:
                    forward_context.moe_layer_index = 0
                self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self.positions[:num_input_tokens],
                    inputs_embeds=None,
                    request_slots=self._request_slots_buffer[:num_input_tokens],
                    slot_mapping=self._slot_mapping_buffer[:num_input_tokens],
                )
            else:
                forward_context = get_forward_context()
                if forward_context is not None:
                    forward_context.moe_layer_index = 0
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=graph_batch_size,
                    token_indices_to_sample=self.token_indices_to_sample[:num_sample],
                    target_positions=self.positions[:num_input_tokens],
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_input_tokens, [])

    def _update_full_graph_params(self, forward_context, num_tokens, draft_attn_metadatas=None):
        if not self.draft_attn_groups:
            return
        super()._update_full_graph_params(forward_context, num_tokens, draft_attn_metadatas)

    def _assign_request_slots(self, batch_size: int) -> list[int]:
        if self.runner is None or not hasattr(self.runner, "input_batch"):
            return list(range(batch_size))
        input_batch = self.runner.input_batch
        req_ids = list(input_batch.req_ids[:batch_size])
        active_req_ids = set(input_batch.req_ids[: input_batch.num_reqs])
        stale_req_ids = [req_id for req_id in self._dspark_req_id_to_slot if req_id not in active_req_ids]
        for req_id in stale_req_ids:
            slot = self._dspark_req_id_to_slot.pop(req_id)
            if slot not in self._dspark_free_slots:
                self._dspark_free_slots.append(slot)
        self._dspark_free_slots.sort()
        slots: list[int] = []
        self._dspark_slots_to_reset = []
        for req_id in req_ids:
            if req_id not in self._dspark_req_id_to_slot:
                if not self._dspark_free_slots:
                    raise ValueError("No free DSpark request cache slots")
                slot = self._dspark_free_slots.pop(0)
                self._dspark_req_id_to_slot[req_id] = slot
                self._dspark_slots_to_reset.append(slot)
            slots.append(self._dspark_req_id_to_slot[req_id])
        return slots

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        del token_indices_to_sample, req_scheduled_tokens, long_seq_metadata
        is_prefill = (num_decode_reqs == 0 and num_prefill_reqs > 0)
        batch_size = num_prefill_reqs + num_decode_reqs
        block_size = self.num_speculative_tokens
        num_query_total = batch_size * block_size
        request_slots = self._assign_request_slots(batch_size)
        request_slots_tensor = torch.tensor(request_slots, dtype=torch.int32, device=self.device)
        self._dspark_seed_buffer[:batch_size].copy_(next_token_ids)
        has_num_rejected = num_rejected_tokens_gpu is not None
        query_start_loc = cad.query_start_loc[: batch_size + 1]
        query_start = query_start_loc[:-1]
        query_end = query_start_loc[1:]
        if has_num_rejected:
            assert num_rejected_tokens_gpu is not None
            valid_query_end = query_end - num_rejected_tokens_gpu[:batch_size].to(query_end.dtype)
        else:
            valid_query_end = query_end
        valid_query_end = torch.maximum(query_start, valid_query_end)

        num_context = min(target_token_ids.shape[0], target_hidden_states.shape[0], target_positions.shape[0])
        self._dflash_num_context = num_context
        if num_context > 0:
            context_token_indices = self.arange_dspark[:num_context]
            context_req_indices = torch.searchsorted(query_end, context_token_indices, right=True).to(torch.long)
            context_req_indices = torch.clamp(context_req_indices, max=batch_size - 1)
            valid_context_end = valid_query_end.index_select(0, context_req_indices)
            valid_context_mask = context_token_indices < valid_context_end
            invalid_context_position = torch.full_like(target_positions[:num_context], -1)

            self._dflash_hidden_states[:num_context] = target_hidden_states[:num_context]
            self._context_positions_buffer[:num_context] = torch.where(
                valid_context_mask,
                target_positions[:num_context].to(self._context_positions_buffer.dtype),
                invalid_context_position.to(self._context_positions_buffer.dtype),
            )
            self._context_request_slots_buffer[:num_context] = request_slots_tensor.index_select(0, context_req_indices)
            if getattr(cad, "slot_mapping", None) is not None:
                self._context_slot_mapping_buffer[:num_context] = torch.where(
                    valid_context_mask,
                    cad.slot_mapping[:num_context].to(self._context_slot_mapping_buffer.dtype),
                    torch.full_like(self._context_slot_mapping_buffer[:num_context], -1),
                )
        if is_prefill:
            return num_query_total, token_indices_to_sample, cad, None
        token_indices_to_sample = torch.arange(num_query_total, dtype=torch.int32, device=self.device)
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
        last_token_indices = torch.clamp(valid_query_end - 1, min=0, max=target_positions.shape[0] - 1).to(torch.long)
        last_positions = target_positions.index_select(0, last_token_indices).to(self.positions.dtype)
        draft_offsets = self.arange_dspark[:block_size].view(1, block_size)
        draft_positions = last_positions.view(batch_size, 1) + 1 + draft_offsets
        if max_model_len > 0:
            exceeds_max_model_len = draft_positions >= max_model_len
            draft_positions = torch.where(exceeds_max_model_len, torch.zeros_like(draft_positions), draft_positions)
        else:
            exceeds_max_model_len = torch.zeros_like(draft_positions, dtype=torch.bool)
        self.positions[:num_query_total] = draft_positions.flatten()
        input_ids_view = self.input_ids[:num_query_total].view(batch_size, block_size)
        input_ids_view.fill_(self.parallel_drafting_token_id)
        input_ids_view[:, 0].copy_(next_token_ids[:batch_size])
        self._request_slots_buffer[:num_query_total] = (
            request_slots_tensor.view(batch_size, 1).expand(-1, block_size).flatten()
        )
        if getattr(cad, "block_table_tensor", None) is not None:
            block_nums = draft_positions // self.kernel_block_size
            block_offsets = draft_positions % self.kernel_block_size
            block_ids = torch.gather(cad.block_table_tensor[:batch_size], 1, block_nums.long())
            slot_mapping = block_ids.to(torch.int32) * self.kernel_block_size + block_offsets
        else:
            slot_mapping = draft_positions.to(torch.int32)
        slot_mapping = torch.where(
            exceeds_max_model_len,
            torch.full_like(slot_mapping, -1),
            slot_mapping,
        )
        self._slot_mapping_buffer[:num_query_total] = slot_mapping.flatten()
        effective_seq_lens = cad.seq_lens - num_rejected_tokens_gpu if has_num_rejected else cad.seq_lens
        cad.query_start_loc = self.arange_dspark[: batch_size + 1] * block_size
        cad.seq_lens = effective_seq_lens + block_size
        if max_model_len > 0:
            cad.seq_lens = cad.seq_lens.clamp(max=max_model_len)
        cad.query_start_loc_cpu = (torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * block_size).to(torch.int32)
        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [block_size] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = block_size
        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = block_size
        cad.max_seq_len = cad.max_seq_len + block_size
        if max_model_len > 0:
            cad.max_seq_len = min(cad.max_seq_len, max_model_len)
        cad.num_reqs = batch_size
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        cad.positions = self.positions[:num_query_total]
        cad.causal = False
        cad.attn_mask = None
        return num_query_total, token_indices_to_sample, cad, None

    def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self.positions[:num_input_tokens],
            "inputs_embeds": None,
            "request_slots": self._request_slots_buffer[:num_input_tokens],
            "slot_mapping": self._slot_mapping_buffer[:num_input_tokens],
        }

    def _reset_pending_request_slots(self) -> None:
        if not self._dspark_slots_to_reset:
            return
        reset_slots = torch.tensor(self._dspark_slots_to_reset, dtype=torch.int32, device=self.device)
        self.model.reset_request_slots(reset_slots)
        self._dspark_slots_to_reset = []

    def _prepare_dspark_context_cache(self) -> None:
        self._reset_pending_request_slots()
        num_context = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
            self._context_request_slots_buffer[:num_context],
        )

    def _pad_dspark_decode_inputs(self, num_tokens: int, num_input_tokens: int) -> None:
        if num_input_tokens <= num_tokens:
            return
        self.input_ids[num_tokens:num_input_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[num_tokens:num_input_tokens].zero_()
        self._request_slots_buffer[num_tokens:num_input_tokens].zero_()
        self._slot_mapping_buffer[num_tokens:num_input_tokens].zero_()

    def _get_draft_idx_mapping(self, num_reqs: int, device: torch.device) -> torch.Tensor:
        return self._dspark_idx_mapping_buffer[:num_reqs].to(device=device, dtype=torch.int32).contiguous()

    def _get_runner_idx_mapping(self, num_reqs: int, device: torch.device) -> torch.Tensor | None:
        input_batch = getattr(getattr(self, "runner", None), "input_batch", None)
        runner_idx_mapping = getattr(input_batch, "idx_mapping", None)
        if isinstance(runner_idx_mapping, torch.Tensor):
            return runner_idx_mapping[:num_reqs].to(device=device, dtype=torch.long).contiguous()
        return None

    def _get_draft_sampling_temperature(self, sampling_metadata: SamplingMetadata, num_reqs: int, device: torch.device) -> torch.Tensor:
        temperature = sampling_metadata.temperature
        if temperature is None:
            default = 0.0 if sampling_metadata.all_greedy else 1.0
            return torch.full((num_reqs,), default, dtype=torch.float32, device=device)
        runner_idx_mapping = self._get_runner_idx_mapping(num_reqs, temperature.device)
        if runner_idx_mapping is not None and temperature.shape[0] > num_reqs:
            return temperature.index_select(0, runner_idx_mapping).to(device=device, dtype=torch.float32).contiguous()
        if runner_idx_mapping is not None and temperature.shape[0] == num_reqs:
            return temperature.to(device=device, dtype=torch.float32).contiguous()
        return temperature[:num_reqs].to(device=device, dtype=torch.float32).contiguous()

    def _get_draft_sampling_seeds(self, sampling_metadata: SamplingMetadata, num_reqs: int, device: torch.device) -> torch.Tensor:
        runner = getattr(self, "runner", None)
        sampler = getattr(runner, "sampler", None)
        sampling_states = getattr(sampler, "sampling_states", None)
        runner_seeds = getattr(getattr(sampling_states, "seeds", None), "gpu", None)
        if isinstance(runner_seeds, torch.Tensor):
            runner_idx_mapping = self._get_runner_idx_mapping(num_reqs, runner_seeds.device)
            if runner_idx_mapping is not None and runner_seeds.shape[0] > num_reqs:
                return runner_seeds.index_select(0, runner_idx_mapping).to(device=device, dtype=torch.int64).contiguous()
            return runner_seeds[:num_reqs].to(device=device, dtype=torch.int64).contiguous()
        seeds = self._dspark_sampling_seed_buffer[:num_reqs]
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        base_seed = int(getattr(model_config, "seed", 0) or 0)
        runner_idx_mapping = self._get_runner_idx_mapping(num_reqs, device)
        runner_idx_mapping_cpu = runner_idx_mapping.cpu().tolist() if runner_idx_mapping is not None else None
        for req_idx in range(num_reqs):
            generator_idx = runner_idx_mapping_cpu[req_idx] if runner_idx_mapping_cpu is not None else req_idx
            generator = sampling_metadata.generators.get(generator_idx)
            seeds[req_idx] = int(generator.initial_seed()) if generator is not None else base_seed + req_idx * 9973
        return seeds.to(device=device, dtype=torch.int64).contiguous()

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
        truncate_by_confidence: bool = True,
    ) -> torch.Tensor:
        block_size = self.num_speculative_tokens
        num_sample = num_reqs * block_size
        sample_hidden_states = head_hidden[token_indices_to_sample[:num_sample]]
        head_hidden = self.model.model.compute_head_hidden(sample_hidden_states)
        base_logits = self.model.compute_logits(sample_hidden_states)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, block_size, vocab_size)
        use_probabilistic = sampling_metadata is not None and self._dspark_probabilistic and not sampling_metadata.all_greedy
        draft_logits = torch.empty((num_reqs, block_size, vocab_size), dtype=torch.float32, device=base_logits.device) if use_probabilistic else None
        self._dspark_last_draft_logits = None
        self._dspark_last_draft_probs = None
        self._dspark_last_confidence = None
        self._dspark_last_req_ids = None
        prev_ids = self._dspark_seed_buffer[:num_reqs]
        gumbel_positions = self.positions[:num_sample].view(num_reqs, block_size)
        markov_embeds = []
        for idx in range(block_size):
            markov_embed = self.model.markov_embed(prev_ids)
            markov_bias = self.model.markov_bias(markov_embed)
            logits = base_logits[:, idx, :] + markov_bias
            if use_probabilistic:
                assert sampling_metadata is not None and draft_logits is not None
                draft_ids = gumbel_sample(
                    logits.contiguous(),
                    self._get_draft_idx_mapping(num_reqs, logits.device),
                    self._get_draft_sampling_temperature(sampling_metadata, num_reqs, logits.device),
                    self._get_draft_sampling_seeds(sampling_metadata, num_reqs, logits.device),
                    (gumbel_positions[:, idx].to(device=logits.device, dtype=torch.int32) - 1).contiguous(),
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=torch.tensor(idx, dtype=torch.int32, device=logits.device),
                )
            else:
                draft_ids = greedy_sample(logits)
            self._dspark_draft_buffer[:num_reqs, idx].copy_(draft_ids)
            markov_embeds.append(markov_embed)
            prev_ids = self._dspark_draft_buffer[:num_reqs, idx]
        if use_probabilistic:
            assert draft_logits is not None
            self._dspark_last_draft_logits = draft_logits.contiguous()
        confidence = self.model.confidence(head_hidden.view(num_reqs, block_size, -1), torch.stack(markov_embeds, dim=1))
        self._dspark_last_confidence = confidence
        proposal_len = block_size
        if truncate_by_confidence:
            proposal_len = self._dspark_confident_prefix_length(confidence, block_size)
        if truncate_by_confidence and use_probabilistic and proposal_len > 0:
            assert draft_logits is not None
            self._dspark_last_draft_probs = draft_logits[:, :proposal_len, :].softmax(dim=-1, dtype=torch.float32).contiguous()
            self._dspark_last_req_ids = self._current_req_ids(num_reqs)
        return self._dspark_draft_buffer[:num_reqs, :proposal_len]

    def _truncate_dspark_draft_tokens(
        self,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
        sampling_metadata: SamplingMetadata | None,
    ) -> torch.Tensor:
        proposal_len = self._dspark_confident_prefix_length(
            self._dspark_last_confidence,
            self.num_speculative_tokens,
        )
        use_probabilistic = (
            sampling_metadata is not None
            and self._dspark_probabilistic
            and not sampling_metadata.all_greedy
        )
        if use_probabilistic and proposal_len > 0:
            assert self._dspark_last_draft_logits is not None
            self._dspark_last_draft_probs = self._dspark_last_draft_logits[
                :, :proposal_len, :
            ].softmax(dim=-1, dtype=torch.float32).contiguous()
            self._dspark_last_req_ids = self._current_req_ids(num_reqs)
        return draft_token_ids[:num_reqs, :proposal_len]

    def _run_dspark_draft(
        self,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        multi_steps_attn_metadata: list[dict[str, Any]] | None = None,
        num_tokens: int | None = None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        del target_positions, inputs_embeds, multi_steps_attn_metadata, num_tokens
        model_inputs = self.build_model_inputs_first_pass(num_input_tokens)
        hidden_states = self.model(**model_inputs)
        return self._sample_sequential(
            batch_size,
            hidden_states,
            token_indices_to_sample,
            sampling_metadata,
            truncate_by_confidence=False,
        )

    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput | None = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mm_embed_inputs, scheduler_output, num_scheduled_tokens
        is_prefill = (num_decode_reqs == 0 and num_prefill_reqs > 0)
        num_tokens, _, _, _ = self.set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
        )
        assert self.runner is not None
        actual_num_reqs = common_attn_metadata.num_reqs
        block_size = self.num_speculative_tokens
        use_cuda_graph = self.use_cuda_graph and not is_prefill
        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        uniform_decode = target_model_batch_desc.uniform
        if use_cuda_graph:
            _, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
            )
            num_input_tokens = batch_descriptor.num_tokens
        else:
            num_input_tokens = num_tokens

        num_input_tokens, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(
            num_input_tokens,
            is_draft_model=True,
        )

        if use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
            )
            num_input_tokens = batch_descriptor.num_tokens
            self._pad_dspark_decode_inputs(num_tokens, num_input_tokens)
            if num_input_tokens % block_size != 0:
                dspark_padded = ((num_input_tokens + block_size - 1) // block_size) * block_size
                self._pad_dspark_decode_inputs(num_input_tokens, dspark_padded)
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp = num_tokens_across_dp.clone()
                    num_tokens_across_dp[:] = dspark_padded
                num_input_tokens = dspark_padded
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        graph_batch_size = num_input_tokens // block_size
        num_sample = graph_batch_size * block_size
        self.token_indices_to_sample[:num_sample].copy_(self.arange_dspark[:num_sample])

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            self._prepare_dspark_context_cache()
            if is_prefill:
                return common_attn_metadata.num_reqs.new_zeros(
                    common_attn_metadata.num_reqs, 0, dtype=torch.long
                )
            else:
                run_kwargs = dict(
                    num_input_tokens=num_input_tokens,
                    batch_size=graph_batch_size,
                    token_indices_to_sample=self.token_indices_to_sample[:num_sample],
                    target_positions=target_positions,
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                    sampling_metadata=sampling_metadata,
                )
                if self.enable_enpu:
                    self._update_full_graph_params_if_needed(forward_context, num_input_tokens, [])
                    draft_token_ids = self._runnable(**run_kwargs)
                else:
                    draft_token_ids = self._runnable(**run_kwargs)
                    self._update_full_graph_params_if_needed(forward_context, num_input_tokens, [])
                draft_token_ids = self._truncate_dspark_draft_tokens(
                    draft_token_ids,
                    actual_num_reqs,
                    sampling_metadata,
                )
                return draft_token_ids
