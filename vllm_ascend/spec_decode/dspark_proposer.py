# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections import defaultdict
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.llm_base_proposer import greedy_sample
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample


def _dspark_reject_debug_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_DSPARK_REJECT_DEBUG", "0") == "1"


def _dspark_standard_dsa_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "0") == "1"


def _debug_tensor_head(name: str, tensor: torch.Tensor, limit: int = 16) -> str:
    flat = tensor.detach().flatten()
    return f"{name}={flat[:limit].cpu().tolist()}"


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        draft_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        self._dspark_probabilistic = vllm_config.speculative_config.draft_sample_method == "probabilistic"
        self._dspark_last_draft_logits: torch.Tensor | None = None
        self._dspark_last_draft_probs: torch.Tensor | None = None
        dspark_target_layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
        if dspark_target_layer_ids:
            self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size() * len(
                dspark_target_layer_ids
            )
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            self._dflash_hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        self.method = "dflash"
        self.parallel_drafting = True
        self.extra_slots_per_request = self.num_speculative_tokens
        self.net_num_new_slots_per_request = self.num_speculative_tokens
        self.needs_extra_input_slots = True
        self.is_rejected_token_mask: torch.Tensor | None = getattr(self, "is_rejected_token_mask", None)
        if self.is_rejected_token_mask is None:
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,),
                dtype=torch.bool,
                device=device,
            )
        self.is_masked_token_mask: torch.Tensor | None = getattr(self, "is_masked_token_mask", None)
        if self.is_masked_token_mask is None:
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,),
                dtype=torch.bool,
                device=device,
            )
        self.parallel_drafting_token_id = getattr(
            draft_hf_config,
            "ptd_token_id",
            getattr(draft_hf_config, "dspark_noise_token_id", 0),
        )
        self.use_cuda_graph = False
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._request_slots_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._context_request_slots_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._dspark_seed_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._dspark_sampling_seed_buffer = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._dspark_idx_mapping_buffer = torch.arange(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._dspark_draft_buffer = torch.zeros(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.int64,
            device=device,
        )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self._dspark_max_request_slots = max(
            1,
            int(getattr(scheduler_config, "max_num_seqs", self.max_batch_size) or self.max_batch_size),
        )
        self._dspark_req_id_to_slot: dict[str, int] = {}
        self._dspark_free_slots = list(range(self._dspark_max_request_slots))
        self._dspark_slots_to_reset: list[int] = []
        self.arange_dspark = torch.arange(
            self.max_positions + 1,
            device=device,
            dtype=torch.int32,
        )

    def take_last_draft_logits(self) -> torch.Tensor | None:
        draft_logits = self._dspark_last_draft_logits
        self._dspark_last_draft_logits = None
        return draft_logits

    def take_last_draft_probs(self) -> torch.Tensor | None:
        draft_probs = self._dspark_last_draft_probs
        self._dspark_last_draft_probs = None
        return draft_probs

    def _get_draft_sampling_temperature(
        self,
        sampling_metadata: SamplingMetadata,
        num_reqs: int,
        device: torch.device,
    ) -> torch.Tensor:
        temperature = sampling_metadata.temperature
        if temperature is None:
            default = 0.0 if sampling_metadata.all_greedy else 1.0
            return torch.full((num_reqs,), default, dtype=torch.float32, device=device)
        return temperature[:num_reqs].to(device=device, dtype=torch.float32).contiguous()

    def _get_runner_sampling_state_seeds(self, num_reqs: int, device: torch.device) -> torch.Tensor | None:
        runner = getattr(self, "runner", None)
        sampler = getattr(runner, "sampler", None)
        sampling_states = getattr(sampler, "sampling_states", None)
        seeds = getattr(getattr(sampling_states, "seeds", None), "gpu", None)
        if not isinstance(seeds, torch.Tensor):
            return None

        input_batch = getattr(runner, "input_batch", None)
        idx_mapping = getattr(input_batch, "idx_mapping", None)
        if isinstance(idx_mapping, torch.Tensor):
            return seeds.index_select(0, idx_mapping[:num_reqs].to(device=seeds.device, dtype=torch.long)).to(device)
        return seeds[:num_reqs].to(device)

    def _get_draft_sampling_seeds(
        self,
        sampling_metadata: SamplingMetadata,
        num_reqs: int,
        device: torch.device,
    ) -> torch.Tensor:
        runner_seeds = self._get_runner_sampling_state_seeds(num_reqs, device)
        if runner_seeds is not None:
            return runner_seeds.to(dtype=torch.int64).contiguous()

        seed_buffer = getattr(self, "_dspark_sampling_seed_buffer", None)
        if isinstance(seed_buffer, torch.Tensor) and seed_buffer.numel() >= num_reqs:
            seeds = seed_buffer[:num_reqs]
        else:
            seeds = torch.empty((num_reqs,), dtype=torch.int64, device=device)

        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        base_seed = int(getattr(model_config, "seed", 0) or 0)
        for req_idx in range(num_reqs):
            generator = sampling_metadata.generators.get(req_idx)
            if generator is not None:
                seeds[req_idx] = int(generator.initial_seed())
            else:
                seeds[req_idx] = base_seed + req_idx * 9973
        return seeds.to(device=device, dtype=torch.int64).contiguous()

    def _get_draft_idx_mapping(self, num_reqs: int, device: torch.device) -> torch.Tensor:
        idx_mapping = getattr(self, "_dspark_idx_mapping_buffer", None)
        if isinstance(idx_mapping, torch.Tensor) and idx_mapping.numel() >= num_reqs:
            return idx_mapping[:num_reqs].to(device=device, dtype=torch.int32).contiguous()
        return torch.arange(num_reqs, dtype=torch.int32, device=device)

    def _sample_draft_ids(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        num_reqs: int,
        step_idx: int,
        gumbel_positions: torch.Tensor,
    ) -> torch.Tensor:
        return gumbel_sample(
            logits.contiguous(),
            self._get_draft_idx_mapping(num_reqs, logits.device),
            self._get_draft_sampling_temperature(sampling_metadata, num_reqs, logits.device),
            self._get_draft_sampling_seeds(sampling_metadata, num_reqs, logits.device),
            (gumbel_positions.to(device=logits.device, dtype=torch.int32) - 1).contiguous(),
            apply_temperature=True,
            output_processed_logits=draft_logits,
            output_processed_logits_col=torch.tensor(step_idx, dtype=torch.int32, device=logits.device),
            use_fp64=getattr(self, "use_fp64_gumbel", False),
        )

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
                    raise ValueError(
                        "No free DSpark request cache slots: "
                        f"batch_size={batch_size}, max_request_slots={self._dspark_max_request_slots}"
                    )
                slot = self._dspark_free_slots.pop(0)
                self._dspark_req_id_to_slot[req_id] = slot
                self._dspark_slots_to_reset.append(slot)
            slots.append(self._dspark_req_id_to_slot[req_id])
        return slots

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        self._draft_attn_layer_names: set[str] = set()
        self.attn_layer_names: list[str] = []
        self.piece_all_attn_layer_name: list[list[str]] = [[] for _ in range(self.num_speculative_tokens)]
        self.draft_attn_groups: list[Any] = []
        self.kv_cache_gid = 0

        if _dspark_standard_dsa_enabled() and hasattr(self.model, "get_draft_kv_cache_layer_names"):
            draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
            self._draft_attn_layer_names = draft_attn_layer_names
            self.attn_layer_names = sorted(draft_attn_layer_names)
            self.piece_all_attn_layer_name = [
                [name for name in self.attn_layer_names] for _ in range(self.num_speculative_tokens)
            ]

            layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            )

            for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
                layer_names = [
                    name for name in kv_cache_group_spec.layer_names if name in draft_attn_layer_names
                ]
                if not layer_names:
                    continue

                attn_backend_layers: dict[tuple[str, Any], list[str]] = defaultdict(list)
                attn_backends: dict[tuple[str, Any], tuple[type[Any], Any]] = {}
                for layer_name in layer_names:
                    attn_backend = layers[layer_name].get_attn_backend()
                    kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                        kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
                    key = (attn_backend.full_cls_name(), kv_cache_spec)
                    attn_backends[key] = (attn_backend, kv_cache_spec)
                    attn_backend_layers[key].append(layer_name)

                for key, grouped_layer_names in attn_backend_layers.items():
                    attn_backend, kv_cache_spec = attn_backends[key]
                    metadata_builder = attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        grouped_layer_names,
                        self.vllm_config,
                        self.device,
                    )
                    self.draft_attn_groups.append(
                        AttentionGroup(
                            attn_backend,
                            grouped_layer_names,
                            kv_cache_spec,
                            kv_cache_gid,
                            [metadata_builder],
                        )
                    )

            if self.draft_attn_groups:
                self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
                self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
                self.block_size = self.kernel_block_size
                return
            raise RuntimeError(
                "VLLM_ASCEND_DSPARK_USE_STANDARD_DSA=1 but no DSpark draft "
                f"attention groups were found for layers: {sorted(draft_attn_layer_names)}"
            )

        kernel_block_size = kernel_block_sizes
        while isinstance(kernel_block_size, list):
            if not kernel_block_size:
                kernel_block_size = None
                break
            kernel_block_size = kernel_block_size[0]
        if kernel_block_size is None and kv_cache_config.kv_cache_groups:
            kernel_block_size = getattr(
                kv_cache_config.kv_cache_groups[0].kv_cache_spec,
                "block_size",
                None,
            )
        if kernel_block_size is not None:
            self.kernel_block_size = int(kernel_block_size)
            self.block_size = int(kernel_block_size)

    def _build_standard_dsa_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_input_tokens: int,
        num_actual_tokens: int,
    ) -> list[dict[str, Any]]:
        if not self.draft_attn_groups:
            return []

        if num_input_tokens > num_actual_tokens:
            self.positions[num_actual_tokens:num_input_tokens].fill_(0)
            self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(-1)

        common_attn_metadata.positions = self.positions[:num_input_tokens]
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        common_attn_metadata.num_input_tokens = num_input_tokens
        common_attn_metadata.num_actual_tokens = num_actual_tokens
        common_attn_metadata.causal = False
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill

        per_layer_attn_metadata: dict[str, Any] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata,
                draft_index=1,
                block_size=attn_group.kv_cache_spec.block_size,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return [per_layer_attn_metadata]

    def _pad_draft_query_buffers(
        self,
        num_actual_tokens: int,
        num_input_tokens: int,
    ) -> None:
        if num_input_tokens <= num_actual_tokens:
            return

        self.input_ids[num_actual_tokens:num_input_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[num_actual_tokens:num_input_tokens].fill_(0)
        self._request_slots_buffer[num_actual_tokens:num_input_tokens].fill_(0)
        self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(-1)

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
        del (
            target_token_ids,
            token_indices_to_sample,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
        )
        batch_size = cad.num_reqs
        block_size = self.num_speculative_tokens
        num_query_total = batch_size * block_size
        has_num_rejected = num_rejected_tokens_gpu is not None
        request_slots = self._assign_request_slots(batch_size)
        self._dspark_seed_buffer[:batch_size].copy_(next_token_ids)
        if batch_size < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[batch_size:].fill_(0)

        context_cursor = 0
        for req_idx in range(batch_size):
            request_slot = request_slots[req_idx]
            ctx_start = int(cad.query_start_loc[req_idx].item())
            ctx_end = int(cad.query_start_loc[req_idx + 1].item())
            valid_ctx_end = ctx_end
            if has_num_rejected:
                assert num_rejected_tokens_gpu is not None
                valid_ctx_end -= int(num_rejected_tokens_gpu[req_idx].item())
            valid_ctx_end = max(ctx_start, valid_ctx_end)
            valid_ctx_len = valid_ctx_end - ctx_start
            if valid_ctx_len == 0:
                continue
            out_end = context_cursor + valid_ctx_len
            self._dflash_hidden_states[context_cursor:out_end] = target_hidden_states[ctx_start:valid_ctx_end]
            self._context_positions_buffer[context_cursor:out_end] = target_positions[ctx_start:valid_ctx_end]
            self._context_request_slots_buffer[context_cursor:out_end] = request_slot
            if getattr(cad, "slot_mapping", None) is not None:
                self._context_slot_mapping_buffer[context_cursor:out_end] = cad.slot_mapping[ctx_start:valid_ctx_end]
            context_cursor = out_end
        self._dflash_num_context = context_cursor

        token_indices_to_sample = torch.arange(
            num_query_total,
            dtype=torch.int32,
            device=self.device,
        )

        for req_idx in range(batch_size):
            request_slot = request_slots[req_idx]
            ctx_start = int(cad.query_start_loc[req_idx].item())
            ctx_end = int(cad.query_start_loc[req_idx + 1].item())
            valid_ctx_end = ctx_end
            if has_num_rejected:
                assert num_rejected_tokens_gpu is not None
                valid_ctx_end -= int(num_rejected_tokens_gpu[req_idx].item())
            last_pos = target_positions[valid_ctx_end - 1]
            out_start = req_idx * block_size
            out_end = out_start + block_size
            self.positions[out_start:out_end] = last_pos + 1 + self.arange_dspark[:block_size]
            self.input_ids[out_start] = next_token_ids[req_idx]
            if block_size > 1:
                self.input_ids[out_start + 1 : out_end] = self.parallel_drafting_token_id
            self._request_slots_buffer[out_start:out_end] = request_slot

            if getattr(cad, "block_table_tensor", None) is not None:
                block_nums = self.positions[out_start:out_end] // self.kernel_block_size
                block_offsets = self.positions[out_start:out_end] % self.kernel_block_size
                block_ids = cad.block_table_tensor[req_idx].index_select(0, block_nums.long())
                self._slot_mapping_buffer[out_start:out_end] = (
                    block_ids.to(torch.int32) * self.kernel_block_size + block_offsets
                )
            else:
                self._slot_mapping_buffer[out_start:out_end] = self.positions[out_start:out_end]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dspark[: batch_size + 1] * block_size
        cad.seq_lens = effective_seq_lens + block_size
        cad.query_start_loc_cpu = (torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * block_size).to(
            torch.int32
        )

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [block_size] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = block_size

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = block_size
        cad.max_seq_len = cad.max_seq_len + block_size
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        cad.positions = self.positions[:num_query_total]
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

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
        block_size = self.num_speculative_tokens
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
        )
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_total = min(num_reqs * block_size, num_query_tokens)
        model_num_query_tokens = num_query_total
        if _dspark_standard_dsa_enabled():
            self._pad_draft_query_buffers(num_query_total, num_input_tokens)
            model_num_query_tokens = num_input_tokens

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
            self._dflash_num_context = num_input_tokens
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_input_tokens],
                self._context_positions_buffer[:num_input_tokens],
                None,
                self._context_request_slots_buffer[:num_input_tokens],
            )
            if model_num_query_tokens:
                self.model(
                    input_ids=self.input_ids[:model_num_query_tokens],
                    positions=self.positions[:model_num_query_tokens],
                    inputs_embeds=None,
                    request_slots=self._request_slots_buffer[:model_num_query_tokens],
                    slot_mapping=self._slot_mapping_buffer[:model_num_query_tokens],
                )
            forward_context = get_forward_context()
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not _EXTRA_CTX.capturing
                and self.draft_attn_groups
            ):
                self._update_full_graph_params(forward_context, num_tokens, [])

    def build_model_inputs_first_pass(self, num_input_tokens: int) -> dict[str, Any]:
        num_context = self._dflash_num_context
        if self._dspark_slots_to_reset:
            reset_slots = torch.tensor(self._dspark_slots_to_reset, dtype=torch.int32, device=self.device)
            self.model.reset_request_slots(reset_slots)
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
            self._context_request_slots_buffer[:num_context],
        )
        return dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
            request_slots=self._request_slots_buffer[:num_input_tokens],
            slot_mapping=self._slot_mapping_buffer[:num_input_tokens],
        )

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        block_size = self.num_speculative_tokens
        num_sample = num_reqs * block_size
        sample_hidden_states = head_hidden[token_indices_to_sample[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden_states)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, block_size, vocab_size)

        use_probabilistic = (
            sampling_metadata is not None
            and getattr(self, "_dspark_probabilistic", False)
            and not sampling_metadata.all_greedy
        )
        draft_logits = None
        if use_probabilistic:
            draft_logits = torch.empty(
                (num_reqs, block_size, vocab_size),
                dtype=torch.float32,
                device=base_logits.device,
            )
        self._dspark_last_draft_logits = None
        self._dspark_last_draft_probs = None

        prev_ids = self._dspark_seed_buffer[:num_reqs]
        gumbel_positions = None
        if use_probabilistic:
            gumbel_positions = self.positions[:num_sample].view(num_reqs, block_size)
        for idx in range(block_size):
            markov_embed = self.model.markov_embed(prev_ids)
            markov_bias = self.model.markov_bias(markov_embed)
            logits = base_logits[:, idx, :] + markov_bias
            if use_probabilistic:
                assert sampling_metadata is not None
                assert draft_logits is not None
                draft_ids = self._sample_draft_ids(
                    logits,
                    draft_logits,
                    sampling_metadata,
                    num_reqs,
                    idx,
                    gumbel_positions[:, idx],
                )
            else:
                draft_ids = greedy_sample(logits)
            self._dspark_draft_buffer[:num_reqs, idx].copy_(draft_ids)
            prev_ids = self._dspark_draft_buffer[:num_reqs, idx]
        if use_probabilistic:
            assert draft_logits is not None
            self._dspark_last_draft_logits = draft_logits.contiguous()
        return self._dspark_draft_buffer[:num_reqs, :block_size]

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
        del (
            target_model_batch_desc,
            mm_embed_inputs,
            scheduler_output,
            num_scheduled_tokens,
        )

        num_tokens, token_indices_to_sample, _, _ = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
        assert self.runner is not None

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)
        multi_steps_attn_metadata = (
            self._build_standard_dsa_attn_metadata(common_attn_metadata, num_input_tokens, num_tokens)
            if _dspark_standard_dsa_enabled()
            else []
        )
        model_num_tokens = num_tokens
        if _dspark_standard_dsa_enabled():
            self._pad_draft_query_buffers(num_tokens, num_input_tokens)
            model_num_tokens = num_input_tokens

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=None,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            num_context = self._dflash_num_context
            if self._dspark_slots_to_reset:
                reset_slots = torch.tensor(self._dspark_slots_to_reset, dtype=torch.int32, device=self.device)
                self.model.reset_request_slots(reset_slots)
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states[:num_context],
                self._context_positions_buffer[:num_context],
                self._context_slot_mapping_buffer[:num_context],
                self._context_request_slots_buffer[:num_context],
            )
            hidden_states = self.model(
                input_ids=self.input_ids[:model_num_tokens],
                positions=self.positions[:model_num_tokens],
                inputs_embeds=None,
                request_slots=self._request_slots_buffer[:model_num_tokens],
                slot_mapping=self._slot_mapping_buffer[:model_num_tokens],
            )
            draft_token_ids = self._sample_sequential(
                common_attn_metadata.num_reqs,
                hidden_states,
                token_indices_to_sample,
                sampling_metadata,
            )
            if _dspark_reject_debug_enabled():
                print(
                    "[dspark-propose-debug] "
                    f"num_tokens={num_tokens} "
                    f"num_context={num_context} "
                    f"{_debug_tensor_head('input_ids', self.input_ids[:num_tokens])} "
                    f"{_debug_tensor_head('positions', self.positions[:num_tokens])} "
                    f"{_debug_tensor_head('target_positions', target_positions)} "
                    f"{_debug_tensor_head('next_token_ids', next_token_ids)} "
                    f"{_debug_tensor_head('draft_token_ids', draft_token_ids)}",
                    flush=True,
                )
        return draft_token_ids
