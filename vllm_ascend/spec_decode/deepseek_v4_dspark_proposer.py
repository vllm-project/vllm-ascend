# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from copy import copy
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

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import greedy_sample
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample


def _greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    try:
        reduce_sample = get_ascend_config().enable_reduce_sample
    except RuntimeError:
        reduce_sample = False
    return greedy_sample(logits) if reduce_sample else logits.argmax(dim=-1)


class AscendDeepSeekV4DSparkProposer(AscendDsparkProposer):
    """Eager DeepSeek V4 DSpark proposer using the standard DSA cache."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layers = tuple(draft_config.dspark_target_layer_ids)
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size() * len(target_layers)
        self.hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device)
        self._dflash_hidden_states = torch.zeros_like(self.hidden_states)

        self.use_cuda_graph = False
        self.method = "dflash"
        self.parallel_drafting = True
        self.block_size = self.num_speculative_tokens
        self.extra_slots_per_request = self.block_size
        self.net_num_new_slots_per_request = self.block_size
        self.needs_extra_input_slots = True
        self.parallel_drafting_token_id = getattr(
            draft_config,
            "ptd_token_id",
            getattr(draft_config, "dspark_noise_token_id", 0),
        )
        self.max_query_tokens = self.max_batch_size * self.block_size
        self.positions = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self._seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        self._draft_buffer = torch.zeros(
            (self.max_batch_size, self.block_size),
            dtype=torch.int64,
            device=device,
        )
        self._sampling_seed_buffer = torch.zeros_like(self._seed_buffer)
        self._idx_mapping_buffer = torch.arange(self.max_batch_size, dtype=torch.int32, device=device)
        self._token_to_req_buffer = torch.zeros(self.max_query_tokens, dtype=torch.int32, device=device)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._query_slots_by_gid: dict[int, torch.Tensor] = {}
        self._context_slots_by_gid: dict[int, torch.Tensor] = {}
        self._query_slot_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_buffers: dict[int, torch.Tensor] = {}
        self._block_tables_by_gid: dict[int, torch.Tensor] = {}
        self._last_draft_logits: torch.Tensor | None = None
        self._probabilistic = vllm_config.speculative_config.draft_sample_method == "probabilistic"
        self.arange_dspark = torch.arange(
            self.max_num_tokens + self.max_query_tokens + 1,
            dtype=torch.int32,
            device=device,
        )

    def take_last_draft_logits(self) -> torch.Tensor | None:
        logits = self._last_draft_logits
        self._last_draft_logits = None
        return logits

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        del kernel_block_sizes
        layer_names = list(self.model.get_draft_kv_cache_layer_names())
        if not layer_names:
            raise RuntimeError("DSpark requires registered draft DSA cache layers")
        self._draft_attn_layer_names = set(layer_names)
        self._draft_attn_layer_names_ordered = layer_names
        self.attn_layer_names = sorted(layer_names)
        self.piece_all_attn_layer_name = [self.attn_layer_names[:] for _ in range(self.block_size)]
        self.draft_attn_groups = []

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        wanted = set(layer_names)
        for gid, group_spec in enumerate(kv_cache_config.kv_cache_groups):
            group_layers = [name for name in group_spec.layer_names if name in wanted]
            if not group_layers:
                continue
            backend_layers: dict[tuple[str, Any], list[str]] = defaultdict(list)
            backends: dict[tuple[str, Any], tuple[type[Any], Any]] = {}
            for name in group_layers:
                backend = layers[name].get_attn_backend()
                cache_spec = group_spec.kv_cache_spec
                if isinstance(cache_spec, UniformTypeKVCacheSpecs):
                    cache_spec = cache_spec.kv_cache_specs[name]
                key = (backend.full_cls_name(), cache_spec)
                backend_layers[key].append(name)
                backends[key] = (backend, cache_spec)
            for key, names in backend_layers.items():
                backend, cache_spec = backends[key]
                builder = backend.get_builder_cls()(cache_spec, names, self.vllm_config, self.device)
                if hasattr(builder, "block_size"):
                    builder.block_size = int(cache_spec.block_size)
                self.draft_attn_groups.append(AttentionGroup(backend, names, cache_spec, gid, [builder]))
        if not self.draft_attn_groups:
            raise RuntimeError(f"DSpark draft cache groups are missing for layers: {layer_names}")

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        del slot_mapping
        self._per_group_block_tables[gid] = block_table

    @staticmethod
    def _block_table_tensor(block_table, batch_size: int) -> torch.Tensor:
        try:
            return block_table.get_device_tensor(batch_size)
        except TypeError:
            return block_table.get_device_tensor()

    def _get_block_table(self, gid: int, cad: CommonAttentionMetadata, batch_size: int) -> torch.Tensor:
        table = self._per_group_block_tables.get(gid)
        input_batch = getattr(self.runner, "input_batch", None)
        runner_tables = getattr(input_batch, "block_table", None)
        if table is None and runner_tables is not None:
            try:
                table = self._block_table_tensor(runner_tables[gid], batch_size)
            except (IndexError, KeyError, TypeError):
                table = None
        if table is None and self.draft_attn_groups[0].kv_cache_group_id == gid:
            table = cad.block_table_tensor
        if table is None:
            raise ValueError(f"DSpark requires a block table for draft cache group {gid}")
        return table[:batch_size]

    def _slot_buffer(self, gid: int, *, context: bool) -> torch.Tensor:
        buffers = self._context_slot_buffers if context else self._query_slot_buffers
        buffer = buffers.get(gid)
        size = self.max_num_tokens if context else self.max_query_tokens
        if buffer is None:
            buffer = torch.zeros(size, dtype=torch.int32, device=self.device)
            buffers[gid] = buffer
        return buffer

    @staticmethod
    def _slots_from_table(
        positions: torch.Tensor,
        req_idx: int,
        block_table: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[req_idx].index_select(0, logical_blocks.long())
        return physical_blocks.to(torch.int32) * block_size + offsets

    def _layer_values(self, values_by_gid: dict[int, torch.Tensor], num_tokens: int | None = None):
        values: dict[str, torch.Tensor] = {}
        for group in self.draft_attn_groups:
            value = values_by_gid[group.kv_cache_group_id]
            for name in group.layer_names:
                values[name] = value
        ordered = []
        for name in self._draft_attn_layer_names_ordered:
            value = values[name]
            ordered.append(value if num_tokens is None else value[:num_tokens])
        return tuple(ordered)

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
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, None]:
        del target_token_ids, req_scheduled_tokens, long_seq_metadata, num_prefill_reqs, num_decode_reqs
        batch_size = cad.num_reqs
        num_reqs_across_dp = getattr(self.runner, "_num_reqs_across_dp", None)
        if num_reqs_across_dp is not None:
            dp_rank = getattr(self, "dp_rank", getattr(self.runner, "dp_rank", 0))
            actual_batch_size = int(num_reqs_across_dp[dp_rank].item())
            if actual_batch_size > batch_size:
                raise ValueError(
                    f"DSpark actual request count exceeds target shape: actual={actual_batch_size}, target={batch_size}"
                )
            batch_size = actual_batch_size
        num_query_tokens = batch_size * self.block_size
        query_start_cpu = getattr(cad, "query_start_loc_cpu", None)
        sample_indices_cpu = None
        effective_rejected_tokens = None
        if token_indices_to_sample is not None:
            sample_indices_cpu = token_indices_to_sample[:batch_size].to(device="cpu", dtype=torch.int32)
            effective_rejected_tokens = (
                cad.query_start_loc[1 : batch_size + 1] - 1 - token_indices_to_sample[:batch_size]
            )
        rejected_cpu = None
        if sample_indices_cpu is None and num_rejected_tokens_gpu is not None:
            rejected_cpu = num_rejected_tokens_gpu[:batch_size].to(device="cpu", dtype=torch.int32)

        self._block_tables_by_gid = {
            group.kv_cache_group_id: self._get_block_table(group.kv_cache_group_id, cad, batch_size)
            for group in self.draft_attn_groups
        }
        self._seed_buffer[:batch_size].copy_(next_token_ids[:batch_size])
        context_cursor = 0
        for req_idx in range(batch_size):
            if query_start_cpu is None:
                start = int(cad.query_start_loc[req_idx].item())
                end = int(cad.query_start_loc[req_idx + 1].item())
            else:
                start = int(query_start_cpu[req_idx])
                end = int(query_start_cpu[req_idx + 1])
            if sample_indices_cpu is not None:
                valid_end = int(sample_indices_cpu[req_idx]) + 1
            else:
                valid_end = end if rejected_cpu is None else end - int(rejected_cpu[req_idx])
            if valid_end <= start:
                gpu_query_range = cad.query_start_loc[req_idx : req_idx + 2].to(device="cpu").tolist()
                rejected = (
                    None
                    if num_rejected_tokens_gpu is None
                    else int(num_rejected_tokens_gpu[req_idx].to(device="cpu").item())
                )
                raise ValueError(
                    "DSpark input must retain a valid anchor token: "
                    f"req={req_idx}, cpu_range=({start}, {end}), "
                    f"gpu_range={gpu_query_range}, anchor_end={valid_end}, "
                    f"rejected={rejected}"
                )
            if valid_end > end:
                raise ValueError(
                    f"DSpark anchor token exceeds request range: req={req_idx}, end={end}, anchor={valid_end}"
                )
            context_end = context_cursor + end - start
            self._dflash_hidden_states[context_cursor:context_end].copy_(target_hidden_states[start:end])
            self._context_positions_buffer[context_cursor:context_end].copy_(target_positions[start:end])
            for group in self.draft_attn_groups:
                gid = group.kv_cache_group_id
                self._slot_buffer(gid, context=True)[context_cursor:context_end] = self._slots_from_table(
                    target_positions[start:end],
                    req_idx,
                    self._block_tables_by_gid[gid],
                    int(group.kv_cache_spec.block_size),
                )
            context_cursor = context_end

            out_start = req_idx * self.block_size
            out_end = out_start + self.block_size
            draft_positions = target_positions[valid_end - 1] + 1 + self.arange_dspark[: self.block_size]
            max_model_len = int(self.max_model_len)
            invalid = draft_positions >= max_model_len
            self.positions[out_start:out_end] = torch.where(invalid, 0, draft_positions)
            self.input_ids[out_start] = next_token_ids[req_idx]
            self.input_ids[out_start + 1 : out_end] = self.parallel_drafting_token_id
            self._token_to_req_buffer[out_start:out_end] = req_idx
            for group in self.draft_attn_groups:
                gid = group.kv_cache_group_id
                slots = self._slots_from_table(
                    self.positions[out_start:out_end],
                    req_idx,
                    self._block_tables_by_gid[gid],
                    int(group.kv_cache_spec.block_size),
                )
                slots.masked_fill_(invalid, -1)
                self._slot_buffer(gid, context=False)[out_start:out_end] = slots

        self._dflash_num_context = context_cursor
        self._context_slots_by_gid = {
            gid: self._slot_buffer(gid, context=True)[:context_cursor] for gid in self._block_tables_by_gid
        }
        self._query_slots_by_gid = {
            gid: self._slot_buffer(gid, context=False)[:num_query_tokens] for gid in self._block_tables_by_gid
        }
        cad.query_start_loc = self.arange_dspark[: batch_size + 1] * self.block_size
        cad.query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * self.block_size
        effective_seq_lens = cad.seq_lens[:batch_size]
        if effective_rejected_tokens is not None:
            effective_seq_lens = effective_seq_lens - effective_rejected_tokens
        elif num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu[:batch_size]
        cad.seq_lens = (effective_seq_lens + self.block_size).clamp(max=self.max_model_len)
        cad.num_reqs = batch_size
        cad.num_actual_tokens = num_query_tokens
        cad.num_input_tokens = num_query_tokens
        cad.max_query_len = self.block_size
        cad.max_seq_len = min(cad.max_seq_len + self.block_size, self.max_model_len)
        cad.positions = self.positions[:num_query_tokens]
        cad.slot_mapping = self._query_slots_by_gid[self.draft_attn_groups[0].kv_cache_group_id]
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        cad.token_to_req_indices = self._token_to_req_buffer[:num_query_tokens]
        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.block_size] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.block_size
        return num_query_tokens, torch.arange(num_query_tokens, dtype=torch.int32, device=self.device), cad, None

    def _build_attn_metadata(self, cad: CommonAttentionMetadata) -> list[dict[str, Any]]:
        per_layer: dict[str, Any] = {}
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            group_cad = copy(cad)
            group_cad.block_table_tensor = self._block_tables_by_gid[gid]
            group_cad.slot_mapping = self._query_slots_by_gid[gid]
            metadata = group.get_metadata_builder().build_for_drafting(
                group_cad,
                draft_index=1,
                block_size=group.kv_cache_spec.block_size,
            )
            for name in group.layer_names:
                per_layer[name] = metadata
        return [per_layer]

    def _precompute_context_kv(self) -> None:
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[: self._dflash_num_context],
            self._context_positions_buffer[: self._dflash_num_context],
            self._layer_values(self._context_slots_by_gid, self._dflash_num_context),
        )

    def _sampling_temperature(
        self,
        metadata: SamplingMetadata,
        num_reqs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if metadata.temperature is None:
            default = 0.0 if metadata.all_greedy else 1.0
            return torch.full((num_reqs,), default, dtype=torch.float32, device=device)
        return metadata.temperature[:num_reqs].to(device=device, dtype=torch.float32).contiguous()

    def _sampling_seeds(self, metadata: SamplingMetadata, num_reqs: int, device: torch.device) -> torch.Tensor:
        sampler = getattr(getattr(self.runner, "sampler", None), "sampling_states", None)
        seeds = getattr(getattr(sampler, "seeds", None), "gpu", None)
        idx_mapping = getattr(getattr(self.runner, "input_batch", None), "idx_mapping", None)
        if isinstance(seeds, torch.Tensor):
            if isinstance(idx_mapping, torch.Tensor):
                seeds = seeds.index_select(0, idx_mapping[:num_reqs].to(device=seeds.device, dtype=torch.long))
            return seeds[:num_reqs].to(device=device, dtype=torch.int64).contiguous()
        out = self._sampling_seed_buffer[:num_reqs]
        base_seed = int(getattr(self.vllm_config.model_config, "seed", 0) or 0)
        for req_idx in range(num_reqs):
            generator = metadata.generators.get(req_idx)
            out[req_idx] = generator.initial_seed() if generator is not None else base_seed + req_idx * 9973
        return out

    def _sample_sequential(
        self,
        hidden_states: torch.Tensor,
        num_reqs: int,
        metadata: SamplingMetadata,
    ) -> torch.Tensor:
        base_logits = self.model.compute_logits(hidden_states).view(num_reqs, self.block_size, -1)
        probabilistic = self._probabilistic and not metadata.all_greedy
        draft_logits = torch.empty_like(base_logits, dtype=torch.float32) if probabilistic else None
        prev_ids = self._seed_buffer[:num_reqs]
        gumbel_positions = self.positions[: num_reqs * self.block_size].view(num_reqs, self.block_size) - 1
        idx_mapping = None
        temperatures = None
        seeds = None
        if probabilistic:
            idx_mapping = getattr(getattr(self.runner, "input_batch", None), "idx_mapping", None)
            if not isinstance(idx_mapping, torch.Tensor):
                idx_mapping = self._idx_mapping_buffer
            idx_mapping = idx_mapping[:num_reqs].to(device=base_logits.device, dtype=torch.int32).contiguous()
            temperatures = self._sampling_temperature(metadata, num_reqs, base_logits.device)
            seeds = self._sampling_seeds(metadata, num_reqs, base_logits.device)
        for step in range(self.block_size):
            logits = base_logits[:, step] + self.model.markov_bias(self.model.markov_embed(prev_ids))
            if probabilistic:
                assert draft_logits is not None
                assert idx_mapping is not None
                assert temperatures is not None
                assert seeds is not None
                draft_ids = gumbel_sample(
                    logits.contiguous(),
                    idx_mapping,
                    temperatures,
                    seeds,
                    gumbel_positions[:, step].to(dtype=torch.int32).contiguous(),
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=self.arange_dspark[step],
                    use_fp64=getattr(self, "use_fp64_gumbel", False),
                )
            else:
                draft_ids = _greedy_sample(logits)
            self._draft_buffer[:num_reqs, step].copy_(draft_ids)
            prev_ids = self._draft_buffer[:num_reqs, step]
        self._last_draft_logits = None if draft_logits is None else draft_logits.contiguous()
        return self._draft_buffer[:num_reqs]

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
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            target_model_batch_desc,
            mm_embed_inputs,
            req_scheduled_tokens,
            long_seq_metadata,
            num_prefill_reqs,
            num_decode_reqs,
            scheduler_output,
            num_scheduled_tokens,
        )
        num_tokens, _, cad, _ = self.set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )
        assert self.runner is not None
        num_tokens, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(
            num_tokens,
            is_draft_model=True,
        )
        if num_tokens != cad.num_actual_tokens:
            raise RuntimeError("Eager DSpark does not support DP token padding")
        attn_metadata = self._build_attn_metadata(cad)
        self._precompute_context_kv()
        with set_ascend_forward_context(
            attn_metadata[0],
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            is_draft_model=True,
            draft_attn_metadatas=attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            hidden_states = self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                inputs_embeds=None,
            )
            return self._sample_sequential(hidden_states, cad.num_reqs, sampling_metadata)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        is_profile=False,
        **kwargs,
    ) -> None:
        del num_tokens_across_dp, aclgraph_runtime_mode, batch_descriptor, is_profile, kwargs
        if not self.draft_attn_groups or num_reqs == 0:
            return
        num_query_tokens = min(num_reqs * self.block_size, num_tokens, self.max_query_tokens)
        if num_query_tokens == 0:
            return
        self.input_ids[:num_query_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[:num_query_tokens].zero_()
