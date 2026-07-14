# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from copy import copy
from dataclasses import replace
from typing import Any

import torch
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
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

        target_has_full_graph = vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        base_use_cuda_graph = bool(getattr(self, "use_cuda_graph", False))
        self._draft_aclgraph_enabled = base_use_cuda_graph and target_has_full_graph
        if base_use_cuda_graph and not target_has_full_graph:
            logger.warning_once("DSpark drafter ACLGraph requires FULL graph mode; using the eager drafter")
        self.use_cuda_graph = self._draft_aclgraph_enabled
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
        self._block_table_buffers: dict[int, torch.Tensor] = {}
        self._stable_attn_buffers: dict[tuple[str, tuple[int, ...], torch.dtype, str], torch.Tensor] = {}
        self._query_start_loc_buffer = torch.zeros(self.max_batch_size + 1, dtype=torch.int32, device=device)
        self._query_start_loc_cpu_buffer = torch.zeros(self.max_batch_size + 1, dtype=torch.int32)
        self._query_start_loc_cpu_base = torch.arange(self.max_batch_size + 1, dtype=torch.int32) * self.block_size
        self._seq_lens_buffer = torch.zeros(self.max_batch_size, dtype=torch.int32, device=device)
        self._seq_lens_cpu_buffer = torch.zeros(self.max_batch_size, dtype=torch.int32)
        self._draft_capture_sizes: list[int] = []
        self._graph_model_inputs: dict[str, Any] | None = None
        self._last_draft_logits: torch.Tensor | None = None
        self._probabilistic = vllm_config.speculative_config.draft_sample_method == "probabilistic"
        self.arange_dspark = torch.arange(
            self.max_num_tokens + self.max_query_tokens + 1,
            dtype=torch.int32,
            device=device,
        )

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        dispatcher_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if self.use_cuda_graph and cudagraph_mode.has_full_cudagraphs()
            else CUDAGraphMode.NONE
        )
        compilation_config = copy(self.vllm_config.compilation_config)
        if compilation_config.cudagraph_capture_sizes is not None:
            compilation_config.cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes)
        if dispatcher_mode != CUDAGraphMode.NONE:
            capture_sizes = self._derive_draft_capture_sizes()
            if capture_sizes:
                compilation_config.cudagraph_capture_sizes = capture_sizes
                compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
            else:
                compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                    self.block_size,
                    self.vllm_config.parallel_config.tensor_parallel_size,
                )
        draft_config = replace(self.vllm_config, compilation_config=compilation_config)
        self.cudagraph_dispatcher = CudagraphDispatcher(draft_config)
        self.cudagraph_dispatcher.uniform_decode_query_len = self.block_size
        self.cudagraph_dispatcher.initialize_cudagraph_keys(dispatcher_mode, self.block_size)
        self._draft_capture_sizes = sorted(
            {desc.num_tokens for _, descs in self.cudagraph_dispatcher.get_capture_descs() for desc in descs}
        )

    def _derive_draft_capture_sizes(self) -> list[int]:
        dispatcher = getattr(self.runner, "cudagraph_dispatcher", None)
        if dispatcher is None:
            return []
        sizes = {
            int(desc.num_reqs) * self.block_size
            for mode, descs in dispatcher.get_capture_descs()
            if mode == CUDAGraphMode.FULL
            for desc in descs
            if desc.uniform and desc.num_reqs is not None
        }
        return sorted(size for size in sizes if size > 0)

    def get_cudagraph_capture_sizes(self) -> list[int]:
        return list(self._draft_capture_sizes)

    def load_model(self, model: torch.nn.Module) -> None:
        graph_enabled = self._draft_aclgraph_enabled
        self.use_cuda_graph = False
        try:
            super().load_model(model)
        finally:
            self.use_cuda_graph = graph_enabled

        if graph_enabled:
            self.update_stream = torch.npu.Stream()
            self._runnable = ACLGraphWrapper(
                self._run_model_from_graph_buffers,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )
            logger.info_once("DSpark drafter ACLGraph is enabled; speculative_config.enforce_eager=true disables it")
        else:
            self._runnable = self._run_model

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_config = super()._create_draft_vllm_config()
        model_config = copy(draft_config.model_config)
        compilation_config = copy(draft_config.compilation_config)
        if self._draft_aclgraph_enabled:
            model_config.enforce_eager = True
            compilation_config.mode = CompilationMode.NONE
        return replace(
            draft_config,
            model_config=model_config,
            compilation_config=compilation_config,
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
        table = table[:batch_size]
        if not getattr(self, "use_cuda_graph", False):
            return table
        buffer = self._block_table_buffers.get(gid)
        required_shape = (self.max_batch_size, *table.shape[1:])
        if buffer is None:
            buffer = torch.zeros(required_shape, dtype=table.dtype, device=table.device)
            self._block_table_buffers[gid] = buffer
        elif buffer.shape != required_shape or buffer.dtype != table.dtype or buffer.device != table.device:
            raise RuntimeError(f"DSpark block table shape changed after graph initialization for cache group {gid}")
        buffer[:batch_size].zero_()
        buffer[:batch_size].copy_(table)
        return buffer[:batch_size]

    def _ensure_query_metadata_buffers(self, num_reqs: int) -> None:
        capacity = max(int(getattr(self, "max_batch_size", 0)), num_reqs)
        device = getattr(self, "device", None)
        if device is None:
            device = self.positions.device
        if not hasattr(self, "_query_start_loc_buffer") or self._query_start_loc_buffer.numel() < capacity + 1:
            self._query_start_loc_buffer = torch.zeros(capacity + 1, dtype=torch.int32, device=device)
            self._query_start_loc_cpu_buffer = torch.zeros(capacity + 1, dtype=torch.int32)
            self._query_start_loc_cpu_base = torch.arange(capacity + 1, dtype=torch.int32) * self.block_size
        if not hasattr(self, "_seq_lens_buffer") or self._seq_lens_buffer.numel() < capacity:
            self._seq_lens_buffer = torch.zeros(capacity, dtype=torch.int32, device=device)
            self._seq_lens_cpu_buffer = torch.zeros(capacity, dtype=torch.int32)

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
        self._ensure_query_metadata_buffers(batch_size)
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
                    f"gpu_range={gpu_query_range}, anchor_end={valid_end}, rejected={rejected}"
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
        self._query_start_loc_buffer[: batch_size + 1].copy_(self.arange_dspark[: batch_size + 1] * self.block_size)
        self._query_start_loc_cpu_buffer[: batch_size + 1].copy_(self._query_start_loc_cpu_base[: batch_size + 1])
        cad.query_start_loc = self._query_start_loc_buffer[: batch_size + 1]
        cad.query_start_loc_cpu = self._query_start_loc_cpu_buffer[: batch_size + 1]
        effective_seq_lens = cad.seq_lens[:batch_size]
        if effective_rejected_tokens is not None:
            effective_seq_lens = effective_seq_lens - effective_rejected_tokens
        elif num_rejected_tokens_gpu is not None:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu[:batch_size]
        self._seq_lens_buffer[:batch_size].copy_((effective_seq_lens + self.block_size).clamp(max=self.max_model_len))
        self._seq_lens_cpu_buffer[:batch_size].copy_(self._seq_lens_buffer[:batch_size].to(device="cpu"))
        cad.seq_lens = self._seq_lens_buffer[:batch_size]
        cad._seq_lens_cpu = self._seq_lens_cpu_buffer[:batch_size]
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

    def _pad_request_rows(
        self,
        cad: CommonAttentionMetadata,
        actual_num_reqs: int,
        model_num_reqs: int,
    ) -> int:
        if model_num_reqs < actual_num_reqs or model_num_reqs > self.max_batch_size:
            raise ValueError(f"Invalid DSpark padded request count: actual={actual_num_reqs}, model={model_num_reqs}")
        actual_tokens = actual_num_reqs * self.block_size
        model_tokens = model_num_reqs * self.block_size
        if model_tokens > actual_tokens:
            self.input_ids[actual_tokens:model_tokens].fill_(self.parallel_drafting_token_id)
            self.positions[actual_tokens:model_tokens].zero_()
            self._token_to_req_buffer[actual_tokens:model_tokens].copy_(
                self.arange_dspark[actual_num_reqs:model_num_reqs].repeat_interleave(self.block_size)
            )
            self._seed_buffer[actual_num_reqs:model_num_reqs].zero_()
            for gid in self._query_slots_by_gid:
                self._slot_buffer(gid, context=False)[actual_tokens:model_tokens].fill_(-1)

        self._query_start_loc_buffer[: model_num_reqs + 1].copy_(
            self.arange_dspark[: model_num_reqs + 1] * self.block_size
        )
        self._query_start_loc_cpu_buffer[: model_num_reqs + 1].copy_(
            self._query_start_loc_cpu_base[: model_num_reqs + 1]
        )
        self._seq_lens_buffer[actual_num_reqs:model_num_reqs].fill_(self.block_size)
        self._seq_lens_cpu_buffer[actual_num_reqs:model_num_reqs].fill_(self.block_size)
        for gid, table in tuple(self._block_tables_by_gid.items()):
            buffer = self._block_table_buffers.get(gid)
            if buffer is not None:
                buffer[actual_num_reqs:model_num_reqs].zero_()
                self._block_tables_by_gid[gid] = buffer[:model_num_reqs]
            elif model_num_reqs != actual_num_reqs:
                padded = table.new_zeros((model_num_reqs, *table.shape[1:]))
                padded[:actual_num_reqs].copy_(table[:actual_num_reqs])
                self._block_tables_by_gid[gid] = padded
        self._query_slots_by_gid = {
            gid: self._slot_buffer(gid, context=False)[:model_tokens] for gid in self._block_tables_by_gid
        }

        cad.num_reqs = model_num_reqs
        cad.num_input_tokens = model_tokens
        cad.num_actual_tokens = model_tokens
        cad.query_start_loc = self._query_start_loc_buffer[: model_num_reqs + 1]
        cad.query_start_loc_cpu = self._query_start_loc_cpu_buffer[: model_num_reqs + 1]
        cad.seq_lens = self._seq_lens_buffer[:model_num_reqs]
        cad._seq_lens_cpu = self._seq_lens_cpu_buffer[:model_num_reqs]
        cad.positions = self.positions[:model_tokens]
        cad.token_to_req_indices = self._token_to_req_buffer[:model_tokens]
        cad.slot_mapping = self._query_slots_by_gid[self.draft_attn_groups[0].kv_cache_group_id]
        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.block_size] * model_num_reqs
        return model_tokens

    def _stable_attn_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        buffer_key = (key, tuple(tensor.shape), tensor.dtype, str(tensor.device))
        buffer = self._stable_attn_buffers.get(buffer_key)
        if buffer is None:
            buffer = torch.empty_like(tensor)
            self._stable_attn_buffers[buffer_key] = buffer
        buffer.copy_(tensor)
        return buffer

    def _stabilize_attn_metadata(self, per_layer: dict[str, Any]) -> None:
        attrs = (
            "block_table",
            "cos",
            "dspark_swa_indices",
            "dspark_swa_lens",
            "full_compress_cos",
            "full_compress_sin",
            "input_positions",
            "qli_metadata",
            "query_start_loc",
            "query_start_loc_cpu",
            "sas_metadata",
            "seq_lens",
            "sin",
            "slot_mapping",
            "start_pos",
        )
        seen: set[int] = set()

        def stabilize(obj: Any, prefix: str) -> None:
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            for attr in attrs:
                value = getattr(obj, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(obj, attr, self._stable_attn_tensor(f"{prefix}.{attr}", value))
            for attr in ("cp_metadata", "decode", "prefill", "req_metadata"):
                stabilize(getattr(obj, attr, None), f"{prefix}.{attr}")

        for layer_name, metadata in per_layer.items():
            stabilize(metadata, layer_name)

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
        if self.use_cuda_graph:
            self._stabilize_attn_metadata(per_layer)
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
        actual_num_reqs = cad.num_reqs
        num_reqs_across_dp = self.runner._num_reqs_across_dp
        model_num_reqs = int(num_reqs_across_dp.max().item()) if num_reqs_across_dp is not None else actual_num_reqs
        num_input_tokens = model_num_reqs * self.block_size
        has_lora = bool(getattr(self.runner.input_batch, "lora_id_to_lora_request", {}))
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens,
                uniform_decode=True,
                has_lora=has_lora,
            )
            num_input_tokens = batch_descriptor.num_tokens
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        num_tokens_across_dp = (
            torch.full_like(num_reqs_across_dp, num_input_tokens) if num_reqs_across_dp is not None else None
        )
        if num_input_tokens % self.block_size:
            raise RuntimeError("DSpark DP padding must preserve complete draft request rows")
        model_num_reqs = num_input_tokens // self.block_size
        num_input_tokens = self._pad_request_rows(cad, actual_num_reqs, model_num_reqs)
        attn_metadata = self._build_attn_metadata(cad)
        self._precompute_context_kv()
        with set_ascend_forward_context(
            attn_metadata[0],
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing and self.draft_attn_groups:
                self._update_full_graph_params(forward_context, num_input_tokens, attn_metadata)
            model_inputs = dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                inputs_embeds=None,
            )
            if self.use_cuda_graph:
                self._graph_model_inputs = model_inputs
                hidden_states = self._runnable()
            else:
                hidden_states = self._runnable(**model_inputs)
            hidden_states = hidden_states[:num_tokens]
            return self._sample_sequential(hidden_states, actual_num_reqs, sampling_metadata)

    def _run_model(self, **model_inputs: Any) -> torch.Tensor:
        return self.model(**model_inputs)

    def _run_model_from_graph_buffers(self) -> torch.Tensor:
        if self._graph_model_inputs is None:
            raise RuntimeError("DSpark ACLGraph inputs were not prepared")
        return self._run_model(**self._graph_model_inputs)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        is_profile=False,
        is_graph_capturing=False,
        **kwargs,
    ) -> None:
        del batch_descriptor, kwargs
        if not self.draft_attn_groups or num_reqs == 0:
            return
        num_reqs_across_dp = getattr(self.runner, "_num_reqs_across_dp", None)
        if not is_profile and not is_graph_capturing and num_reqs_across_dp is not None:
            num_reqs = max(num_reqs, int(num_reqs_across_dp.max().item()))
        num_query_tokens = min(num_reqs * self.block_size, self.max_query_tokens)
        if num_query_tokens == 0 or num_query_tokens % self.block_size:
            return
        has_lora = bool(getattr(self.runner.input_batch, "lora_id_to_lora_request", {}))
        use_graph = self.use_cuda_graph and aclgraph_runtime_mode == CUDAGraphMode.FULL
        if use_graph:
            aclgraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_query_tokens,
                uniform_decode=True,
                has_lora=has_lora,
                valid_modes={CUDAGraphMode.FULL},
            )
            num_query_tokens = batch_descriptor.num_tokens
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        draft_tokens_across_dp = None
        if num_tokens_across_dp is not None:
            draft_tokens_across_dp = torch.full_like(num_tokens_across_dp, num_query_tokens)
        if num_query_tokens % self.block_size:
            raise RuntimeError("DSpark graph capture size must contain complete request rows")
        model_num_reqs = num_query_tokens // self.block_size
        self.input_ids[:num_query_tokens].fill_(self.parallel_drafting_token_id)
        self.positions[:num_query_tokens].zero_()
        self._token_to_req_buffer[:num_query_tokens].copy_(
            torch.arange(model_num_reqs, dtype=torch.int32, device=self.device).repeat_interleave(self.block_size)
        )
        self._query_start_loc_buffer[: model_num_reqs + 1].copy_(
            self.arange_dspark[: model_num_reqs + 1] * self.block_size
        )
        self._query_start_loc_cpu_buffer[: model_num_reqs + 1].copy_(
            self._query_start_loc_cpu_base[: model_num_reqs + 1]
        )
        self._seq_lens_buffer[:model_num_reqs].fill_(self.block_size)
        self._seq_lens_cpu_buffer[:model_num_reqs].fill_(self.block_size)
        cad = AscendCommonAttentionMetadata(
            query_start_loc=self._query_start_loc_buffer[: model_num_reqs + 1],
            query_start_loc_cpu=self._query_start_loc_cpu_buffer[: model_num_reqs + 1],
            seq_lens=self._seq_lens_buffer[:model_num_reqs],
            _seq_lens_cpu=self._seq_lens_cpu_buffer[:model_num_reqs],
            seq_lens_cpu=None,
            num_computed_tokens_cpu=None,
            num_reqs=model_num_reqs,
            num_actual_tokens=num_query_tokens,
            num_input_tokens=num_query_tokens,
            max_query_len=self.block_size,
            actual_seq_lengths_q=[self.block_size] * model_num_reqs,
            block_table_tensor=None,
            slot_mapping=self._slot_buffer(self.draft_attn_groups[0].kv_cache_group_id, context=False)[
                :num_query_tokens
            ],
            positions=self.positions[:num_query_tokens],
            attn_state=AscendAttentionState.ChunkedPrefill,
            decode_token_per_req=self.block_size,
            max_seq_len=self.block_size,
            causal=False,
        )
        cad.token_to_req_indices = self._token_to_req_buffer[:num_query_tokens]
        self._block_tables_by_gid = {
            group.kv_cache_group_id: self._get_block_table(group.kv_cache_group_id, cad, model_num_reqs)
            for group in self.draft_attn_groups
        }
        self._query_slots_by_gid = {}
        for gid in self._block_tables_by_gid:
            slots = self._slot_buffer(gid, context=False)
            slots[:num_query_tokens].fill_(-1)
            self._query_slots_by_gid[gid] = slots[:num_query_tokens]
        cad.block_table_tensor = self._block_tables_by_gid[self.draft_attn_groups[0].kv_cache_group_id]
        cad.slot_mapping = self._query_slots_by_gid[self.draft_attn_groups[0].kv_cache_group_id]
        attn_metadata = self._build_attn_metadata(cad)
        with set_ascend_forward_context(
            attn_metadata[0],
            self.vllm_config,
            num_tokens=num_query_tokens,
            num_tokens_across_dp=draft_tokens_across_dp,
            num_actual_tokens=num_query_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=attn_metadata,
        ):
            forward_context = get_forward_context()
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing and self.draft_attn_groups:
                self._update_full_graph_params(forward_context, num_query_tokens, attn_metadata)
            model_inputs = dict(
                input_ids=self.input_ids[:num_query_tokens],
                positions=self.positions[:num_query_tokens],
                inputs_embeds=None,
            )
            if self.use_cuda_graph:
                self._graph_model_inputs = model_inputs
                self._runnable()
            else:
                self._runnable(**model_inputs)
