# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.attn_utils import (
    get_shared_kv_cache_layers,
    init_attn_backend,
)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.utils import bind_kv_cache

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.states import Ascend310PRequestState
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310V2(NPUModelRunner):
    """Model runner v2 for Ascend 310P."""

    # TODO: Refactor Triton-dependent overrides to register 310P
    # implementations through Triton Dispatcher after vLLM RFC #45133 lands.
    request_state_cls = Ascend310PRequestState

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int32, device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64, device="cpu")
        self.next_prefill_tokens_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device="cpu")

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        # TODO: Support multimodal and hybrid models in the next 310P MRV2 iteration.
        if model_config.is_multimodal_model or model_config.is_hybrid:
            raise NotImplementedError("Multimodal and hybrid models are not supported by model runner v2 on 310P.")
        if model_config.use_mla:
            raise NotImplementedError("MLA is not supported by model runner v2 on 310P.")
        # TODO: Support multi-dimensional RoPE in the next 310P MRV2 iteration.
        if getattr(model_config, "uses_mrope", False):
            raise NotImplementedError("Multi-dimensional RoPE is not supported by model runner v2 on 310P.")
        if getattr(model_config, "enable_sleep_mode", False):
            raise NotImplementedError("Sleep mode is not supported by model runner v2 on 310P.")

        parallel_config = vllm_config.parallel_config
        # TODO: Restore MRV1 data parallel support in the next 310P MRV2 iteration.
        # Pipeline and context parallelism remain unsupported on 310P.
        unsupported_parallel = {
            "pipeline_parallel_size": getattr(parallel_config, "pipeline_parallel_size", 1),
            "data_parallel_size": getattr(parallel_config, "data_parallel_size", 1),
            "decode_context_parallel_size": getattr(parallel_config, "decode_context_parallel_size", 1),
            "prefill_context_parallel_size": getattr(parallel_config, "prefill_context_parallel_size", 1),
        }
        enabled = [name for name, size in unsupported_parallel.items() if size != 1]
        if enabled:
            raise NotImplementedError(
                f"310P model runner v2 only supports tensor parallelism; unsupported settings: {', '.join(enabled)}."
            )
        if getattr(parallel_config, "enable_expert_parallel", False):
            raise NotImplementedError("Expert parallelism is not supported by model runner v2 on 310P.")
        # TODO: Support speculative decoding in the next 310P MRV2 iteration.
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("Speculative decoding is not supported by model runner v2 on 310P.")
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("KV cache transfer is not supported by model runner v2 on 310P.")
        # TODO: Support prefix caching in the next 310P MRV2 iteration.
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("Prefix caching is not supported by model runner v2 on 310P.")
        # TODO: Support LoRA in the next 310P MRV2 iteration.
        if vllm_config.lora_config is not None:
            raise NotImplementedError("LoRA is not supported by model runner v2 on 310P.")

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        super().finish_requests(scheduler_output)
        if scheduler_output.finished_req_ids:
            # A freed request slot can be reused and its CPU-owned block table
            # rewritten in this step. Drain the previous ACLGraph replay before
            # the new layout is gathered and copied to attention metadata.
            torch.npu.current_stream().synchronize()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate the 310P attention cache as separate K/V NZ tensors."""
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            max_num_blocks = cdiv(self.max_model_len, spec.block_size)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            kv_cache_config, self.vllm_config, self.device
        )
        self._adjust_kernel_block_sizes(kv_cache_config)
        self.block_tables = Ascend310PBlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )

        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=kv_cache_config,
            max_num_reqs=self.max_num_reqs,
        )
        self.cudagraph_manager = ModelAclGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.decode_query_len,
            self,
            lora_capture_cases=self.lora_capture_cases,
        )
        check_attention_cp_compatibility(self.vllm_config)

        shared_layers = get_shared_kv_cache_layers(self.vllm_config)
        kv_caches_dict = self._allocate_kv_cache_tensors(kv_cache_config, shared_layers)
        self.kv_caches: list[Any] = []
        bind_kv_cache(
            kv_caches_dict,
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _adjust_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> None:
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                specs = tuple(group_spec.kv_cache_specs.values())
            else:
                specs = (group_spec,)
            attention_specs = [spec for spec in specs if isinstance(spec, AttentionSpec)]
            if len(attention_specs) != len(specs):
                # TODO: Support non-attention KV cache specs in the next 310P MRV2 iteration.
                raise NotImplementedError("Non-attention KV cache specs are not supported by model runner v2 on 310P.")
            max_head_size = max(spec.head_size for spec in attention_specs)
            if max_head_size > 256:
                raise NotImplementedError(f"310P paged attention requires head_size <= 256, got {max_head_size}.")
            backend = self.attn_groups[group_id][0].backend
            supported_sizes = [
                block_size
                for block_size in backend.get_supported_kernel_block_sizes()
                if block_size * max_head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
            ]
            if not supported_sizes:
                raise NotImplementedError(
                    f"310P paged attention requires block_size * head_size <= {_ATTENTION_BLOCK_SIZE_LIMIT}."
                )
            self.kernel_block_sizes[group_id] = supported_sizes[0]

    def _allocate_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        shared_layers: dict[str, str],
    ) -> dict[str, Any]:
        layer_specs: dict[str, KVCacheSpec] = {}
        layer_group_ids: dict[str, int] = {}
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_specs[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_specs[layer_name] = group_spec
                layer_group_ids[layer_name] = group_id

        layer_backends = {
            layer_name: group.backend
            for groups in self.attn_groups
            for group in groups
            for layer_name in group.layer_names
        }
        kv_caches: dict[str, Any] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            layer_names = [name for name in kv_cache_tensor.shared_by if name not in shared_layers]
            if not layer_names:
                continue
            cache_groups: dict[tuple[Any, ...], list[str]] = {}
            for layer_name in layer_names:
                spec = layer_specs[layer_name]
                if not isinstance(spec, AttentionSpec):
                    raise NotImplementedError(f"Unsupported 310P KV cache spec: {type(spec).__name__}.")
                backend = layer_backends[layer_name]
                group_id = layer_group_ids[layer_name]
                storage_block_size = getattr(spec, "storage_block_size", spec.block_size)
                kernel_block_size = (
                    storage_block_size if storage_block_size != spec.block_size else self.kernel_block_sizes[group_id]
                )
                cache_groups.setdefault((spec, backend, kernel_block_size), []).append(layer_name)

            for (spec, backend, kernel_block_size), cache_layer_names in cache_groups.items():
                if not issubclass(backend, AscendAttentionBackend310):
                    raise TypeError(f"310P selected unexpected attention backend {backend}.")
                if kv_cache_tensor.size % spec.page_size_bytes != 0:
                    raise ValueError("KV cache allocation is not page aligned.")
                num_blocks = kv_cache_tensor.size // spec.page_size_bytes
                if num_blocks < kv_cache_config.num_blocks:
                    raise ValueError("KV cache allocation contains fewer blocks than requested.")
                blocks_per_kv_block = spec.block_size // kernel_block_size
                kv_cache_shape = backend.get_kv_cache_shape(
                    num_blocks * blocks_per_kv_block,
                    kernel_block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    self.cache_config.cache_dtype,
                )
                if getattr(spec, "head_size_v", spec.head_size) != spec.head_size:
                    raise NotImplementedError("310P MRV2 does not support asymmetric K/V head sizes.")
                cache_shape = kv_cache_shape[1:]
                k_cache = torch_npu.empty_with_format(
                    size=cache_shape,
                    dtype=spec.dtype,
                    device=self.device,
                    acl_format=ACL_FORMAT_FRACTAL_NZ,
                )
                v_cache = torch_npu.empty_with_format(
                    size=cache_shape,
                    dtype=spec.dtype,
                    device=self.device,
                    acl_format=ACL_FORMAT_FRACTAL_NZ,
                )
                for layer_name in cache_layer_names:
                    kv_caches[layer_name] = (k_cache, v_cache)

        for layer_name, target_layer_name in shared_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]
        expected_layers = {
            layer_name
            for kv_cache_group in kv_cache_config.kv_cache_groups
            for layer_name in kv_cache_group.layer_names
        }
        if expected_layers != set(kv_caches):
            raise RuntimeError("Some 310P KV cache layers were not initialized.")
        return kv_caches

    def _prepare_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        *,
        idx_mapping_np: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> None:
        # TODO: Refactor this CPU fallback to use Triton Dispatcher after vLLM
        # RFC #45133 lands.
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
                self.next_prefill_tokens_cpu[req_idx] = self.req_states.all_token_ids.cpu[req_idx, next_position]
        input_ids.copy_(self.input_ids_cpu[: input_ids.shape[0]], non_blocking=True)
        next_prefill_tokens.copy_(self.next_prefill_tokens_cpu, non_blocking=True)

    def _prepare_pos_seq_lens(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        idx_mapping_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        # TODO: Refactor this CPU fallback to use Triton Dispatcher after vLLM
        # RFC #45133 lands.
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
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
        num_bonus_tokens: int,
        *,
        idx_mapping_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        seq_lens_np: np.ndarray,
        prefill_len_np: np.ndarray,
    ) -> torch.Tensor:
        # TODO: Refactor this CPU fallback to use Triton Dispatcher after vLLM
        # RFC #45133 lands.
        del idx_mapping, query_start_loc, seq_lens, prefill_len
        del draft_tokens, cu_num_logits, num_bonus_tokens
        if num_logits != len(idx_mapping_np):
            # TODO: Support draft tokens in the next 310P MRV2 iteration.
            raise NotImplementedError("310P MRV2 does not support draft tokens.")
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
        # TODO: Refactor block-table preparation to use Triton Dispatcher after
        # vLLM RFC #45133 lands.
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping_np,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        positions_np = np.zeros(input_batch.num_tokens_after_padding, dtype=np.int64)
        for batch_idx, (start_position, num_scheduled_tokens) in enumerate(
            zip(input_batch.num_computed_tokens_np, input_batch.num_scheduled_tokens)
        ):
            start = int(input_batch.query_start_loc_np[batch_idx])
            end = start + int(num_scheduled_tokens)
            positions_np[start:end] = np.arange(
                start_position,
                start_position + num_scheduled_tokens,
                dtype=np.int64,
            )
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
        # TODO: Refactor 310P sampling to use Triton Dispatcher after vLLM RFC
        # #45133 lands.
        if grammar_output is not None:
            # TODO: Restore MRV1 structured output support in the next 310P MRV2 iteration.
            raise NotImplementedError("Structured output is not supported by model runner v2 on 310P.")
        logits = self.model.compute_logits(hidden_states[input_batch.logits_indices])
        sampler_output = self.sampler(logits, input_batch)
        can_sample_np = input_batch.seq_lens_np[: input_batch.num_reqs] >= input_batch.prefill_len_np
        num_sampled = async_copy_to_gpu(can_sample_np.astype(np.int32), device=self.device)
        num_rejected = torch.zeros_like(num_sampled)
        sampler_output.num_sampled = num_sampled
        sampler_output.num_rejected = num_rejected
        return sampler_output, num_sampled, num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # TODO: Refactor this 310P state update to use Triton Dispatcher after
        # vLLM RFC #45133 lands.
        del num_rejected
        num_entries = min(idx_mapping.shape[0], sampled_tokens.shape[0], num_sampled.shape[0])
        idx_mapping = idx_mapping[:num_entries]
        sampled_tokens = sampled_tokens[:num_entries]
        num_sampled = num_sampled[:num_entries]
        valid_mask = idx_mapping >= 0
        valid_indices = idx_mapping.masked_select(valid_mask)
        sampled = sampled_tokens[:, 0].masked_select(valid_mask).to(self.req_states.last_sampled_tokens.dtype)
        valid_num_sampled = num_sampled.masked_select(valid_mask)
        has_sample = valid_num_sampled > 0

        token_positions = self.req_states.total_len.gpu[valid_indices].to(torch.int64)
        old_tokens = self.req_states.all_token_ids.gpu[valid_indices, token_positions]
        stored_tokens = torch.where(has_sample, sampled.to(torch.int32), old_tokens)
        self.req_states.all_token_ids.gpu.index_put_((valid_indices, token_positions), stored_tokens)
        old_last = self.req_states.last_sampled_tokens[valid_indices, 0]
        self.req_states.last_sampled_tokens.index_copy_(
            0,
            valid_indices,
            torch.where(has_sample, sampled, old_last).unsqueeze(-1),
        )
        self.req_states.total_len.gpu.index_add_(0, valid_indices, valid_num_sampled)

        if query_start_loc is not None:
            query_lens = self._get_valid_query_lens(idx_mapping, query_start_loc)
            self.req_states.num_computed_tokens.gpu.index_add_(
                0,
                valid_indices,
                query_lens.to(self.req_states.num_computed_tokens.gpu.dtype),
            )
        self.model_state.postprocess_state(idx_mapping, num_sampled)

    @staticmethod
    def _get_valid_query_lens(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Return real request query lengths without ACLGraph padding."""
        num_query_lens = min(idx_mapping.shape[0], query_start_loc.shape[0] - 1)
        query_lens = query_start_loc[1 : num_query_lens + 1] - query_start_loc[:num_query_lens]
        return query_lens.masked_select(idx_mapping[:num_query_lens] >= 0)

    def postprocess_num_computed_tokens(self, input_batch: AscendInputBatch) -> None:
        # TODO: Refactor this 310P state update to use Triton Dispatcher after
        # vLLM RFC #45133 lands.
        query_lens = input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]
        self.req_states.num_computed_tokens.gpu.index_add_(
            0,
            input_batch.idx_mapping,
            query_lens.to(self.req_states.num_computed_tokens.gpu.dtype),
        )
