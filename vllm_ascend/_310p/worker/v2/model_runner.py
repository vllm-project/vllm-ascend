# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import initialize_mamba_ssu_backend
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers, init_attn_backend
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.utils import bind_kv_cache

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2.aclgraph import ModelAclGraphManager310
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.feature_support import (
    FIRST_RELEASE_FEATURE_SUPPORT,
    MRv2FeatureSupport,
)
from vllm_ascend._310p.worker.v2.kernel_registry import register_310p_kernels
from vllm_ascend._310p.worker.v2.kv_block_zeroer import AscendKVBlockZeroer310V2
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend._310p.worker.v2.states import Ascend310PRequestState
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.pcp_manager import maybe_build_ascend_pcp_manager

# Probe the optional future dispatcher only when the 310P V2 runner is loaded.
# Model Runner V1 never imports this module.
register_310p_kernels()

_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310V2(NPUModelRunner):
    """First-release Model Runner V2 implementation for Ascend 310P."""

    aclgraph_manager_cls = ModelAclGraphManager310
    request_state_cls = Ascend310PRequestState
    feature_support: MRv2FeatureSupport = FIRST_RELEASE_FEATURE_SUPPORT

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_first_release_config(vllm_config)
        super().__init__(vllm_config, device)
        self.sampler = Ascend310PGreedySampler()
        self.input_ids_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int32, device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64, device="cpu")
        self.next_prefill_tokens_cpu = torch.zeros(1, self.max_num_reqs, dtype=torch.int32, device="cpu")

    @property
    def supports_prefix_caching(self) -> bool:
        return self.feature_support.prefix_caching

    @property
    def supports_qwen3_5_mtp(self) -> bool:
        return self.feature_support.qwen3_5_mtp

    @property
    def supports_mtp(self) -> bool:
        """Compatibility alias for callers that do not distinguish MTP models."""
        return self.supports_qwen3_5_mtp

    @classmethod
    def _validate_first_release_config(cls, vllm_config: VllmConfig) -> None:
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

        cls.feature_support.validate_config(vllm_config)
        if vllm_config.lora_config is not None:
            raise NotImplementedError("LoRA is outside the 310P Model Runner V2 V1-alignment scope.")
        if getattr(parallel_config, "enable_expert_parallel", False):
            raise NotImplementedError("Expert parallelism is outside the 310P Model Runner V2 first-release scope.")
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("KV transfer is outside the 310P Model Runner V2 first-release scope.")
        if getattr(vllm_config.model_config, "enable_sleep_mode", False):
            raise NotImplementedError("Sleep mode is outside the 310P Model Runner V2 first-release scope.")

    def _get_uniform_decode_query_len(self) -> int:
        """Bridge vLLM versions that do not expose this V2 attribute."""
        return getattr(self, "uniform_decode_query_len", self.decode_query_len)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Restore linear-attention specs omitted by some upstream V2 versions."""
        kv_cache_spec = super().get_kv_cache_spec()
        static_forward_context = self.compilation_config.static_forward_context
        for layer_name, layer in static_forward_context.items():
            if "linear_attn" not in layer_name or layer_name in kv_cache_spec:
                continue
            get_spec = getattr(layer, "get_kv_cache_spec", None)
            if get_spec is None:
                continue
            if spec := get_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec
        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize 310P V2 KV cache directly in its required formats."""
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            block_table_max_model_len = max(
                block_table_max_model_len,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            max_num_blocks = cdiv(block_table_max_model_len, spec.block_size * self.dcp_size)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            if isinstance(spec, MambaSpec):
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )
        self._adjust_kernel_block_sizes_310p(kv_cache_config)
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
        initialize_mamba_ssu_backend(self.vllm_config.mamba_config, self.kv_cache_config)

        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self._get_uniform_decode_query_len(),
            self.parallel_config.tensor_parallel_size,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        self.cudagraph_manager = self.aclgraph_manager_cls(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.decode_query_len,
            self,
        )
        if self.speculator is not None:
            self.speculator.init_cudagraph_manager(cudagraph_mode)
        check_attention_cp_compatibility(self.vllm_config)

        shared_layers = get_shared_kv_cache_layers(self.vllm_config)
        kv_caches_dict = self._allocate_kv_cache_tensors_310p(kv_cache_config, shared_layers)
        self.kv_caches: list[torch.Tensor | list[torch.Tensor]] = []
        bind_kv_cache(
            kv_caches_dict,
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )
        self._init_kv_zero_meta_if_needed(kv_cache_config)
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)
        self.pcp_manager = maybe_build_ascend_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
        )

    def _adjust_kernel_block_sizes_310p(self, kv_cache_config: KVCacheConfig) -> None:
        """Apply the 310P paged-attention block/head-size constraint."""
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                specs = tuple(group_spec.kv_cache_specs.values())
            else:
                specs = (group_spec,)
            attention_specs = [spec for spec in specs if isinstance(spec, AttentionSpec)]
            if not attention_specs:
                continue

            max_head_size = max(spec.head_size for spec in attention_specs)
            if max_head_size > 256:
                raise NotImplementedError(
                    "310P paged attention requires head_size <= 256, "
                    f"but group {group_id} has head_size={max_head_size}."
                )
            backend = self.attn_groups[group_id][0].backend
            supported_sizes = [
                block_size
                for block_size in backend.get_supported_kernel_block_sizes()
                if block_size * max_head_size <= _ATTENTION_BLOCK_SIZE_LIMIT
            ]
            if not supported_sizes:
                raise NotImplementedError(
                    "310P paged attention requires block_size * head_size "
                    f"<= {_ATTENTION_BLOCK_SIZE_LIMIT}, but group {group_id} "
                    f"has head_size={max_head_size}."
                )
            self.kernel_block_sizes[group_id] = supported_sizes[0]

    def _init_kv_zero_meta_if_needed(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize Mamba/GDN block zeroing for non-speculative hybrid models."""
        if kv_cache_config.needs_kv_cache_zeroing:
            self._init_kv_zero_meta()

    def _allocate_kv_cache_tensors_310p(
        self,
        kv_cache_config: KVCacheConfig,
        shared_layers: dict[str, str],
    ) -> dict[str, Any]:
        """Allocate attention caches as NZ and hybrid state caches as ND."""
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
                kv_cache_spec = layer_specs[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec):
                    backend = layer_backends[layer_name]
                    group_id = layer_group_ids[layer_name]
                    if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                        kernel_block_size = kv_cache_spec.storage_block_size
                    else:
                        kernel_block_size = self.kernel_block_sizes[group_id]
                    cache_key = (kv_cache_spec, backend, kernel_block_size)
                else:
                    cache_key = (kv_cache_spec,)
                cache_groups.setdefault(cache_key, []).append(layer_name)

            for cache_key, cache_layer_names in cache_groups.items():
                layer_name = cache_layer_names[0]
                kv_cache_spec = layer_specs[layer_name]
                assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                assert num_blocks >= kv_cache_config.num_blocks

                if isinstance(kv_cache_spec, AttentionSpec):
                    _, backend, kernel_block_size = cache_key
                    if not issubclass(backend, AscendAttentionBackend310):
                        raise TypeError(f"310P attention layer {layer_name} selected unexpected backend {backend}.")
                    blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
                    kv_cache_shape = backend.get_kv_cache_shape(
                        num_blocks * blocks_per_kv_block,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        self.cache_config.cache_dtype,
                    )
                    head_size_v = getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size)
                    if head_size_v != kv_cache_spec.head_size:
                        raise NotImplementedError("310P V2 does not support asymmetric K/V head sizes.")
                    k_shape = v_shape = kv_cache_shape[1:]
                    k_cache = torch_npu.empty_with_format(
                        size=k_shape,
                        dtype=kv_cache_spec.dtype,
                        device=self.device,
                        acl_format=ACL_FORMAT_FRACTAL_NZ,
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=v_shape,
                        dtype=kv_cache_spec.dtype,
                        device=self.device,
                        acl_format=ACL_FORMAT_FRACTAL_NZ,
                    )
                    cache = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        state_tensors.append(
                            torch.as_strided(
                                raw_tensor.view(dtype),
                                size=target_shape,
                                # Mamba state pages are contiguous in the hybrid raw cache.
                                # page_size_bytes is the allocation unit, not a tensor stride.
                                stride=(stride[0], *stride[1:]),
                                storage_offset=storage_offset_bytes // dtype_size,
                            )
                        )
                        storage_offset_bytes += stride[0] * dtype_size
                    cache = state_tensors
                else:
                    raise NotImplementedError(f"Unsupported 310P KV cache spec: {type(kv_cache_spec).__name__}")

                for name in cache_layer_names:
                    kv_caches[name] = cache

        for layer_name, target_layer_name in shared_layers.items():
            kv_caches[layer_name] = kv_caches[target_layer_name]

        expected_layers = {
            layer_name
            for kv_cache_group in kv_cache_config.kv_cache_groups
            for layer_name in kv_cache_group.layer_names
        }
        assert expected_layers == set(kv_caches), "Some 310P KV cache layers were not initialized."
        return kv_caches

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
            computed_delta = self._get_valid_query_lens(idx_mapping, query_start_loc)
            self.req_states.num_computed_tokens.gpu.index_add_(
                0,
                valid_indices,
                computed_delta.to(self.req_states.num_computed_tokens.gpu.dtype),
            )

        self.model_state.postprocess_state(idx_mapping, num_sampled)
        self._copy_num_computed_tokens_to_cpu()

    @staticmethod
    def _get_valid_query_lens(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Return query lengths for real requests, excluding graph padding."""
        # A FULL graph may pad idx_mapping to its capture bucket while
        # query_start_loc still contains boundaries for real requests only.
        # Padding entries are trailing -1 sentinels and must not participate in
        # the num_computed_tokens update.
        num_query_lens = min(idx_mapping.shape[0], query_start_loc.shape[0] - 1)
        query_lens = query_start_loc[1 : num_query_lens + 1] - query_start_loc[:num_query_lens]
        return query_lens.masked_select(idx_mapping[:num_query_lens] >= 0)

    def postprocess_num_computed_tokens(self, input_batch: AscendInputBatch) -> None:
        query_lens = input_batch.query_start_loc[1:] - input_batch.query_start_loc[:-1]
        self.req_states.num_computed_tokens.gpu.index_add_(
            0,
            input_batch.idx_mapping,
            query_lens.to(self.req_states.num_computed_tokens.gpu.dtype),
        )
        self._copy_num_computed_tokens_to_cpu()

    def _init_kv_zero_meta(self) -> None:
        self.kv_block_zeroer = AscendKVBlockZeroer310V2(self.device, is_pin_memory_available())
        self.kv_block_zeroer.init_meta(
            attn_groups_iter=(group for groups in self.attn_groups for group in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=getattr(self, "runner_only_attn_layers", set()),
            static_forward_context=self.compilation_config.static_forward_context,
        )
