# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import initialize_mamba_ssu_backend
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1 import kv_cache_interface
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu.attn_utils import (
    get_shared_kv_cache_layers,
    init_attn_backend,
)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids
from vllm.v1.worker.utils import bind_kv_cache

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.kv_block_zeroer import AscendKVBlockZeroer310V2
from vllm_ascend._310p.worker.v2.states import Ascend310PRequestState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, get_kv_cache_tensor_layers, vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

if not vllm_version_is("0.27.1"):
    from vllm.v1.worker.gpu.model_runner import BatchReqState

_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310V2(NPUModelRunner):
    """Model runner v2 for Ascend 310P."""

    # 310P Attention requires private ACL NZ K/V buffers while Mamba uses
    # private contiguous ND state buffers. It cannot consume vLLM main's
    # standardized shared backing, so the worker must scale multi-group KV
    # capacity before the engine computes num_blocks.
    supports_standardized_shared_kv_backing = False

    # TODO: Refactor Triton-dependent overrides to register 310P
    # implementations through Triton Dispatcher after vLLM RFC #45133 lands.
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        super().__init__(vllm_config, device)
        self.req_states = Ascend310PRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_ids_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int32, device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64, device="cpu")
        self.next_prefill_tokens_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device="cpu")

    @staticmethod
    def _validate_config(vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        # Qwen3-VL (multimodal + MRoPE) and Qwen3.5 (hybrid + GDN + MRoPE) are
        # in scope for 310P MRv2. MLA / sleep remain unsupported.
        if model_config.use_mla:
            raise NotImplementedError("MLA is not supported by model runner v2 on 310P.")
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

    def _prepare_inputs_310p(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> AscendInputBatch:
        # TODO: Refactor this Triton-free input preparation through Triton
        # Dispatcher after vLLM RFC #45133 lands.
        # ``super().execute_model`` has already run finish/add/update_requests and
        # ``apply_staged_writes``; sync GPU counts now so mamba preprocess matches
        # the CPU/np values used for positions and slot mappings.
        self._sync_num_computed_tokens_gpu_from_np()

        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        if vllm_version_is("0.27.1"):
            req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
        else:
            req_ids = sort_batch_req_ids(
                num_tokens_per_req,
                scheduler_output.scheduled_spec_decode_tokens,
                self.decode_query_len,
            )
        self._update_seq_lens_cpu(scheduler_output, req_ids)

        num_scheduled_tokens = np.fromiter(
            map(num_tokens_per_req.get, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens,
            num_scheduled_tokens,
        )
        idx_mapping_np = np.fromiter(
            map(self.req_states.req_id_to_index.get, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_after_padding,
                num_reqs_padded,
                num_reqs,
                query_start_loc_np,
                batch_desc.cg_mode,
                batch_desc.num_reqs,
            )
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]

        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
        batch_has_prefill = bool(np.any(is_prefilling_np))
        self.eplb.set_batch_phase(batch_has_prefill)
        if batch_has_prefill:
            self._prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
                idx_mapping_np=idx_mapping_np,
                query_start_loc_np=query_start_loc_np,
            )

        self._prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
            idx_mapping_np=idx_mapping_np,
            query_start_loc_np=query_start_loc_np,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        self.input_buffers.seq_lens_np[num_reqs_padded:] = 0
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
        logits_indices = self._combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            num_reqs,
            self.model_state.num_new_sampled_tokens_per_step,
            idx_mapping_np=idx_mapping_np,
            query_start_loc_np=query_start_loc_np,
            seq_lens_np=self.input_buffers.seq_lens_np[:num_reqs],
            prefill_len_np=prefill_len_np,
        )

        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            self.req_states.num_computed_tokens_np[idx_mapping_np],
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        input_batch_kwargs: dict[str, Any] = dict(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=None,
            is_prefilling_np=is_prefilling_np,
            num_computed_tokens_np=self.req_states.num_computed_tokens_np[idx_mapping_np],
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=None,
            seq_lens_np=self.input_buffers.seq_lens_np,
            attn_state=attn_state,
        )
        if vllm_version_is("0.27.1"):
            input_batch_kwargs["max_seq_len_np"] = None
        else:
            input_batch_kwargs["has_prefill"] = batch_has_prefill
        input_batch = AscendInputBatch(**input_batch_kwargs)
        # MRoPE positions are built in ``model_state.prepare_inputs``; the 1D
        # arange buffer above is only for slot-mapping / non-MRoPE paths.
        if not self.model_config.uses_mrope:
            update_cos_sin(input_batch.positions)
        return input_batch

    def _sync_num_computed_tokens_gpu_from_np(self) -> None:
        """Mirror ``num_computed_tokens_np`` onto GPU before mamba preprocess."""
        np_vals = self.req_states.num_computed_tokens_np
        gpu = self.req_states.num_computed_tokens.gpu
        gpu.copy_(torch.from_numpy(np_vals).to(device=gpu.device, dtype=gpu.dtype))
        self.req_states.num_computed_tokens_cpu.copy_(torch.from_numpy(np_vals))
        self.req_states.num_computed_tokens.cpu.copy_(torch.from_numpy(np_vals))

    if vllm_version_is("0.27.1"):

        def prepare_inputs(
            self,
            scheduler_output: SchedulerOutput,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            return self._prepare_inputs_310p(scheduler_output, batch_desc)

    else:

        def prepare_inputs(  # type: ignore[misc, override]
            self,
            scheduler_output: SchedulerOutput,
            batch_req_state: BatchReqState,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            del batch_req_state
            return self._prepare_inputs_310p(scheduler_output, batch_desc)

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        super().finish_requests(scheduler_output)
        if scheduler_output.finished_req_ids:
            # Same barrier as 310P MRv1 ``_update_states``: ACLGraph may still
            # be reading the previous block-table layout while finish_requests
            # rewrites CPU NumPy tables for a reused slot. Upstream GPU/MRv2
            # does not need this because it does not use that CPU gather path.
            torch.npu.current_stream().synchronize()

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
        """Allocate 310P attention caches as NZ and hybrid Mamba caches as ND."""
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        circular_buffer_spec = getattr(kv_cache_interface, "CircularBufferSpec", None)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = next(iter(spec.kv_cache_specs.values())) if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            slot_mapping_enabled.append(
                circular_buffer_spec is None or not isinstance(layer_spec, circular_buffer_spec)
            )
            max_num_blocks = cdiv(self.max_model_len, spec.block_size)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            if isinstance(spec, MambaSpec):
                # Without prefix caching, hybrid recurrent state uses one page
                # per request plus speculative slots (same as 820_new).
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
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
            slot_mapping_enabled=slot_mapping_enabled,
        )
        initialize_mamba_ssu_backend(self.vllm_config.mamba_config, self.kv_cache_config)

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
        if kv_cache_config.needs_kv_cache_zeroing:
            self._init_kv_zero_meta()
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _adjust_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> None:
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                specs = tuple(group_spec.kv_cache_specs.values())
            else:
                specs = (group_spec,)
            attention_specs = [spec for spec in specs if isinstance(spec, AttentionSpec)]
            # Hybrid groups may be pure MambaSpec; skip kernel-block sizing.
            if not attention_specs:
                continue
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

    def _init_kv_zero_meta(self) -> None:
        self.kv_block_zeroer = AscendKVBlockZeroer310V2(self.device, is_pin_memory_available())
        self.kv_block_zeroer.init_meta(
            attn_groups_iter=(group for groups in self.attn_groups for group in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=getattr(self, "runner_only_attn_layers", set()),
            static_forward_context=self.compilation_config.static_forward_context,
        )

    def _allocate_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        shared_layers: dict[str, str],
    ) -> dict[str, Any]:
        """Allocate attention caches as NZ and hybrid Mamba state as ND."""
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
            layer_names = [name for name in get_kv_cache_tensor_layers(kv_cache_tensor) if name not in shared_layers]
            if not layer_names:
                continue
            cache_groups: dict[tuple[Any, ...], list[str]] = {}
            for layer_name in layer_names:
                kv_cache_spec = layer_specs[layer_name]
                cache_key: tuple[Any, ...]
                if isinstance(kv_cache_spec, AttentionSpec):
                    backend = layer_backends[layer_name]
                    group_id = layer_group_ids[layer_name]
                    storage_block_size = getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)
                    kernel_block_size = (
                        storage_block_size
                        if storage_block_size != kv_cache_spec.block_size
                        else self.kernel_block_sizes[group_id]
                    )
                    cache_key = (kv_cache_spec, backend, kernel_block_size)
                else:
                    cache_key = (kv_cache_spec,)
                cache_groups.setdefault(cache_key, []).append(layer_name)

            for cache_key, cache_layer_names in cache_groups.items():
                layer_name = cache_layer_names[0]
                kv_cache_spec = layer_specs[layer_name]
                legacy_shared_by = vllm_version_is("0.27.1")
                if legacy_shared_by:
                    if kv_cache_tensor.size % kv_cache_spec.page_size_bytes != 0:
                        raise ValueError("KV cache allocation is not page aligned.")
                    num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                    if num_blocks < kv_cache_config.num_blocks:
                        raise ValueError("KV cache allocation contains fewer blocks than requested.")
                else:
                    # On main, descriptor.size is the complete standardized
                    # backing size. 310P does not materialize that backing;
                    # its private tensors must use the manager's per-layer
                    # block count instead.
                    num_blocks = kv_cache_config.num_blocks

                if isinstance(kv_cache_spec, AttentionSpec):
                    backend = cache_key[1]
                    kernel_block_size = cache_key[2]
                    if not issubclass(backend, AscendAttentionBackend310):
                        raise TypeError(f"310P selected unexpected attention backend {backend}.")
                    blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
                    kv_cache_shape = backend.get_kv_cache_shape(
                        num_blocks * blocks_per_kv_block,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        self.cache_config.cache_dtype,
                    )
                    if getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size) != kv_cache_spec.head_size:
                        raise NotImplementedError("310P MRV2 does not support asymmetric K/V head sizes.")
                    # Symmetric NZ only: K/V share the 4D view ``kv_cache_shape[1:]``.
                    kv_view_shape = kv_cache_shape[1:]
                    if legacy_shared_by:
                        cache: Any = (
                            torch_npu.empty_with_format(
                                size=kv_view_shape,
                                dtype=kv_cache_spec.dtype,
                                device=self.device,
                                acl_format=ACL_FORMAT_FRACTAL_NZ,
                            ),
                            torch_npu.empty_with_format(
                                size=kv_view_shape,
                                dtype=kv_cache_spec.dtype,
                                device=self.device,
                                acl_format=ACL_FORMAT_FRACTAL_NZ,
                            ),
                        )
                        for name in cache_layer_names:
                            kv_caches[name] = cache
                    else:
                        # Standardized descriptors list distinct layer
                        # regions. Allocate one private NZ K/V pair per layer;
                        # only explicit shared_layers below may alias.
                        for name in cache_layer_names:
                            kv_caches[name] = (
                                torch_npu.empty_with_format(
                                    size=kv_view_shape,
                                    dtype=kv_cache_spec.dtype,
                                    device=self.device,
                                    acl_format=ACL_FORMAT_FRACTAL_NZ,
                                ),
                                torch_npu.empty_with_format(
                                    size=kv_view_shape,
                                    dtype=kv_cache_spec.dtype,
                                    device=self.device,
                                    acl_format=ACL_FORMAT_FRACTAL_NZ,
                                ),
                            )
                elif isinstance(kv_cache_spec, MambaSpec):
                    # Hybrid recurrent state stays ND (int8 raw plus views).
                    # Main's descriptor.size is the entire virtual backing;
                    # private 310P state uses only this layer's pages.
                    raw_size = kv_cache_tensor.size if legacy_shared_by else num_blocks * kv_cache_spec.page_size_bytes

                    def allocate_mamba_cache(
                        raw_size: int = raw_size,
                        kv_cache_spec: MambaSpec = kv_cache_spec,
                        num_blocks: int = num_blocks,
                    ) -> list[torch.Tensor]:
                        raw_tensor = torch.zeros(
                            raw_size,
                            dtype=torch.int8,
                            device=self.device,
                        )
                        state_tensors = []
                        storage_offset_bytes = 0
                        for shape, dtype in zip(
                            kv_cache_spec.shapes,
                            kv_cache_spec.dtypes,
                        ):
                            dtype_size = get_dtype_size(dtype)
                            target_shape = (num_blocks, *shape)
                            stride = torch.empty(target_shape).stride()
                            state_tensors.append(
                                torch.as_strided(
                                    raw_tensor.view(dtype),
                                    size=target_shape,
                                    stride=(stride[0], *stride[1:]),
                                    storage_offset=(storage_offset_bytes // dtype_size),
                                )
                            )
                            storage_offset_bytes += target_shape[0] * stride[0] * dtype_size
                        return state_tensors

                    if legacy_shared_by:
                        cache = allocate_mamba_cache()
                        for name in cache_layer_names:
                            kv_caches[name] = cache
                    else:
                        for name in cache_layer_names:
                            kv_caches[name] = allocate_mamba_cache()
                else:
                    raise NotImplementedError(f"Unsupported 310P KV cache spec: {type(kv_cache_spec).__name__}.")

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
