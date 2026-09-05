# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence
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
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
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
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import bind_kv_cache, copy_kv_cache_blocks_inplace

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.kv_block_zeroer import AscendKVBlockZeroer310V2
from vllm_ascend._310p.worker.v2.spec_utils import (
    combine_sampled_and_draft_tokens_cpu,
    expand_idx_mapping_cpu,
)
from vllm_ascend._310p.worker.v2.states import Ascend310PRequestState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

if not vllm_version_is("0.27.1"):
    from vllm.v1.worker.gpu.model_runner import BatchReqState

_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128


class NPUModelRunner310V2(NPUModelRunner):
    """Model runner v2 for Ascend 310P."""

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
        # PrefillCacheHit / ChunkedPrefill must not replay FULL mixed ACLGraphs
        # (same as MRv1 `_determine_batch_execution_and_padding`). FULL_DECODE_ONLY
        # already keeps those batches eager via mixed_mode=NONE.
        self._force_eager_pc_batch = False
        self._force_eager_spec_batch = False
        self._spec_dummy_capture = False
        # Populated in initialize_kv_cache; also set here so UT can call
        # ``_allocate_kv_cache_tensors`` without going through that path.
        self._attn_kv_copy_params: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self._attn_kv_storage_ptrs: set[int] = set()

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
        if vllm_config.speculative_config is not None:
            spec = vllm_config.speculative_config
            if spec.method != "mtp":
                raise NotImplementedError(
                    f"310P model runner v2 only supports MTP speculative decoding, got {spec.method!r}."
                )
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("KV cache transfer is not supported by model runner v2 on 310P.")
        # Prefix caching is supported: 310P MRv2 reuses CPU Ascend310PBlockTables /
        # PrefillCacheHit→splitfuse (attention_v1) and hybrid Mamba page sizing below.
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
        num_valid_tokens = num_scheduled_tokens
        draft_tokens_map = scheduler_output.scheduled_spec_decode_tokens
        if draft_tokens_map:
            num_valid_tokens = np.array(
                [
                    num_tokens - len(draft_tokens_map.get(req_id, ()))
                    for num_tokens, req_id in zip(num_scheduled_tokens, req_ids)
                ],
                dtype=np.int32,
            )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens,
            num_valid_tokens,
        )
        idx_mapping_np = np.fromiter(
            map(self.req_states.req_id_to_index.get, req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        num_draft_tokens_per_req = None
        if not draft_tokens_map:
            total_num_draft_tokens = 0
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens_map.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            num_logits_per_req = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits_per_req, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)
            total_num_logits = int(cu_num_logits_np[-1])
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping_cpu(
                idx_mapping,
                total_num_logits,
                cu_num_logits_np,
            )

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
        total_num_logits = num_reqs if not draft_tokens_map else int(cu_num_logits_np[-1])
        logits_indices = self._combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
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
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens if draft_tokens_map else 0,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=None,
            is_prefilling_np=is_prefilling_np,
            num_computed_tokens_np=self.req_states.num_computed_tokens_np[idx_mapping_np],
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            max_seq_len_np=None,
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
        if not vllm_version_is("0.27.1"):
            input_batch_kwargs["has_prefill"] = batch_has_prefill
        input_batch = AscendInputBatch(**input_batch_kwargs)
        # MRoPE positions are built in ``model_state.prepare_inputs``; the 1D
        # arange buffer above is only for slot-mapping / non-MRoPE paths.
        if not self.model_config.uses_mrope:
            update_cos_sin(input_batch.positions)
        return input_batch

    def _scheduler_output_needs_pc_eager(self, scheduler_output: SchedulerOutput) -> bool:
        """Force eager for PrefillCacheHit / ChunkedPrefill when mixed FULL graphs exist."""
        if not self.cache_config.enable_prefix_caching:
            return False
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if not cudagraph_mode.has_full_cudagraphs():
            return False
        # FULL_DECODE_ONLY: mixed_mode is NONE → prefill/PC hits are already eager.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL:
            return False

        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        if num_reqs == 0:
            return False

        computed_by_req: dict[str, int] = {}
        for req in scheduler_output.scheduled_new_reqs:
            computed_by_req[req.req_id] = int(req.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None:
            for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
                computed_by_req[req_id] = int(num_computed)
        for req_id in num_tokens_per_req:
            if req_id in computed_by_req:
                continue
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is not None:
                computed_by_req[req_id] = int(self.req_states.num_computed_tokens_np[req_idx])

        req_ids = list(num_tokens_per_req.keys())
        num_scheduled = np.fromiter(
            (num_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        seq_lens = np.fromiter(
            (computed_by_req.get(req_id, 0) + int(num_tokens_per_req[req_id]) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        attn_state = build_attn_state(
            self.vllm_config,
            seq_lens,
            num_reqs,
            num_scheduled,
            num_scheduled,
        )
        # Avoid importing AscendAttentionState at module top (heavy attention_v1).
        return attn_state.name in ("PrefillCacheHit", "ChunkedPrefill")

    def _scheduler_output_needs_spec_eager(self, scheduler_output: SchedulerOutput) -> bool:
        """Force eager when MTP verify batch is not uniform spec-decode."""
        if self.speculative_config is None:
            return False
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if not cudagraph_mode.has_full_cudagraphs():
            return False

        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        if num_reqs == 0:
            return False

        num_scheduled = np.fromiter(num_tokens_per_req.values(), dtype=np.int32, count=num_reqs)
        if not np.all(num_scheduled == self.decode_query_len):
            return True
        if scheduler_output.total_num_scheduled_tokens != int(num_scheduled.sum()):
            return True

        computed_by_req: dict[str, int] = {}
        for req in scheduler_output.scheduled_new_reqs:
            computed_by_req[req.req_id] = int(req.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None:
            for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
                computed_by_req[req_id] = int(num_computed)
        for req_id in num_tokens_per_req:
            if req_id in computed_by_req:
                continue
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is not None:
                computed_by_req[req_id] = int(self.req_states.num_computed_tokens_np[req_idx])

        req_ids = list(num_tokens_per_req.keys())
        if any(computed_by_req.get(req_id, 0) == 0 for req_id in req_ids):
            return True

        seq_lens = np.fromiter(
            (computed_by_req[req_id] + num_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        attn_state = build_attn_state(
            self.vllm_config,
            seq_lens,
            num_reqs,
            num_scheduled,
            num_scheduled,
        )
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        return attn_state != AscendAttentionState.SpecDecoding

    def _install_pc_eager_cudagraph_dispatch(self) -> None:
        """Wrap ACLGraph dispatch so PrefillCacheHit cannot replay FULL mixed graphs."""
        manager = self.cudagraph_manager
        if manager is None or getattr(manager, "_310p_pc_eager_wrapped", False):
            return
        orig_dispatch = manager.dispatch
        runner = self

        def dispatch(
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: int | None,
            num_active_loras: int,
            max_query_len: int | None = None,
        ) -> BatchExecutionDescriptor:
            if runner._force_eager_pc_batch or runner._force_eager_spec_batch:
                return BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
            return orig_dispatch(
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len=max_query_len,
            )

        manager.dispatch = dispatch  # type: ignore[method-assign]
        manager._310p_pc_eager_wrapped = True  # type: ignore[attr-defined]

    def _sync_num_computed_tokens_gpu_from_np(self) -> None:
        """Mirror ``num_computed_tokens_np`` onto GPU before mamba preprocess.

        Must run after ``add_requests`` / ``update_requests`` (see ``prepare_inputs``):
        ``update_requests`` only refreshes the CPU/np mirror for cached requests,
        and prefix-cache hits seed ``num_computed_tokens_np`` in ``add_request`` while
        the GPU tensor may still hold a freed slot or pre-``apply_staged_writes``
        value. Hybrid align ``preprocess_state`` reads the GPU tensor, so syncing too
        early in ``execute_model`` leaves stale counts and corrupts recurrent state.
        """
        np_vals = self.req_states.num_computed_tokens_np
        gpu = self.req_states.num_computed_tokens.gpu
        gpu.copy_(torch.from_numpy(np_vals).to(device=gpu.device, dtype=gpu.dtype))
        self.req_states.num_computed_tokens_cpu.copy_(torch.from_numpy(np_vals))
        self.req_states.num_computed_tokens.cpu.copy_(torch.from_numpy(np_vals))

    def _advance_num_computed_tokens(self, valid_indices: torch.Tensor, query_lens: torch.Tensor) -> None:
        """Advance per-request computed counts on both CPU mirror and GPU tensor."""
        if valid_indices.numel() == 0:
            return
        vi = valid_indices.detach().cpu().numpy()
        ql = query_lens.detach().cpu().numpy().astype(np.int32, copy=False)
        self.req_states.num_computed_tokens_np[vi] += ql
        self.req_states.num_computed_tokens.gpu.index_add_(
            0,
            valid_indices,
            query_lens.to(self.req_states.num_computed_tokens.gpu.dtype),
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ):
        self._force_eager_pc_batch = False
        self._force_eager_spec_batch = False
        if not dummy_run:
            self._force_eager_pc_batch = self._scheduler_output_needs_pc_eager(scheduler_output)
            self._force_eager_spec_batch = self._scheduler_output_needs_spec_eager(scheduler_output)
        try:
            if vllm_version_is("0.27.1"):
                return super().execute_model(
                    scheduler_output,
                    intermediate_tensors=intermediate_tensors,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    is_profile=is_profile,
                )
            return super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                context_len=context_len,
            )
        finally:
            self._force_eager_pc_batch = False
            self._force_eager_spec_batch = False

    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ):
        if self.speculative_config is not None and uniform_decode and not is_profile:
            self._spec_dummy_capture = True
            try:
                return super()._dummy_run(
                    num_tokens,
                    *args,
                    skip_attn=skip_attn,
                    uniform_decode=uniform_decode,
                    skip_eplb=skip_eplb,
                    is_profile=is_profile,
                    **kwargs,
                )
            finally:
                self._spec_dummy_capture = False
        return super()._dummy_run(
            num_tokens,
            *args,
            skip_attn=skip_attn,
            uniform_decode=uniform_decode,
            skip_eplb=skip_eplb,
            is_profile=is_profile,
            **kwargs,
        )

    def _build_attention_metadata(self, *args: Any, **kwargs: Any):
        if self._spec_dummy_capture:
            from vllm_ascend.attention.attention_v1 import AscendAttentionState

            self.attn_state = AscendAttentionState.SpecDecoding
        return super()._build_attention_metadata(*args, **kwargs)

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

    @staticmethod
    def _dedupe_kv_cache_block_copies(
        kv_cache_block_copies: Sequence[KVCacheBlockCopy],
    ) -> list[KVCacheBlockCopy]:
        """Drop duplicate CoW pairs from hybrid multi-manager prefix-cache hits.

        Qwen3.5 hybrid models register the same physical block copy once per
        KV-cache group (attention + mamba align). Upstream assumes a unified
        backing allocation, so repeating the pair is harmless there. 310P
        attention NZ caches are separate storages; applying the merged list
        twice overflows the block view during ``copy_kv_cache_blocks_inplace``.
        """
        seen: set[tuple[int, int]] = set()
        deduped: list[KVCacheBlockCopy] = []
        for copy in kv_cache_block_copies:
            key = (copy.src_block_id, copy.dst_block_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(copy)
        return deduped

    def _copy_kv_cache_blocks_310p(self, kv_cache_block_copies: Sequence[KVCacheBlockCopy]) -> None:
        """Copy-on-write for hybrid prefix cache on 310P.

        Upstream ``copy_kv_cache_blocks_inplace`` assumes every KV storage aliases
        one block-major ``[num_blocks, page_size]`` byte view. 310P attention
        caches are separate FRACTAL_NZ K/V tensors whose kernel-block layout does
        not match ``page_size_bytes``; applying the generic copy on them triggers
        ``setStorage`` OOM during gsm8k prefix-cache hits. Mamba state still uses
        the block-major backing buffer and keeps the upstream copy path.
        """
        if not kv_cache_block_copies:
            return

        indices_np = np.array(
            [[copy.src_block_id, copy.dst_block_id] for copy in kv_cache_block_copies],
            dtype=np.int64,
        )
        seen_attn_storage: set[int] = set()
        for k_cache, v_cache, blocks_per_kv_block in self._attn_kv_copy_params:
            storage_ptr = k_cache.untyped_storage().data_ptr()
            if storage_ptr in seen_attn_storage:
                continue
            seen_attn_storage.add(storage_ptr)
            for src_block_id, dst_block_id in indices_np:
                src_start = int(src_block_id) * blocks_per_kv_block
                src_end = src_start + blocks_per_kv_block
                dst_start = int(dst_block_id) * blocks_per_kv_block
                dst_end = dst_start + blocks_per_kv_block
                k_cache[dst_start:dst_end].copy_(k_cache[src_start:src_end])
                v_cache[dst_start:dst_end].copy_(v_cache[src_start:src_end])

        mamba_entries = [entry for entry in self.kv_caches if isinstance(entry, list)]
        if mamba_entries:
            copy_kv_cache_blocks_inplace(
                mamba_entries,
                self.kv_cache_config.num_blocks,
                kv_cache_block_copies,
            )

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        copies = scheduler_output.kv_cache_block_copies
        pending_copies: list[KVCacheBlockCopy] | None = None
        if copies:
            pending_copies = self._dedupe_kv_cache_block_copies(copies)
        # Run block-table / zeroing updates without the upstream copy, which
        # mishandles 310P NZ attention storages.
        scheduler_output.kv_cache_block_copies = None
        super().update_requests(scheduler_output)
        if pending_copies:
            self._copy_kv_cache_blocks_310p(pending_copies)

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
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
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
        if isinstance(self.speculator, DraftModelSpeculator):
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        shared_layers = get_shared_kv_cache_layers(self.vllm_config)
        self._attn_kv_copy_params = []
        self._attn_kv_storage_ptrs = set()
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
        self._install_pc_eager_cudagraph_dispatch()

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
            layer_names = [name for name in kv_cache_tensor.shared_by if name not in shared_layers]
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
                if kv_cache_tensor.size % kv_cache_spec.page_size_bytes != 0:
                    raise ValueError("KV cache allocation is not page aligned.")
                num_blocks = kv_cache_config.num_blocks
                if kv_cache_tensor.size // kv_cache_spec.page_size_bytes < num_blocks:
                    raise ValueError("KV cache allocation contains fewer blocks than requested.")

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
                    k_cache = torch_npu.empty_with_format(
                        size=kv_view_shape,
                        dtype=kv_cache_spec.dtype,
                        device=self.device,
                        acl_format=ACL_FORMAT_FRACTAL_NZ,
                    )
                    v_cache = torch_npu.empty_with_format(
                        size=kv_view_shape,
                        dtype=kv_cache_spec.dtype,
                        device=self.device,
                        acl_format=ACL_FORMAT_FRACTAL_NZ,
                    )
                    cache: Any = (k_cache, v_cache)
                    storage_ptr = k_cache.untyped_storage().data_ptr()
                    if storage_ptr not in self._attn_kv_storage_ptrs:
                        self._attn_kv_storage_ptrs.add(storage_ptr)
                        self._attn_kv_copy_params.append((k_cache, v_cache, blocks_per_kv_block))
                elif isinstance(kv_cache_spec, MambaSpec):
                    # Hybrid recurrent state stays ND (int8 raw + as_strided views).
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
                                stride=(stride[0], *stride[1:]),
                                storage_offset=storage_offset_bytes // dtype_size,
                            )
                        )
                        storage_offset_bytes += stride[0] * dtype_size
                    cache = state_tensors
                else:
                    raise NotImplementedError(f"Unsupported 310P KV cache spec: {type(kv_cache_spec).__name__}.")

                for name in cache_layer_names:
                    kv_caches[name] = cache

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
        cu_num_logits_np = cu_num_logits.detach().cpu().numpy().astype(np.int32, copy=False)
        return combine_sampled_and_draft_tokens_cpu(
            input_ids,
            idx_mapping_np,
            last_sampled_tokens,
            query_start_loc_np,
            seq_lens_np,
            prefill_len_np,
            draft_tokens,
            cu_num_logits_np,
            num_logits,
            num_bonus_tokens,
            device=self.device,
        )

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
        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            sampler_output = self.sampler(logits, input_batch)
            can_sample_np = input_batch.seq_lens_np[: input_batch.num_reqs] >= input_batch.prefill_len_np
            num_sampled = async_copy_to_gpu(can_sample_np.astype(np.int32), device=self.device)
            num_rejected = torch.zeros_like(num_sampled)
            sampler_output.num_sampled = num_sampled
            sampler_output.num_rejected = num_rejected
            return sampler_output, num_sampled, num_rejected

        assert self.speculator is not None
        sampler_output = self.rejection_sampler(
            logits,
            input_batch,
            self.speculator.draft_logits,
        )
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        num_entries = min(idx_mapping.shape[0], sampled_tokens.shape[0], num_sampled.shape[0])
        idx_mapping = idx_mapping[:num_entries]
        sampled_tokens = sampled_tokens[:num_entries]
        num_sampled = num_sampled[:num_entries]
        valid_mask = idx_mapping >= 0
        valid_indices = idx_mapping.masked_select(valid_mask)
        valid_num_sampled = num_sampled.masked_select(valid_mask)
        has_sample = valid_num_sampled > 0

        if self.speculator is not None and sampled_tokens.ndim == 2:
            for batch_idx in range(num_entries):
                if idx_mapping[batch_idx] < 0:
                    continue
                req_idx = int(idx_mapping[batch_idx].item())
                count = int(num_sampled[batch_idx].item())
                if count <= 0:
                    continue
                start_pos = int(self.req_states.total_len.gpu[req_idx].item())
                tokens = sampled_tokens[batch_idx, :count].to(torch.int32)
                self.req_states.all_token_ids.gpu[req_idx, start_pos : start_pos + count] = tokens
                self.req_states.last_sampled_tokens[req_idx, 0] = tokens[-1]
                self.req_states.total_len.gpu[req_idx] = start_pos + count
        else:
            sampled = sampled_tokens[:, 0].masked_select(valid_mask).to(self.req_states.last_sampled_tokens.dtype)
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
            if self.speculator is not None:
                query_lens = self._get_valid_query_lens(idx_mapping, query_start_loc)
                num_rejected_valid = num_rejected.masked_select(valid_mask)
                advance_lens = query_lens - num_rejected_valid.to(query_lens.dtype)
                self._advance_num_computed_tokens(valid_indices, advance_lens)
            else:
                query_lens = self._get_valid_query_lens(idx_mapping, query_start_loc)
                self._advance_num_computed_tokens(valid_indices, query_lens)

        self.model_state.postprocess_state(
            idx_mapping,
            num_sampled,
            self.req_states.num_computed_tokens.gpu,
        )

        if self.speculator is not None:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self) -> None:
        default_stream = torch.npu.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ) -> None:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        if self.speculator is not None:
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]
                self.req_states.num_computed_tokens_np[req_index] = int(self.num_computed_tokens_cpu[req_index].item())
        else:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.req_states.num_computed_tokens_np[req_index]

        for i, req_id in enumerate(req_ids):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]
            self.input_buffers.seq_lens_np[i] = self.input_buffers.seq_lens_cpu[i]

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
        # ``postprocess_sampled`` already advances ``num_computed_tokens`` on
        # both the CPU mirror and GPU tensor. Upstream GPU MRv2 splits the work
        # across ``post_update`` + ``postprocess_num_computed_tokens``; our
        # Triton-free ``postprocess_sampled`` performs both updates in one pass.
        del input_batch
