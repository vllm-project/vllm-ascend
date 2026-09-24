# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from contextlib import AbstractContextManager, contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d as async_copy_to_gpu
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.eplb_utils import step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import (
    BatchReqState,
    ExecuteModelState,
    GPUModelRunner,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.profiling_chunk_predictor import (
    _finish_profiling_chunk_timing,
    _start_profiling_chunk_timing,
)
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor, SamplePhaseResult
from vllm_ascend.runtime_guard.runner_bridge import (
    maybe_wrap_v2_async_output,
    need_pre_sample_hook,
    wrap_compute_logits_for_pre_sample,
)
from vllm_ascend.utils import (
    is_pd_decode_recompute_scheduler_enabled,
    lmhead_tp_enable,
    set_potential_max_tokens,
)
from vllm_ascend.worker.utils import disable_compilation
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.kvpp import KVPPRuntime
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.pp_utils import (
    bypass_upstream_spec_pp_guard,
    resolve_spec_pp_support,
    restore_pp_after_upstream_init,
    use_legacy_spec_pp,
)
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    # vLLM #51718 overlays hybrid Attention/Mamba groups in one standardized
    # backing allocation. Ascend MRV2 preserves that layout in
    # allocate_kv_cache_main and exposes contiguous backend-specific views.
    supports_standardized_shared_kv_backing = True

    execute_model_state: ExecuteModelState | None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        self.kvpp = KVPPRuntime()
        # Adaptive verification uses this flag to apply FIA-specific query
        # boundary and sequence length padding during FULL graph execution.
        self.use_fia = False
        # FusedMoE can be constructed by the parent initializer and reads this
        # capacity while setting up MC2 communication.
        set_potential_max_tokens(vllm_config)
        parallel_config = vllm_config.parallel_config

        # Only release versions need PP hidden during upstream initialization.
        spec_pp_support = resolve_spec_pp_support(vllm_config)
        with torch_cuda_wrapper():
            with bypass_upstream_spec_pp_guard(vllm_config, spec_pp_support) as pp_disabled:
                super().__init__(vllm_config, device)
            if pp_disabled:
                restore_pp_after_upstream_init(self, vllm_config)
        # Native PP owns token broadcast/writeback; only releases use our packing.
        # Legacy Spec+PP transport (0.28/0.29 only); deleted when 0.30+ is the floor.
        self.use_spec_pp = spec_pp_support is not None and use_legacy_spec_pp()
        # These draft heads consume target aux states collected across PP ranks.
        if spec_pp_support is not None and spec_pp_support.needs_aux_hidden_states:
            self.use_aux_hidden_state_outputs = True

        # Only a real split (size > 1) exchanges across ranks, and graph dispatch keeps it aligned.
        ftpc = self.ascend_config.finegrained_tp_config
        self._finegrained_tp_requires_graph = (
            ftpc.oproj_tensor_parallel_size > 1 or ftpc.mlp_tensor_parallel_size > 1
        ) and self.dp_size > 1

        self.use_aclgraph = (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                or breakable_cudagraph.is_breakable_cudagraph_enabled()
            )
            and not self.model_config.enforce_eager
        )
        load_collection_phase = self.ascend_config.eplb_config.load_collection_phase
        self.eplb = AscendEPLBController(
            parallel_config,
            device,
            load_collection_phase=(load_collection_phase if parallel_config.enable_eplb else "all"),
        )

        self.update_stream = None
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.req_states
        del self.input_buffers
        del self.speculator

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle.speculator
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        if self.speculative_config is not None and self.is_last_pp_rank:
            self.speculator = init_speculator(self.vllm_config, self.device)
            # Shared update_stream: main model (ModelAclGraphManager) and draft
            # (Eagle/DFlash/DSpark AclGraphManager) all use this same stream.
            self.speculator.update_stream = self.update_stream

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        if self.use_spec_pp:
            from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
                install_upstream_spec_pp_protocol,
            )

            assert self.pp_handler is not None
            install_upstream_spec_pp_protocol(self.pp_handler, self.req_states, self.num_speculative_steps)
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # Pinned D2H staging for corrected device state after spec rejection.
        # The authoritative host state is the shared NumPy/torch RequestState view.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # NOTE: In GPUModelRunner, decode_query_len is initialized in load_model(),
        # +1 is hardcoded here but not in vllm.
        self.decode_query_len = self.num_speculative_steps + 1
        # Set _mc2_tokens_capacity and _reserved_mc2_mask for MoE communication optimization.
        # TODO: remove set_cos_and_sin (together with update_cos_sin) when mla can properly handle cos/sin internally
        set_cos_and_sin(vllm_config, self.max_num_reqs, self.decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        set_potential_max_tokens(vllm_config)

        self.runtime_guard = RuntimeGuardProcessor.bind(self)

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def _restore_replicated_draft_target_states(self) -> None:
        """Restore target states consumed by a replicated PCP draft."""
        state = self.execute_model_state
        pcp_manager = self.pcp_manager
        if (
            state is None
            or pcp_manager is None
            or not self.is_last_pp_rank
            or not getattr(self.speculator, "replicated_pcp", False)
        ):
            return

        get_hidden_states = getattr(
            self.model,
            "get_mtp_target_hidden_states",
            None,
        )
        if get_hidden_states is not None:
            mtp_target_hidden_states = get_hidden_states()
            if mtp_target_hidden_states is not None:
                pcp_manager.restore_hidden_state_buffer(mtp_target_hidden_states)

        # vLLM main captures draft_hidden_states before maybe_restore_pcp_for_sampling,
        # so a replicated draft would read the PCP-local target output. Restore
        # it to the global layout up front. aux_hidden_states need no handling
        # here: upstream sample_tokens (#56107) already restores them, per
        # tensor, before speculator.propose.
        if state.hidden_states is not None:
            state = state._replace(hidden_states=pcp_manager.restore_hidden_states(state.hidden_states))
            # Tell restore_for_sampling to skip its second all-gather for this
            # step. The layout is tracked explicitly because a length match is
            # ambiguous under piecewise/FULL graphs: every rank pads its local
            # batch to the same global padded length.
            pcp_manager._sampling_hidden_restored = True
        self.execute_model_state = state

    def sample_tokens(self, grammar_output):
        pcp_manager = self.pcp_manager
        if pcp_manager is not None and not self.is_last_pp_rank and self.execute_model_state is not None:
            assert isinstance(pcp_manager, AscendPCPManager)
            # The last PP stage restores PCP outputs to the global request
            # layout before sampling. Non-last stages do not own final hidden
            # states, but their PP receive/postprocess path must use that same
            # global request layout rather than PCP-local segment rows.
            self.execute_model_state = self.execute_model_state._replace(
                input_batch=pcp_manager.global_batch,
            )

        # The pre-restore marker is scoped to this sampling step; stale state
        # from a previous step must not suppress the upstream all-gather.
        if pcp_manager is not None and isinstance(pcp_manager, AscendPCPManager):
            pcp_manager._sampling_hidden_restored = False
        self._restore_replicated_draft_target_states()

        # Peek before super() pops it: these refs stay valid after the pop.
        state = self.execute_model_state
        input_batch = getattr(state, "input_batch", None) if state is not None else None
        finished_req_ids = getattr(state, "finished_req_ids", None) if state is not None else None
        # Cleared each step; filled by postprocess_sampled during sample_fn.
        self._rg_spec_sampled_tokens = None
        self._rg_spec_num_sampled = None
        # Capture bound super() before nested sample_fn (zero-arg super fails in nested fn).
        _super_sample_tokens = super().sample_tokens

        # ``__new__`` UTs omit ``runtime_guard``; keep the pre-guard sample path.
        runtime_guard = getattr(self, "runtime_guard", None)
        if runtime_guard is None:
            return _super_sample_tokens(grammar_output)

        def sample_fn() -> SamplePhaseResult:
            with (
                wrap_compute_logits_for_pre_sample(self, input_batch)
                if need_pre_sample_hook(runtime_guard)
                else nullcontext()
            ):
                output = _super_sample_tokens(grammar_output)

            req_ids = list(getattr(input_batch, "req_ids", None) or [])
            sampled = self._rg_spec_sampled_tokens
            # AsyncOutput.sampled_token_ids is padded D2H numpy until get_output()
            # trims with num_sampled. run_sample_phase defers check_after_sample
            # for AsyncModelRunnerOutput; AscendAsyncOutput.get_output appends
            # the trimmed rows (W2-3 / D-11 token_repeat false positive).
            return SamplePhaseResult(
                scheduler_output=getattr(self, "_rg_scheduler_output", None),
                input_batch=input_batch,
                model_runner_output=output,
                sampler_output=SimpleNamespace(sampled_token_ids=sampled),
                valid_sampled_token_ids=getattr(output, "sampled_token_ids", None),
                req_ids_output_copy=req_ids,
                invalid_req_indices=None,
                finished_req_ids=finished_req_ids,
            )

        def accepted_token_nums_fn(_result: SamplePhaseResult):
            return self._rg_spec_num_sampled

        use_async = bool(getattr(self, "use_async_scheduling", False))
        speculative_config = getattr(self, "speculative_config", None)
        result, _ = runtime_guard.run_sample_phase(
            sample_fn=sample_fn,
            speculative_config=speculative_config,
            need_accepted_tokens=False,
            use_async=use_async,
            accepted_token_nums_fn=accepted_token_nums_fn
            if speculative_config is not None
            else None,
        )

        if self.use_spec_pp and self.is_last_pp_rank:
            assert self.pp_handler is not None
            self.pp_handler.broadcast_drafts()

        output = result.model_runner_output
        if runtime_guard.needs_sample_phase_hooks():
            output = maybe_wrap_v2_async_output(output, self)
        return output

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(
                kv_cache_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config
                self.pcp_manager.kv_cache_config = kv_cache_config
                self.model_state.pcp_manager = self.pcp_manager
                if self.speculator is not None:
                    self.speculator.pcp_manager = self.pcp_manager

        # Only target-model layers determine whether FIA is in use. This flag
        # is used for adaptive verification handling.
        draft_layer_names: set[str] = getattr(self.speculator, "draft_attn_layer_names", set())
        self.use_fia = any(
            (group.backend is AscendAttentionBackend or group.backend is AscendMLABackend)
            and any(layer_name not in draft_layer_names for layer_name in group.layer_names)
            for groups in self.attn_groups
            for group in groups
        )

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

        self.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            static_forward_context=self.compilation_config.static_forward_context,
        )
        self.model_state.kvpp_runtime = self.kvpp

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ):
        self._cpp_execution_time_ms = None
        profiling_config = self.ascend_config.scheduler_config.profiling_chunk_config
        execution_start_time = _start_profiling_chunk_timing(
            profiling_config,
            scheduler_output,
        )

        # Preemption stores must complete before the parent updates states and
        # reuses or zeroes the preempted requests' physical KV cache blocks.
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        # v1 parity: dummy waves never arm — manual_dump is not burned and the
        # wave counter does not advance (advance(allow_arm=False) is a no-op).
        # ``__new__`` UTs omit ``runtime_guard``; skip hooks when unbound.
        runtime_guard = getattr(self, "runtime_guard", None)
        allow_arm = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0
        if runtime_guard is not None:
            runtime_guard.sync_for_step(
                scheduler_output=scheduler_output,
                allow_arm=allow_arm,
            )
        self._rg_scheduler_output = scheduler_output

        try:
            self.model_state.kvpp_is_dummy_run = dummy_run or is_profile
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                context_len=context_len,
                valid_dummy_state_slots=valid_dummy_state_slots,
            )
            self.model_state.kvpp_is_dummy_run = False
            self.kvpp.complete_forward()
        finally:
            # Collectives must stay lockstep — do not soft-fail this gate.
            # No-sample / early return: do not burn manual_dump.
            if runtime_guard is not None and (dummy_run or self.execute_model_state is None):
                runtime_guard.end_of_wave_sync(allow_arm=False)

        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_config,
            execution_start_time,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                # Use a call-scoped bypass because skip_compiled would require runner-specific ForwardContext plumbing.
                with disable_compilation(self.get_model()):
                    self._dummy_run(mc2_tokens_capacity, skip_attn=True, skip_eplb=True, is_profile=True)
            super().profile_run()

    def gather_batch_req_state(self, scheduler_output: SchedulerOutput, dummy_run: bool):
        batch_state, uniform_token_count = super().gather_batch_req_state(scheduler_output, dummy_run)
        if batch_state is not None and is_pd_decode_recompute_scheduler_enabled(self.vllm_config):
            pd_decode_recompute = (
                batch_state.is_prefilling_np
                & (batch_state.num_computed_prefill_tokens_np > 0)
                & (batch_state.num_scheduled_tokens == self.decode_query_len)
                & (
                    batch_state.num_computed_prefill_tokens_np + batch_state.num_scheduled_tokens
                    >= batch_state.prefill_len_np
                )
            )
            if np.any(pd_decode_recompute):
                batch_state.is_prefilling_np[pd_decode_recompute] = False
                batch_state = batch_state._replace(has_prefill=bool(batch_state.is_prefilling_np.any()))
                uniform_token_count = vllm_model_runner.get_uniform_decode_token_count(
                    len(batch_state.req_ids),
                    batch_state.num_tokens,
                    int(batch_state.num_scheduled_tokens.max()),
                    batch_state.has_prefill,
                )
        return batch_state, uniform_token_count

    def _check_finegrained_tp_graph_step(self, cg_mode: CUDAGraphMode) -> None:
        # Eager dispatch keeps per-rank token counts; the cross-DP o_proj/MLP exchanges would desync.
        if self._finegrained_tp_requires_graph and cg_mode == CUDAGraphMode.NONE:
            raise RuntimeError(
                "o_proj / MLP TP require every step on a captured graph: this step dispatched "
                "to eager, which desyncs the cross-DP HCCL collectives (mixed or oversized "
                "batch, a request-arrival step misclassified as prefill, or a full prefill "
                "scheduled locally — a request sent directly to the decode node)."
            )

    def prepare_inputs(  # type: ignore[misc]
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
    ) -> AscendInputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        self._check_finegrained_tp_graph_step(batch_desc.cg_mode)
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        assert num_tokens > 0

        req_ids = batch_req_state.req_ids

        self._update_seq_lens_cpu(scheduler_output, req_ids)

        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)

        num_valid_tokens = num_scheduled_tokens_np
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(
                [
                    num_toks - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for num_toks, i in zip(num_scheduled_tokens_np, req_ids)
                ],
                dtype=np.int32,
            )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens_np,
            num_valid_tokens,
            kv_cache_config=self.kv_cache_config,
        )

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

        adaptive_verification_manager = self.adaptive_verification
        adaptive_verification_active = (
            adaptive_verification_manager is not None and num_draft_tokens_per_req is not None
        )
        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
        if adaptive_verification_active:
            num_scheduled_tokens_np, cu_num_logits_np = adaptive_verification_manager.compact_batch(
                num_draft_tokens_per_req, num_scheduled_tokens_np, cu_num_logits_np
            )
        # Get query_start_loc.
        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens

        if batch_desc.cg_mode == CUDAGraphMode.FULL and not adaptive_verification_manager:
            # This is only required for vllm-ascend.
            query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_after_padding,
                num_reqs_padded,
                num_reqs,
                query_start_loc_np,
                batch_desc.cg_mode,
                batch_desc.num_reqs,
            )

        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)

        if adaptive_verification_active:
            cu_num_logits, query_start_loc, total_num_draft_tokens = adaptive_verification_manager.reallocate_drafts(
                req_ids, idx_mapping
            )
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens

            # Non-fia backends skip padding query boundary when using adaptive verification
            if self.use_fia:
                query_start_loc_np[: num_reqs + 1] = query_start_loc[: num_reqs + 1].cpu().numpy()
                query_start_loc_np[num_reqs + 1 :] = int(query_start_loc_np[num_reqs])

        if self.use_fia and adaptive_verification_manager:
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                query_start_loc_np, num_reqs_padded = self._pad_adaptive_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                )

            query_start_loc = self.input_buffers.query_start_loc
            async_copy_to_gpu(query_start_loc_np, out=query_start_loc)

        if draft_tokens:
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
            )

        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]
        self.eplb.set_batch_phase(batch_req_state.has_prefill)

        # Graph dispatch may classify a PD prompt-tail step as decode, but
        # its input still comes from all_token_ids rather than sampled tokens.
        # Keep input preparation tied to the actual prefill progress so the
        # prompt tail and MTP lookahead are populated even on a decode graph.
        if np.any(batch_req_state.num_computed_prefill_tokens_np < batch_req_state.prefill_len_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        if adaptive_verification_active and self.use_fia:
            self.input_buffers.seq_lens_np[:num_reqs] = seq_lens[:num_reqs].cpu().numpy()

        # Pad for full CUDA graph mode.
        self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

        dcp_local_seq_lens = None
        # Main computes DCP lengths in the inherited execute_model after PCP
        # partitioning (vLLM #55212).

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
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
        )

        # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
        # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = AscendInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            # TODO: only populated for R-SWA (not supported yet).
            prompt_lens=prompt_lens,
            # extra attributes for ascend npus.
            seq_lens_np=self.input_buffers.seq_lens_np,
            attn_state=attn_state,
        )
        # vLLM main (#53867) changed maybe_partition_pcp_batch to take the
        # whole batch descriptor instead of padded_num_tokens.
        input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(
            self.pcp_manager,
            input_batch,
            batch_desc=batch_desc,
        )

        # For mla/sfa, update cos/sin. Here is for execute_model.
        update_cos_sin(input_batch.positions)

        return input_batch

    def prepare_dummy_attn(
        self, input_batch: AscendInputBatch, valid_state_slots: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is None:
            return super().prepare_dummy_attn(
                input_batch,
                valid_state_slots=valid_state_slots,
            )
        block_tables, slot_mappings = self.pcp_manager.prepare_dummy_attn(input_batch)
        if valid_state_slots:
            # Match the upstream state-slot contract in the persistent PCP views.
            for block_table in block_tables:
                state_slots = torch.arange(1, block_table.shape[0] + 1, dtype=torch.int32, device=block_table.device)
                block_table[:, 0].copy_(state_slots)
        return block_tables, slot_mappings

    def _lmhead_tp_max_num_logits(self) -> int:
        """Logits row capacity shared by every rank of the lmhead-TP group.

        Derived purely from global config so all ranks compute the identical
        value, matching upstream's own logits capacity bound
        (``max_num_reqs * decode_query_len``, see StructuredOutputsWorker init).
        """
        return self.max_num_reqs * self.decode_query_len

    def sample(self, hidden_states, input_batch, grammar_output):
        """Override GPUModelRunner.sample for lmhead TP.

        The LM-head collectives span the whole group, so every rank must feed
        compute_logits the same number of rows: pad hidden states up to
        ``_lmhead_tp_max_num_logits()`` and trim the logits back before
        sampling. ``logits_indices`` stays real (the V2 sampler gathers
        penalties by it). prompt_logprobs is not supported with lmhead TP
        (same as V1).
        """
        if not lmhead_tp_enable():
            return super().sample(hidden_states, input_batch, grammar_output)

        num_logits = input_batch.logits_indices.shape[0]
        capacity = self._lmhead_tp_max_num_logits()
        # A mismatch would desync the LM-head all_gather/all_to_all across the
        # group and hang the collectives. Fail fast instead.
        assert num_logits <= capacity, (
            f"lmhead TP logits rows ({num_logits}) exceed the group-agreed capacity "
            f"({capacity} = max_num_reqs * decode_query_len); the capacity formula "
            "no longer matches upstream logits production."
        )

        sample_hidden_states = hidden_states[input_batch.logits_indices]
        if num_logits < capacity:
            sample_hidden_states = torch.nn.functional.pad(sample_hidden_states, (0, 0, 0, capacity - num_logits))
        logits = self.model.compute_logits(sample_hidden_states)
        logits = logits[:num_logits]

        # Dispatch tail mirrors GPUModelRunner.sample; refresh it on main bumps.
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        context_len: int = 0,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ):
        """Join LM-head TP before stepping EPLB on an idle DP rank."""
        # Adaptive verification profiles eager tail sizes after graph capture.
        # Use balanced dummy routing, as the initial memory profile does, so a
        # synthetic router hotspot cannot exhaust one EP rank during startup.
        profile_adaptive_tail = self.adaptive_verification is not None and context_len > 0
        if profile_adaptive_tail and self.ascend_config.xlite_graph_config.enabled:
            logger.warning_once(
                "Adaptive verification cost profiling with XLite enabled may "
                "produce inaccurate costs because balanced MoE profiling sets "
                "the profile-run marker, which makes XLite bypass its graph path."
            )
        load_balance_ctx = override_mrv2_in_profile_run(True) if profile_adaptive_tail else nullcontext()
        with load_balance_ctx:
            hidden_states, sample_hidden_states = super()._dummy_run(
                num_tokens,
                *args,
                skip_attn=skip_attn,
                uniform_decode=uniform_decode,
                context_len=context_len,
                skip_eplb=True,
                is_profile=is_profile,
                **kwargs,
            )
        if lmhead_tp_enable() and not is_profile and hidden_states is not None:
            dummy_indices = torch.zeros(
                self._lmhead_tp_max_num_logits(),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            self.model.compute_logits(hidden_states[dummy_indices])
        return hidden_states, sample_hidden_states

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Override GPUModelRunner.postprocess_sampled for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to copy num_computed_tokens back to cpu here.

        Also stash spec-accept stats for ``run_sample_phase`` /
        ``check_after_spec`` (called after ``sample_fn`` returns).
        """
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )
        self._rg_spec_sampled_tokens = sampled_tokens
        self._rg_spec_num_sampled = num_sampled

        # Without MTP, update_requests writes the shared NumPy/torch CPU state.
        if self.speculator is not None:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        # npu attention backend still need to use seq_lens_cpu,
        # we need to copy num_computed_tokens back to cpu.
        default_stream = torch.cuda.current_stream()
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
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # MTP needs D2H copy to get reverted num_computed_tokens after rejection.
        # req_states.num_computed_tokens_cpu shares storage with its NumPy view,
        # so this update also corrects the num_computed_tokens_np used by PCP.
        if self.speculator is not None:
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        descriptor_num_reqs = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs_padded
        # This checks query lengths, not request phase: short prefills can also
        # match. Graph dispatch is responsible for excluding incompatible prefills.
        has_uniform_decode_query_lens = np.all(np.diff(query_start_loc_np[: num_reqs + 1]) == self.decode_query_len)
        matches_uniform_decode_graph_shape = (
            has_uniform_decode_query_lens and num_tokens_padded == descriptor_num_reqs * self.decode_query_len
        )
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
            and not matches_uniform_decode_graph_shape
        ):
            num_reqs_padded = num_reqs
        else:
            # Preserve the captured request shape for uniform decode graphs.
            # GDN full graphs capture metadata at request granularity, so
            # collapsing all padded tokens into one request changes the graph
            # topology between capture and replay.
            num_reqs_padded = descriptor_num_reqs

        if has_uniform_decode_query_lens and num_tokens_padded == num_reqs_padded * self.decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * self.decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded

    def _pad_adaptive_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Pad adaptive query boundary to the captured FULL graph request shape."""
        last_loc = int(query_start_loc_np[num_reqs])
        num_padding_tokens = num_tokens_padded - last_loc
        num_padding_reqs = num_reqs_padded - num_reqs
        assert num_padding_tokens >= 0 and num_padding_reqs >= 0

        if num_padding_reqs == 0:
            if num_padding_tokens > 0:
                query_start_loc_np[num_reqs + 1] = num_tokens_padded
                num_reqs_padded += 1
            return query_start_loc_np, num_reqs_padded

        cumulative_padding = np.arange(1, num_padding_reqs + 1, dtype=np.int32) * num_padding_tokens // num_padding_reqs
        query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = last_loc + cumulative_padding
        return query_start_loc_np, num_reqs_padded


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    def factory(  # type: ignore[misc]
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: Any = None,  # vLLM main (#51700)
    ):
        return ModelAclGraphManager(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
