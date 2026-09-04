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

import inspect
import os
from contextlib import contextmanager

import numpy as np
import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import (
    ExecuteModelState,
    GPUModelRunner,
    sort_batch_req_ids,
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
from vllm_ascend.core.profiling_chunk_predictor import (
    _finish_profiling_chunk_timing,
    _start_profiling_chunk_timing,
)
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.utils import lmhead_tp_enable, set_potential_max_tokens, vllm_version_is

if not vllm_version_is("0.27.1"):
    from vllm.v1.worker.gpu.model_runner import BatchReqState

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


@contextmanager
def _use_ascend_pcp_manager_for_vllm_0271():
    """Make the legacy vLLM PCP builder instantiate the Ascend manager.

    vLLM 0.27.1 hard-codes ``PCPManager`` inside
    ``maybe_build_pcp_manager``. Newer versions accept a manager class from
    the model runner, so this compatibility shim is needed only while the
    legacy builder runs.
    """
    if not vllm_version_is("0.27.1"):
        yield
        return

    # Patch the exact module object captured by GPUModelRunner. vLLM 0.27.1
    # resolves PCPManager through this alias inside initialize_kv_cache().
    pcp_module = vllm_model_runner.pcp
    original_pcp_manager_cls = pcp_module.PCPManager
    pcp_module.PCPManager = AscendPCPManager
    try:
        yield
    finally:
        pcp_module.PCPManager = original_pcp_manager_cls


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    execute_model_state: ExecuteModelState | None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        # FusedMoE can be constructed by the parent initializer and reads this
        # capacity while setting up MC2 communication.
        set_potential_max_tokens(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Decode context parallelism is not supported by Ascend NPU model runner v2.")

        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # V2 now receives the Ascend hardware-aware proposal length through
        # the scheduler side channel.  The upstream adaptive-verification
        # manager would make a second, independent budget decision and run a
        # duplicate compaction path, so disable only that manager for the
        # hardware-aware DSpark path.  DSpark's own confidence head remains
        # enabled and is consumed by Ascend's scheduler.
        dynamic_config = getattr(self.ascend_config, "dynamic_spec_config", None)
        self._hardware_aware_dspark = bool(
            dynamic_config is not None
            and dynamic_config.method == "dspark"
            and dynamic_config.policy == "hardware_aware"
        )
        self._v2_hardware_k_log_count = 0
        upstream_adaptive_verification = getattr(
            self, "adaptive_verification", None
        )
        if self._hardware_aware_dspark and upstream_adaptive_verification is not None:
            logger.info(
                "V2 hardware-aware DSpark is using Ascend DynamicSpecScheduler; "
                "disable upstream adaptive-verification compaction to avoid "
                "duplicate budget decisions."
            )
            self.adaptive_verification = None
        self.use_spec_pp = (
            self.use_pp and self.speculative_config is not None and self.speculative_config.method == "mtp"
        )

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
        if self.speculative_config is not None and (not self.use_spec_pp or self.is_last_pp_rank):
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
                install_spec_pp_token_broadcast,
            )

            assert self.pp_handler is not None
            install_spec_pp_token_broadcast(self.pp_handler, self.req_states)
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # we need to copy num_computed_tokens back to cpu to help
        # update actual seq_lens_cpu. gpu attention backend doesn't need these
        # attributes, cause their attention backends doesn't use seq_lens_cpu.
        # and seq_lens_cpu is deprecated in gpu_model_runner_v2.
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

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def sample_tokens(self, grammar_output):
        output = super().sample_tokens(grammar_output)

        if self.use_spec_pp and self.is_last_pp_rank:
            assert self.pp_handler is not None
            # Wait until propose() has populated this step's draft tokens.
            self.pp_handler.broadcast_draft_tokens()

        # Publish the policy result after the parent has completed the draft
        # step.  The scheduler consumes this field for the next iteration.
        # Keep this in the same override as the PP broadcast so the latter is
        # not shadowed by a second ``sample_tokens`` definition below.
        if self.is_last_pp_rank and self.speculator is not None and output is not None:
            dynamic_spec = getattr(self.speculator, "dynamic_spec", None)
            input_batch = getattr(self.speculator, "input_batch", None)
            if dynamic_spec is not None and input_batch is not None:
                self.speculator.update_dynamic_spec_for_v2(input_batch)
                lengths = dynamic_spec.proposal_lengths_for_v2(
                    input_batch.req_ids[: input_batch.num_reqs],
                    max_k=self.num_speculative_steps,
                )
                if lengths is not None:
                    model_runner_output = getattr(output, "model_runner_output", output)
                    # ``patch_pp_mtp`` adds this field on older vLLM checkouts.
                    setattr(model_runner_output, "proposal_lengths", lengths)
                    if self._v2_hardware_k_log_count < 8:
                        logger.warning(
                            "V2 hardware-aware K publish #%d: reqs=%d lengths=%s",
                            self._v2_hardware_k_log_count + 1,
                            len(lengths),
                            lengths,
                        )
                        self._v2_hardware_k_log_count += 1
        return output

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        # The custom Ascend hardware-aware scheduler owns the dynamic-K
        # decision. Hide only the upstream manager during graph/KV
        # initialization for that path to avoid duplicate compaction and to
        # keep the target FULL graph metadata fixed. For the community
        # confidence-scheduled path, keep the upstream manager visible so the
        # PR #15098 PIECEWISE setup is used.
        hardware_aware_dspark = bool(
            getattr(self, "_hardware_aware_dspark", False)
        )
        adaptive_verification = getattr(self, "adaptive_verification", None)
        if hardware_aware_dspark:
            self.adaptive_verification = None
        try:
            with (
                graph_manager_wrapper(self),
                adaptive_verification_gate_wrapper(self),
                _use_ascend_pcp_manager_for_vllm_0271(),
            ):
                super().initialize_kv_cache(kv_cache_config)
                if self.pcp_manager is not None:
                    assert isinstance(self.pcp_manager, AscendPCPManager)
                    self.pcp_manager.vllm_config = self.vllm_config
                    self.model_state.pcp_manager = self.pcp_manager
        finally:
            if hardware_aware_dspark:
                self.adaptive_verification = adaptive_verification
        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

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
        if (
            not dummy_run
            and not is_profile
            and getattr(self, "adaptive_verification", None) is not None
            and self.update_stream is not None
        ):
            # Adaptive verification rewrites the shared per-step buffers
            # (query_start_loc / cu_num_logits / capacities) in prepare_inputs.
            # FULL-graph param updates for the previous step run on
            # ``update_stream`` and are never joined back into the main stream,
            # so fence here to keep those async reads/writes from racing with
            # the in-place reallocation of this step.
            torch.npu.current_stream().wait_stream(self.update_stream)
        profiling_config = self.ascend_config.scheduler_config.profiling_chunk_config
        execution_start_time = _start_profiling_chunk_timing(
            profiling_config,
            scheduler_output,
        )

        execute_kwargs = {
            "intermediate_tensors": intermediate_tensors,
            "dummy_run": dummy_run,
            "skip_attn_for_dummy_run": skip_attn_for_dummy_run,
            "is_profile": is_profile,
            "context_len": context_len,
        }
        if "valid_dummy_state_slots" in inspect.signature(
            super().execute_model
        ).parameters:
            execute_kwargs["valid_dummy_state_slots"] = valid_dummy_state_slots
        output = super().execute_model(scheduler_output, **execute_kwargs)

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
                self._dummy_run(mc2_tokens_capacity, skip_attn=True, skip_eplb=True, is_profile=True)
            super().profile_run()

    if vllm_version_is("0.27.1"):

        def prepare_inputs(
            self,
            scheduler_output: SchedulerOutput,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
            npu attention backends need seq_lens_cpu to work.
            so we need to prepare seq_lens_cpu here.
            """
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0
            num_tokens_per_req = scheduler_output.num_scheduled_tokens
            num_reqs = len(num_tokens_per_req)

            req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            numtoks_iter = map(num_tokens_per_req.get, req_ids)
            num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
            num_valid_tokens = num_scheduled_tokens
            if scheduler_output.scheduled_spec_decode_tokens:
                num_valid_tokens = np.array(
                    [
                        num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                        for num_tokens, i in zip(num_scheduled_tokens, req_ids)
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
            idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
            idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
            idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

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

                max_expand_len = self.decode_query_len
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
                )

            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # This is only required for vllm-ascend.
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

            # Get prefill tokens if any.
            if batch_has_prefill:
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

            # Pad for full CUDA graph mode.
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

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
            seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
            np.add(
                self.req_states.num_computed_tokens_np[idx_mapping_np],
                num_scheduled_tokens,
                out=seq_lens_cpu_upper_bound_np[:num_reqs],
            )
            seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
            num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

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
                num_scheduled_tokens=num_scheduled_tokens,
                num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding,
                num_draft_tokens=total_num_draft_tokens,
                num_draft_tokens_per_req=num_draft_tokens_per_req,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                is_prefilling_np=is_prefilling_np,
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=prefill_len_np,
                num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
                max_seq_len_np=max_seq_len_np,
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

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

    else:

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
            num_tokens = batch_req_state.num_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0

            req_ids = batch_req_state.req_ids

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
            idx_mapping_np = batch_req_state.idx_mapping_np
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
            num_reqs = len(req_ids)

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

            # Draft compaction changes only the speculative suffix. Preserve
            # the original target-token count for Ascend phase classification;
            # it is the number of scheduled tokens minus the scheduler's full
            # draft suffix, not the compacted draft budget.
            num_valid_tokens = num_scheduled_tokens_np
            if draft_tokens:
                num_valid_tokens = np.array(
                    [
                        num_toks - len(draft_tokens.get(i, []))
                        for num_toks, i in zip(num_scheduled_tokens_np, req_ids)
                    ],
                    dtype=np.int32,
                )

            num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
            adaptive_verification = (
                self.adaptive_verification
                if num_draft_tokens_per_req is not None
                else None
            )
            if adaptive_verification is not None:
                # The scheduler stores the original, evenly distributed draft
                # budget. Compact it before building Ascend attention state so
                # the CPU-side phase classification sees the same target/draft
                # split as the V2 input batch.
                num_scheduled_tokens_np, cu_num_logits_np = (
                    adaptive_verification.compact_batch(
                        num_draft_tokens_per_req,
                        num_scheduled_tokens_np,
                        cu_num_logits_np,
                    )
                )
            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = self.input_buffers.query_start_loc_cpu.numpy()
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
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

            if adaptive_verification is not None:
                # Reallocate the compacted draft prefix on device. This keeps
                # the target logits and query layout consistent with the
                # adaptive verification budget selected in gather_batch_req_state.
                cu_num_logits, query_start_loc, total_num_draft_tokens = (
                    adaptive_verification.reallocate_drafts(req_ids, idx_mapping)
                )
                total_num_logits = (
                    num_reqs * self.model_state.num_new_sampled_tokens_per_step
                    + total_num_draft_tokens
                )

                if num_reqs_padded > num_reqs:
                    # reallocate_drafts() updates the real request boundaries and
                    # fills the remaining entries with the compacted token count.
                    # FIA, however, consumes the FULL-graph padded boundary as the
                    # last TND sequence length. Restore only the padding suffix;
                    # the dynamic per-request boundaries must remain untouched.
                    async_copy_to_gpu(
                        query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1],
                        out=query_start_loc[num_reqs + 1 : num_reqs_padded + 1],
                    )
                # Ascend attention metadata is CPU-driven. Reflect the
                # device-side reallocation back to the CPU mirror so the
                # per-request query boundaries and graph input remain
                # consistent after confidence-based compaction.
                self.input_buffers.query_start_loc_cpu.copy_(self.input_buffers.query_start_loc)
                query_start_loc_np = self.input_buffers.query_start_loc_cpu.numpy()

            if draft_tokens:
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
                )

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = query_start_loc[: num_reqs_padded + 1]
            self.eplb.set_batch_phase(batch_req_state.has_prefill)

            attn_state = build_attn_state(
                self.vllm_config,
                self.input_buffers.seq_lens_np,
                num_reqs,
                num_scheduled_tokens_np,
                num_valid_tokens,
            )

            # Get prefill tokens if any.
            if batch_req_state.has_prefill:
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

            # Pad for full CUDA graph mode.
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

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

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

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
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=batch_req_state.prefill_len_np,
                num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
                is_prefilling_np=batch_req_state.is_prefilling_np,
                has_prefill=batch_req_state.has_prefill,
                max_seq_len_np=max_seq_len_np,
                input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                positions=self.input_buffers.positions[:num_tokens_after_padding],
                is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices,
                cu_num_logits=cu_num_logits,
                cu_num_logits_np=cu_num_logits_np,
                has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                # Only populated when adaptive verification compacts the CPU
                # batch below the per-request upper bounds.
                max_query_len=(
                    int(num_scheduled_tokens_upper_bound.max())
                    if adaptive_verification is not None
                    else None
                ),
                # TODO: only populated for R-SWA (not supported yet).
                prompt_lens=prompt_lens,
                # extra attributes for ascend npus.
                seq_lens_np=self.input_buffers.seq_lens_np,
                attn_state=attn_state,
            )

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

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
        """Join the LM-head collectives on dummy batches for lmhead TP.

        Idle DP ranks never call sample(), so without this their ranks would
        be missing from the group collectives and busy ranks would hang.
        Zero-indexed rows at the same capacity as sample() (both from
        ``_lmhead_tp_max_num_logits()``; a mismatch hangs). Skipped for
        profiling and non-last PP ranks. Draft-side alignment is not covered.
        """
        hidden_states, sample_hidden_states = super()._dummy_run(
            num_tokens,
            *args,
            skip_attn=skip_attn,
            uniform_decode=uniform_decode,
            skip_eplb=skip_eplb,
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
        """
        if (
            getattr(self, "adaptive_verification", None) is not None
            and int(os.environ.get("VLLM_ASCEND_DEBUG_ADAPTIVE_OOB", "0")) > 0
        ):
            self._debug_probe_adaptive_post_update(
                idx_mapping,
                sampled_tokens,
                num_sampled,
                num_rejected,
                query_start_loc,
            )
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        # Skip D2H copy without MTP: num_computed_tokens_cpu is synced
        # from num_computed_tokens_np in _update_seq_lens_cpu instead.
        if self.speculator is not None:
            self._copy_num_computed_tokens_to_cpu()

    def _debug_probe_adaptive_post_update(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Temporary ADAPTIVE-verification probe (remove after triage).

        Replicates every address ``vllm input_batch._post_update_kernel``
        touches and logs the first step where any goes out of bounds, so the
        device-side aivec fault can be tied to concrete per-request values.
        Gated by ``VLLM_ASCEND_DEBUG_ADAPTIVE_OOB``.
        """
        rows = idx_mapping.shape[0]
        ns_len = num_sampled.shape[0]
        nr_len = num_rejected.shape[0]
        token_row = sampled_tokens.shape[1] if sampled_tokens.ndim == 2 else 1
        all_token_row = self.req_states.all_token_ids.gpu.shape[1]
        num_req_states = self.req_states.num_computed_tokens.gpu.shape[0]
        qsl_len = query_start_loc.shape[0] if query_start_loc is not None else 0

        torch.accelerator.synchronize()
        idx = idx_mapping.cpu().numpy().reshape(-1)
        ns = num_sampled.cpu().numpy().reshape(-1)
        nr = num_rejected.cpu().numpy().reshape(-1)
        qsl = query_start_loc.cpu().numpy().reshape(-1) if query_start_loc is not None else None
        smp = sampled_tokens.cpu().numpy()
        total_len = self.req_states.total_len.gpu.cpu().numpy()
        num_computed = self.req_states.num_computed_tokens.gpu.cpu().numpy()

        obc_shape = None
        penalties = getattr(self.sampler, "penalties_state", None)
        if penalties is not None:
            obc_t = penalties.output_bin_counts
            if obc_t is not None:
                obc_shape = obc_t.shape

        step = getattr(self, "_dbg_probe_step", 0) + 1
        self._dbg_probe_step = step

        issues = []
        for r in range(rows):
            req = int(idx[r])
            if req < 0:
                continue
            if not (0 <= req < num_req_states):
                issues.append(
                    (r, req, f"idx_mapping out of req-state range [0,{num_req_states})")
                )
                continue
            if r >= ns_len or r >= nr_len:
                issues.append(
                    (
                        r,
                        req,
                        f"row {r} beyond sampler arrays "
                        f"(ns_len={ns_len} nr_len={nr_len})",
                    )
                )
                continue
            if qsl is None or r + 1 >= qsl_len:
                issues.append(
                    (r, req, f"query_start_loc too short (qsl_len={qsl_len})")
                )
                continue
            s = int(ns[r])
            j = int(nr[r])
            qlen = int(qsl[r + 1]) - int(qsl[r])
            total = int(total_len[req])
            if s < 0 or s > token_row:
                issues.append(
                    (r, req, f"num_sampled={s} outside row width {token_row}")
                )
                continue
            if total < 0 or total + s > all_token_row:
                issues.append(
                    (
                        r,
                        req,
                        f"all_token_ids overflow: total_len={total} + "
                        f"num_sampled={s} > row {all_token_row}",
                    )
                )
            delta = qlen - j
            if num_computed[req] + delta < 0:
                issues.append(
                    (
                        r,
                        req,
                        f"num_computed underflow: nc={int(num_computed[req])} "
                        f"delta=qlen({qlen})-num_rejected({j})={delta}",
                    )
                )
            if obc_shape is not None and s > 0:
                vocab = obc_shape[1]
                for i in range(s):
                    tid = int(smp[r, i])
                    if not (0 <= tid < vocab):
                        issues.append(
                            (
                                r,
                                req,
                                f"output_bin_counts token_id={tid} out of "
                                f"[0,{vocab}) at sampled pos {i}",
                            )
                        )
                        break
            if step <= 3 and r < 8:
                toks = [int(t) for t in smp[r, :s]] if s else []
                logger.warning(
                    "[OOB-PROBE step=%d row=%d req=%d] qlen=%d ns=%d nr=%d "
                    "total=%d tokens=%s",
                    step,
                    r,
                    req,
                    qlen,
                    s,
                    j,
                    total,
                    toks,
                )

        logger.warning(
            "[OOB-PROBE step=%d] rows=%d issues=%d", step, rows, len(issues)
        )
        for r, req, msg in issues:
            logger.warning("[OOB-PROBE step=%d row=%d req=%d] %s", step, r, req, msg)

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
        # Without MTP, num_computed_tokens_np is already correct from update_requests.
        if self.speculator is not None:
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]
        else:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.req_states.num_computed_tokens_np[req_index]

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
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        ):
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        # A hardware-aware V2 draft can select a smaller uniform physical K
        # and therefore a smaller query width than the maximum configured
        # decode width.  Infer that width from the selected FULL graph bucket;
        # mixed batches keep the existing dummy-row path.
        runtime_query_len = self.decode_query_len
        if (
            num_reqs_padded > 0
            and num_tokens_padded % num_reqs_padded == 0
        ):
            runtime_query_len = num_tokens_padded // num_reqs_padded

        # A dynamic-K batch can have the same total size as a uniform graph
        # bucket while its request boundaries are uneven (e.g. 46 tokens in an
        # 8-request, 6-token graph).  Use the actual boundary to distinguish
        # that case; otherwise the TND metadata ends at the compacted count
        # instead of the graph's padded token count.
        is_uniform_batch = (
            num_tokens_padded == num_reqs_padded * runtime_query_len
            and query_start_loc_np[num_reqs] == num_reqs * runtime_query_len
        )

        if is_uniform_batch:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * runtime_query_len
                + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    if vllm_version_is("0.27.1"):

        def factory(
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
        ):
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
            )

    else:

        def factory(  # type: ignore[misc]
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
            varlen_decode: bool = False,
        ):
            resolved_mode = cudagraph_mode
            resolved_varlen = varlen_decode
            if getattr(model_runner, "adaptive_verification", None) is not None:
                # Plan A: Ascend attention backends report UNIFORM_BATCH and
                # cannot capture varlen FULL decode graphs (upstream demands
                # ALWAYS). Adaptive verification therefore runs the target
                # decode through PIECEWISE/dynamic graphs instead.
                resolved_mode = CUDAGraphMode.PIECEWISE
                resolved_varlen = False
                logger.warning(
                    "Adaptive verification on Ascend: running target decode "
                    "through PIECEWISE graphs (attention backends report "
                    "UNIFORM_BATCH, no varlen FULL decode capture)."
                )
            elif resolved_varlen:
                # The custom Ascend hardware-aware DSpark path disables the
                # upstream manager to avoid duplicate K decisions. Its target
                # graph still cannot use varlen FIA metadata, so retain the
                # fixed-width graph path and let physical-K be handled by the
                # DSpark graph manager.
                resolved_varlen = False
            return ModelAclGraphManager(
                vllm_config,
                device,
                resolved_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=resolved_varlen,  # type: ignore[call-arg]
            )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager


@contextmanager
def adaptive_verification_gate_wrapper(model_runner):
    """Relax the upstream ``AttentionCGSupport.ALWAYS`` requirement on Ascend.

    Upstream adaptive verification captures varlen FULL decode graphs, so its
    factory refuses to create the manager unless every attention builder
    reports ``AttentionCGSupport.ALWAYS`` (``adaptive_verification.py``).
    Ascend attention backends only report ``UNIFORM_BATCH`` today, which would
    make ``enable_adaptive_verification=true`` fail at startup. Under Plan A
    the decode runs through PIECEWISE graphs (see ``graph_manager_wrapper``),
    so the ALWAYS hard gate is relaxed here while every other upstream
    validation (device/CPU query-len mismatch support, etc.) still runs.
    """
    original_factory = getattr(
        vllm_model_runner, "maybe_create_adaptive_verification_manager", None
    )
    if original_factory is None:
        yield
        return

    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    def make_piecewise_manager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits,
    ) -> AdaptiveVerificationManager:
        """Plan-A manager whose cost curves seed from piecewise dummy runs.

        Upstream seeds its step-cost tables from FULL-decode-graph dummy runs
        (``full_cudagraph=True`` samples) and profile sizes derived from the
        captured full-graph token counts. Under Plan A the target decode runs
        through PIECEWISE graphs, so there are no full graphs to price: profile
        a representative grid of piecewise batch sizes instead and price the
        drafter curve from every sample (not only ``full_cudagraph`` ones).
        """

        class AscendPiecewiseAdaptiveManager(AdaptiveVerificationManager):
            def batches_to_profile(self, capture_sizes):
                del capture_sizes
                # No FULL graphs: leave ``_cudagraph_limit`` at 0 so the cost
                # tables stay smooth (nothing pads to a captured size).
                self._cudagraph_limit = 0
                max_num_tokens = self.req_states.max_num_batched_tokens
                base_size = max(1, self.num_speculative_steps + 1)
                grid = [base_size]
                while grid[-1] < max_num_tokens:
                    grid.append(min(grid[-1] * 2, max_num_tokens))
                from vllm import envs

                context_len = envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
                for num_tokens in grid:
                    for _ in range(3):
                        yield {
                            "num_tokens": num_tokens,
                            "context_len": context_len,
                        }

            def set_initial_cost_curves(self, samples):
                from collections import defaultdict

                def median_curve(points):
                    grouped: dict[int, list[float]] = defaultdict(list)
                    for key, value in points:
                        grouped[key].append(value)
                    return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

                draft_curve = median_curve(
                    (s.num_reqs, s.drafter_ms) for s in samples
                )
                verify_curve = median_curve(
                    (s.num_target_tokens, s.forward_ms) for s in samples
                )
                self.set_cost_curves(draft_curve, verify_curve)

        return AscendPiecewiseAdaptiveManager(
            req_states,
            query_start_loc,
            num_bonus_tokens,
            max_total_logits=max_total_logits,
        )

    def relaxed_factory(
        *,
        enable_adaptive_verification: bool,
        attn_groups,
        attn_cg_support,
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits,
    ):
        if not enable_adaptive_verification:
            return original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
            )
        try:
            original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
            )
        except ValueError as exc:
            # Only the ALWAYS requirement is relaxed on Ascend; any other
            # validation failure must keep failing loudly.
            if "AttentionCGSupport.ALWAYS" not in str(exc):
                raise
            logger.warning(
                "Relaxing the adaptive-verification AttentionCGSupport.ALWAYS "
                "gate for Ascend; decode runs through PIECEWISE graphs: %s",
                exc,
            )

        # The upstream factory can return a manager even when the Ascend graph
        # wrapper has redirected decode to PIECEWISE.  That manager prices the
        # drafter only from ``full_cudagraph`` samples, which leaves an empty
        # draft curve on Ascend and fails with "could not profile step costs".
        # Keep upstream validation above, but always use the Ascend manager so
        # both drafter and target curves are built from the PIECEWISE samples.
        logger.info(
            "Using Ascend PIECEWISE adaptive-verification manager for "
            "confidence-scheduled DSpark."
        )
        return make_piecewise_manager(
            req_states,
            query_start_loc,
            num_bonus_tokens,
            max_total_logits=max_total_logits,
        )

    try:
        vllm_model_runner.maybe_create_adaptive_verification_manager = relaxed_factory
        yield
    finally:
        vllm_model_runner.maybe_create_adaptive_verification_manager = original_factory
