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

from contextlib import contextmanager

import numpy as np
import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
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
from vllm_ascend.worker.v2.pp_utils import (
    bypass_upstream_spec_pp_guard,
    resolve_spec_pp_support,
    restore_pp_after_upstream_init,
)
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import communicator_switch, torch_cuda_wrapper


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

        # Eagle3/DSpark drafters are rank-local. Hide PP from the upstream
        # initializer, then rebuild the skipped PP state.
        spec_pp_support = resolve_spec_pp_support(vllm_config)
        with torch_cuda_wrapper():
            with bypass_upstream_spec_pp_guard(vllm_config, spec_pp_support) as pp_disabled:
                super().__init__(vllm_config, device)
            if pp_disabled:
                restore_pp_after_upstream_init(self, vllm_config)
        self.use_spec_pp = spec_pp_support is not None
        # These draft heads consume target aux states collected across PP ranks.
        if spec_pp_support is not None and spec_pp_support.needs_aux_hidden_states:
            self.use_aux_hidden_state_outputs = True

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
        return output

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        with graph_manager_wrapper(self), _use_ascend_pcp_manager_for_vllm_0271():
            super().initialize_kv_cache(kv_cache_config)
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config
                self.model_state.pcp_manager = self.pcp_manager
        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    @torch.inference_mode()
    def capture_model(self) -> int:
        """Capture decoder and multimodal encoder graphs on Ascend."""
        if hasattr(self, "model_state") and self.model_state.supports_mm_inputs:
            encoder_runner = getattr(self.model_state, "encoder_runner", None)
            encoder_manager = getattr(encoder_runner, "cudagraph_manager", None)
            if encoder_manager is not None and self.update_stream is not None:
                encoder_manager.update_stream = self.update_stream

        # EncoderRunner uses vLLM's generic graph_capture(), whose communicator
        # type check expects the CUDA symbol. Temporarily bind that symbol to
        # NPUCommunicator, as is already done for decoder ACL graph capture.
        with torch_cuda_wrapper(), communicator_switch():
            return super().capture_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ):
        self._cpp_execution_time_ms = None
        profiling_config = self.ascend_config.scheduler_config.profiling_chunk_config
        execution_start_time = _start_profiling_chunk_timing(
            profiling_config,
            scheduler_output,
        )

        if vllm_version_is("0.27.1"):
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
        else:
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                context_len=context_len,
            )

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

            dcp_local_seq_lens = None
            if self.use_dcp:
                prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    self.input_buffers.seq_lens,
                    num_reqs,
                    self.dcp_size,
                    self.dcp_rank,
                    self.cp_interleave,
                )
                dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

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
                dcp_local_seq_lens=dcp_local_seq_lens,
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

            num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
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

            if draft_tokens:
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
                )

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = query_start_loc[: num_reqs_padded + 1]
            self.eplb.set_batch_phase(batch_req_state.has_prefill)

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

            dcp_local_seq_lens = None
            if self.use_dcp:
                prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    self.input_buffers.seq_lens,
                    num_reqs,
                    self.dcp_size,
                    self.dcp_rank,
                    self.cp_interleave,
                )
                dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

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
                dcp_local_seq_lens=dcp_local_seq_lens,
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

        if num_tokens_padded == num_reqs_padded * self.decode_query_len:
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
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,  # type: ignore[call-arg]
            )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
