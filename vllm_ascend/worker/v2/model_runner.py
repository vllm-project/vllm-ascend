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
from copy import deepcopy
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
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.dp_utils import skip_dp_coordination
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
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

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
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.core.profiling_chunk_predictor import (
    _finish_profiling_chunk_timing,
    _start_profiling_chunk_timing,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    bind_dspark_context_receiver,
    configure_dspark_kv_transfer,
    find_dspark_context_connector,
    find_dspark_prefix_connector,
    get_pd_dspark_aux_layer_ids,
    send_dspark_prefill_kv,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import (
    apply_dspark_resident_kv_specs,
    get_resident_dspark_layer_names,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    allocate_kv_offload_topk_profile_buffers,
    init_sparse_kv_offload_manager,
)
from vllm_ascend.models.deepseek_v41.cache_config import uses_a5_packed_cache
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.utils import (
    is_deepseek_v41,
    is_pd_decode_recompute_scheduler_enabled,
    lmhead_tp_enable,
    lmhead_tp_max_num_logits,
    lmhead_tp_pad_rows,
    set_potential_max_tokens,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata
from vllm_ascend.worker.utils import disable_compilation
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_state,
    skip_ring_state_update,
)
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.kvpp import KVPPRuntime
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.pp_transport import (
    bypass_upstream_spec_pp_guard,
    resolve_spec_pp_support,
    restore_pp_after_upstream_init,
    use_legacy_spec_pp,
)
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import (
    prepare_v41_dummy_ring_state,
    prepare_v41_source_rope,
    torch_cuda_wrapper,
)


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    # vLLM #51718 overlays hybrid Attention/Mamba groups in one standardized
    # backing allocation. Ascend MRV2 preserves that layout in
    # allocate_kv_cache_main and exposes contiguous backend-specific views.
    supports_standardized_shared_kv_backing = True

    execute_model_state: ExecuteModelState | None
    max_num_reqs: int

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        self.sparse_kv_offload_manager = None
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
        # These FIA models need post-rejection host counts on every PP stage.
        # TODO: Remove this extra PP sync when FIA and its metadata builders
        # use device lengths instead of exact CPU lengths.
        self.sync_spec_pp_cpu_counts = (
            self.use_pp
            and self.num_speculative_steps > 0
            and self.model_config.architecture
            in (
                "KimiLinearForCausalLM",
                "KimiK3ForCausalLM",
                "KimiK3ForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
            )
        )
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
        self.eplb = AscendEPLBController(
            parallel_config,
            device,
            self.ascend_config.eplb_config if parallel_config.enable_eplb else None,
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
        self.pd_dspark_aux_layer_ids: tuple[int, ...] = ()
        self._dspark_prefill_progress: dict[str, tuple[str, int]] = {}
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

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        aux_layers = get_pd_dspark_aux_layer_ids(self.vllm_config)
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.pd_dspark_aux_layer_ids = aux_layers

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
        if (
            lmhead_tp_enable()
            and self.prompt_logprobs_worker is not None
            and self.prompt_logprobs_worker.uses_prompt_logprobs.any()
        ):
            # The prompt-logprobs worker issues a second compute_logits with
            # unpadded rows that desyncs the LM-head collectives and hangs.
            raise NotImplementedError("prompt_logprobs is not supported with lmhead TP.")

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
        output = super().sample_tokens(grammar_output)
        if self.use_spec_pp and self.is_last_pp_rank:
            assert self.pp_handler is not None
            self.pp_handler.broadcast_drafts()
        return output

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return apply_dspark_resident_kv_specs(
            super().get_kv_cache_spec(),
            self.vllm_config,
            self.speculator,
            sparse_offload_enabled=self.ascend_config.sparse_kv_offload_config.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            shared_kv_cache_layers=getattr(self, "shared_kv_cache_layers", None),
        )

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # Match V1's physical buffer plan without mutating the scheduler's
        # logical cache configuration. Allocation must honor zero-stride aliases.
        kv_cache_config = deepcopy(kv_cache_config)
        sparse_cfg = self.ascend_config.sparse_kv_offload_config
        configure_dspark_kv_transfer(
            self.vllm_config, self.speculator, kv_cache_config, is_last_pp_rank=self.is_last_pp_rank
        )
        resident_draft_names = get_resident_dspark_layer_names(
            self.vllm_config,
            self.speculator,
            sparse_offload_enabled=sparse_cfg.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            shared_kv_cache_layers=getattr(self, "shared_kv_cache_layers", None),
        )
        # P also needs persistent prompt draft KV until D acknowledges its
        # transfer. It must never alias target layerwise scratch buffers.
        persistent_draft_names = resident_draft_names | set(getattr(kv_cache_config, "dspark_draft_layer_names", ()))
        apply_layerwise_kv_cache_plan(kv_cache_config, self.vllm_config, excluded_layer_names=persistent_draft_names)
        if sparse_cfg.enabled:
            self.sparse_kv_offload_manager = init_sparse_kv_offload_manager(
                self.vllm_config, kv_cache_config, sparse_cfg
            )
            self.model_state._offload_live_req_ids = self.req_states.req_id_to_index
            self.model_state._offload_draft_layer_names = (
                getattr(self.speculator, "draft_attn_layer_names", set[str]()) - resident_draft_names
                if sparse_cfg.use_fused_copy_sfa
                else set()
            )
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(
                kv_cache_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config
                self.pcp_manager.kv_cache_config = kv_cache_config
                self.pcp_manager.global_input_buffers = self.input_buffers
                self.model_state.pcp_manager = self.pcp_manager
                if self.speculator is not None:
                    self.speculator.pcp_manager = self.pcp_manager
        bind_dspark_context_receiver(
            self.vllm_config,
            sparse_offload_enabled=sparse_cfg.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            max_requests=self.max_num_reqs,
        )
        if sparse_cfg.enabled and self.speculator is not None:
            # Resident MLA DSpark has no host-pool LRU or LIM tail state.
            self.model_state._offload_draft_attn_groups = (
                [] if resident_draft_names else getattr(self.speculator, "attn_groups", [])
            )
        if any(is_circular_kv_cache_spec(group.kv_cache_spec) for group in self.kv_cache_config.kv_cache_groups):
            from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor

            for module in self.model.modules():
                if isinstance(module, DeepseekV41Compressor) and module.ratio == 2:
                    module.prepare_ring_compressor(self.max_num_tokens, self.device)
        prepare_v41_source_rope(self)
        # Recreate along with KV initialization: profiling capture owns a
        # throwaway model state and must not leak event/buffer bindings.
        self.model_state.device_metadata = (
            TargetDeviceMetadata()
            if uses_a5_packed_cache() and self.model_config.architecture == "DeepseekV41ForCausalLM"
            else None
        )

        # Upstream has bound every local cache; publish sealed plans before
        # the worker can warm up, capture graphs or execute prefill requests.
        from vllm_ascend.ops.kda_state_copy_plan import initialize_kda_state_copy

        initialize_kda_state_copy(
            self.vllm_config.compilation_config.static_forward_context,
            self.vllm_config.scheduler_config.max_num_seqs,
        )

        # Only target-model layers determine whether FIA is in use. This flag
        # is used for adaptive verification handling.
        draft_layer_names: set[str] = getattr(self.speculator, "draft_attn_layer_names", set())
        self.use_fia = any(
            (group.backend is AscendAttentionBackend or group.backend is AscendMLABackend)
            and any(layer_name not in draft_layer_names for layer_name in group.layer_names)
            for groups in self.attn_groups
            for group in groups
        )

        # Legacy (pre-AuxOutput) R3 path; the getattr keeps MRv2 startable on a
        # vLLM lane where ModelConfig no longer exposes the flag.
        if getattr(self.model_config, "enable_return_routed_experts", False):
            self.init_routed_experts_capturer()

        self.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            static_forward_context=self.compilation_config.static_forward_context,
        )
        self.model_state.kvpp_runtime = self.kvpp

    def _register_sparse_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Bind host pools before the V2 KV connector registers its destinations."""
        manager = self.sparse_kv_offload_manager
        if manager is None:
            return
        manager.register_kv_caches(kv_caches)
        if not self.ascend_config.sparse_kv_offload_config.use_fused_copy_sfa:
            return

        from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl

        owners: dict[int, AscendSFAKVOffloadImpl] = {}
        for layer in self.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if not isinstance(impl, AscendSFAKVOffloadImpl):
                continue
            shared = impl.topk_indices_buffer
            if shared is None:
                continue
            key = shared.data_ptr()
            if impl.skip_topk:
                if key not in owners:
                    raise RuntimeError("fused_copy_sfa shared attention precedes its indexer owner")
                impl.lim_indexer_owner = owners[key]
            else:
                owners[key] = impl
        for layer_name in manager.offload_layer_names:
            layer = self.compilation_config.static_forward_context[layer_name]
            layer.impl.bind_copy_sfa_kv_cache(manager, layer_name)

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

        self.model_state.kvpp_is_dummy_run = dummy_run or is_profile
        dp_coordination_context = (
            skip_dp_coordination() if should_skip_allreduce_across_dp_group(self.vllm_config) else nullcontext()
        )
        forward_failed = True
        with dp_coordination_context:
            try:
                output = super().execute_model(
                    scheduler_output,
                    intermediate_tensors=intermediate_tensors,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    is_profile=is_profile,
                    context_len=context_len,
                    valid_dummy_state_slots=valid_dummy_state_slots,
                )
                forward_failed = False
            finally:
                finish_execution = getattr(self.model_state, "finish_execution", None)
                if finish_execution is not None:
                    finish_execution(failed=forward_failed)
        if not dummy_run and not is_profile:
            self._maybe_send_draft_kv(scheduler_output)
        self.model_state.kvpp_is_dummy_run = False
        if dummy_run and lmhead_tp_enable() and not is_profile and self.is_last_pp_rank:
            # lmhead TP: idle ranks never call sample(); join the target head
            # here at capacity, before _dummy_run replays the dummy propose.
            if self.execute_model_state is None:
                raise RuntimeError(
                    "lmhead TP dummy join expects execute_model_state published by the upstream dummy execute_model."
                )
            dummy_indices = torch.zeros(
                self._lmhead_tp_max_num_logits(),
                dtype=torch.int64,
                device=self.device,
            )
            self.model.compute_logits(self.execute_model_state.hidden_states[dummy_indices])
        self.kvpp.complete_forward()

        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_config,
            execution_start_time,
        )
        return output

    def _maybe_send_draft_kv(self, scheduler_output: SchedulerOutput) -> None:
        """Write draft KV on P, then let D pull the exact allocated cache pages."""
        if not getattr(self, "pd_dspark_aux_layer_ids", ()) or not self.is_last_pp_rank:
            return
        connector, metadata = find_dspark_context_connector(
            get_kv_transfer_group(), scheduler_output.kv_connector_metadata
        )
        requests = getattr(metadata, "requests", {})
        if not any(getattr(req_meta, "dspark_context_generation", None) for req_meta in requests.values()):
            return
        state = self.execute_model_state
        if state is None or state.aux_hidden_states is None:
            if scheduler_output.total_num_scheduled_tokens:
                raise RuntimeError("P produced no target auxiliary states for an active remote DSpark request")
            return
        prefix_store = find_dspark_prefix_connector(get_kv_transfer_group(), scheduler_output.kv_connector_metadata)
        send_dspark_prefill_kv(
            self.speculator,
            state.input_batch,
            state.aux_hidden_states,
            requests,
            connector,
            self._dspark_prefill_progress,
            getattr(scheduler_output, "finished_req_ids", ()) or (),
            prefix_connector=prefix_store[0] if prefix_store is not None else None,
            prefix_metadata=prefix_store[1] if prefix_store is not None else None,
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        sparse_cfg = self.ascend_config.sparse_kv_offload_config
        if sparse_cfg.enabled:
            allocate_kv_offload_topk_profile_buffers(self.get_kv_cache_spec(), self.vllm_config, sparse_cfg)
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
        graph_num_reqs = batch_desc.num_reqs
        global_graph_num_reqs = (
            self.pcp_manager.get_global_graph_num_reqs(batch_desc) if self.pcp_manager is not None else None
        )
        if global_graph_num_reqs is not None:
            graph_num_reqs = global_graph_num_reqs
            num_tokens_after_padding = global_graph_num_reqs * batch_desc.uniform_token_count
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
        # Inputs are still global here; eager PCP descriptors carry local counts.
        num_reqs_padded = max(num_reqs, graph_num_reqs or 0)
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
                graph_num_reqs,
                uniform_query_len=batch_desc.uniform_token_count,
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
        self.input_buffers.seq_lens_np[num_reqs:] = 0

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
        block_tables, slot_mappings = super().prepare_dummy_attn(
            input_batch,
            valid_state_slots=valid_state_slots,
        )
        prepare_v41_dummy_ring_state(self, input_batch.num_reqs)
        slot_mappings = self._maybe_extend_slot_mappings(input_batch, slot_mappings)
        return block_tables, slot_mappings

    def _maybe_extend_slot_mappings(self, input_batch: AscendInputBatch, slot_mappings: torch.Tensor) -> torch.Tensor:
        if input_batch.num_tokens_after_padding > input_batch.num_tokens:
            # The parent already filled the entire persistent slot buffer.
            slot_mappings = self.block_tables.slot_mappings[:, : input_batch.num_tokens_after_padding]
        return slot_mappings

    def _lmhead_tp_max_num_logits(self) -> int:
        """Logits row capacity every rank agrees on (config-derived:
        ``max_num_reqs * decode_query_len``); drift desyncs and hangs."""
        return lmhead_tp_max_num_logits(self.max_num_reqs, self.decode_query_len)

    def sample(self, hidden_states, input_batch, grammar_output):
        """Override GPUModelRunner.sample for lmhead TP: every rank must feed
        compute_logits the same row count — pad up to ``_lmhead_tp_max_num_logits()``
        and trim back; prompt_logprobs stays unsupported (same as V1)."""
        if not lmhead_tp_enable():
            return super().sample(hidden_states, input_batch, grammar_output)

        num_logits = input_batch.logits_indices.shape[0]
        capacity = self._lmhead_tp_max_num_logits()
        # V1-style index pad: pad a private copy of the indices (input_batch keeps
        # the real ones; the V2 sampler gathers penalties by them) so one gather
        # feeds compute_logits the capacity rows; zero entries gather row 0.
        sample_indices = lmhead_tp_pad_rows(
            input_batch.logits_indices,
            capacity,
            "max_num_reqs * decode_query_len",
        )
        logits = self.model.compute_logits(hidden_states[sample_indices])
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

    @contextmanager
    def _cap_parallel_draft_dummy_reqs(self, uniform_decode: bool):
        # Profiling can exceed the speculator's query buffer.
        original_max_num_reqs = self.max_num_reqs
        if self.speculator is not None and not uniform_decode:
            # Other speculators use one query row per request in vLLM v0.30.0.
            query_width = getattr(self.speculator, "num_query_per_req", 1)
            self.max_num_reqs = min(original_max_num_reqs, self.max_num_tokens // query_width)
        try:
            yield
        finally:
            self.max_num_reqs = original_max_num_reqs

    @contextmanager
    def _preserve_dummy_query_tokens(self, num_tokens: int, uniform_decode: bool):
        dummy_query_tokens: int | None = None
        if (
            # FULL modes can also pad requests; only change token-only PIECEWISE padding.
            self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
            # PCP prepares and partitions its own dummy input layout.
            and self.pcp_manager is None
            # Hybrid models also need recurrent-state metadata to stay aligned.
            # TODO: Verify whether this hybrid-model guard can be removed.
            and not self.model_config.is_hybrid
        ):
            # Match the logical token count in the upstream dummy scheduler.
            dummy_query_tokens = max(num_tokens, self.decode_query_len) if uniform_decode else num_tokens

        previous_dummy_tokens = self.input_buffers.dummy_num_tokens
        self.input_buffers.dummy_num_tokens = dummy_query_tokens
        try:
            yield
        finally:
            self.input_buffers.dummy_num_tokens = previous_dummy_tokens

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
        """Balanced dummy routing for adaptive-verification profiling; EPLB
        steps via ``step_eplb_after`` (#17233). The lmhead TP join lives in the
        ``execute_model`` tail — joining after ``super()._dummy_run`` deadlocks."""
        skip_ring = bool(kwargs.pop("skip_gdn_state_update", False))
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
        with (
            # Preserve scheduled query tokens before PIECEWISE graph padding.
            self._preserve_dummy_query_tokens(num_tokens, uniform_decode),
            # TODO: Remove this context and its use after the next main2main
            # includes https://github.com/vllm-project/vllm/pull/56448.
            self._cap_parallel_draft_dummy_reqs(uniform_decode),
            skip_ring_state_update(skip_ring),
            load_balance_ctx,
        ):
            return super()._dummy_run(
                num_tokens,
                *args,
                skip_attn=skip_attn,
                uniform_decode=uniform_decode,
                context_len=context_len,
                skip_eplb=True,
                is_profile=is_profile,
                **kwargs,
            )

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
            self.use_pp
            and not self.is_last_pp_rank
            and isinstance(self.model_state, MambaHybridModelState)
            and self.cache_config.mamba_cache_mode == "align"
        ):
            # Deferred PP results belong to an older batch than input_block_tables.
            # Restore its rows before Mamba aligns the accepted recurrent state.
            # Postprocess skips -1 (freed/unsampled) rows; gather needs valid indices.
            self.block_tables.gather_block_tables(idx_mapping.clamp_min(0), idx_mapping.shape[0])
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        # TODO: Gate CPU length synchronization by backend requirements, not
        # speculative decoding alone. V4.1 uses device lengths; extend this
        # exemption to other backends that do not need exact CPU seq_lens.
        # Non-last PP stages receive rejections without owning a speculator.
        if (
            self.speculator is not None and not is_deepseek_v41(self.model_config.hf_config)
        ) or self.sync_spec_pp_cpu_counts:
            self._copy_num_computed_tokens_to_cpu()

    def postprocess_num_computed_tokens(self, input_batch: AscendInputBatch) -> None:
        super().postprocess_num_computed_tokens(input_batch)
        # Unsampled prefill chunks must also refresh the next step's snapshot.
        if self.sync_spec_pp_cpu_counts:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        # Attention metadata still needs exact CPU lengths. This non-blocking
        # D2H is waited on in _update_seq_lens_cpu, introducing a host/device
        # sync point that can break asynchronous scheduling overlap.
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

        # Speculative decoding needs corrected num_computed_tokens after rejection.
        # req_states.num_computed_tokens_cpu shares storage with its NumPy view,
        # so this update also corrects the num_computed_tokens_np used by PCP.
        if (
            self.speculator is not None and not is_deepseek_v41(self.model_config.hf_config)
        ) or self.sync_spec_pp_cpu_counts:
            # Blocks CPU submission until D2H completes; may stall the async pipeline.
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]

        # Without a CPU consumer, retain the upstream optimistic upper bound.
        # prepare_pos_seq_lens still reads exact rejection-corrected NPU state.
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
        uniform_query_len: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        descriptor_num_reqs = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs_padded
        query_len = uniform_query_len or self.decode_query_len
        # This checks query lengths, not request phase: short prefills can also
        # match. Graph dispatch is responsible for excluding incompatible prefills.
        has_uniform_decode_query_lens = np.all(np.diff(query_start_loc_np[: num_reqs + 1]) == query_len)
        matches_uniform_decode_graph_shape = (
            has_uniform_decode_query_lens and num_tokens_padded == descriptor_num_reqs * query_len
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

        if has_uniform_decode_query_lens and num_tokens_padded == num_reqs_padded * query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * query_len + last_loc
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
