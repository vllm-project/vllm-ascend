#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.dcp_utils import DCPManager
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import prepare_replicated_pcp_config


class AscendDSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        vllm_config, self.replicated_pcp = prepare_replicated_pcp_config(vllm_config)
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None
        self.attn_architecture: str | None = None
        self._init_dcp()

    def _init_dcp(self) -> None:
        self.use_dcp = self.attn_vllm_config.parallel_config.decode_context_parallel_size > 1
        self.dcp_manager: DCPManager | None = None
        if not self.use_dcp:
            return
        self.dcp_manager = DCPManager(
            dcp_world_size=self.attn_vllm_config.parallel_config.decode_context_parallel_size,
            dcp_rank=get_dcp_group().rank_in_group,
            max_buffer_num_tokens=self.max_num_tokens,
            max_num_reqs=self.max_num_reqs,
            device=self.device,
            vllm_config=self.attn_vllm_config,
            use_async_scheduling=self.vllm_config.scheduler_config.async_scheduling,
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if hasattr(model, "post_process"):
            with set_current_vllm_config(self.vllm_config):
                model.post_process(self.vllm_config)
        if hasattr(model, "configure_target_aux_hidden_capture"):
            model.configure_target_aux_hidden_capture(target_model)

        return model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        # Initialize the draft attention backend with its PCP=1 config.
        with set_current_vllm_config(self.attn_vllm_config):
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
            self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
            # npu needs attn_backends to update full graph params in run_fullgraph.
            attn_backends: dict[str, type[AttentionBackend]] = {}
            active_layer_names = self.draft_attn_layer_names
            for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
                layer_names = kv_cache_group_spec.layer_names
                if active_layer_names is not None:
                    # Preserve cache-group order so captured graph tasks and
                    # runtime metadata stay aligned.
                    layer_names = [name for name in layer_names if name in active_layer_names]

                layer_type = cast(type[Any], AttentionLayerBase)
                attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

                for layer_name in layer_names:
                    attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

            self.attn_backends = attn_backends
            backend = _get_graph_update_backend(self.attn_groups)
            if issubclass(backend, AscendMLABackend):
                self.attn_architecture = "MLA"
            elif issubclass(backend, AscendAttentionBackend):
                self.attn_architecture = "GQA"
            else:
                self.attn_architecture = None

    def _prepare_draft_dcp_metadata_inputs(
        self, num_reqs: int, num_reqs_padded: int, step: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        is_prefilling = torch.zeros(num_reqs_padded, dtype=torch.bool)
        if not self.use_dcp:
            return None, is_prefilling
        assert self.dcp_manager is not None
        return self.dcp_manager.prepare_draft_dcp_metadata_inputs(
            target_seq_lens_cpu=self.target_input_buffers.seq_lens_cpu,
            is_prefilling=is_prefilling,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            step=step,
            max_model_len=self.max_model_len,
        )

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        assert self.input_batch is not None
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(
            self.input_batch.num_reqs, num_reqs_padded, self.num_query_per_req
        )
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            attn_metadata = super()._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )

        if self.attn_architecture not in ("GQA", "MLA"):
            return [attn_metadata]

        return [self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)]

    def _build_draft_attn_metadata(self, *, num_reqs_padded, **kwargs):
        if self.attn_architecture not in ("GQA", "MLA"):
            return super()._build_draft_attn_metadata(num_reqs_padded=num_reqs_padded, **kwargs)

        # This kwargs["num_tokens_padded"] is only useful in eager/PIECEWISE.
        # TODO: Replace this temporary padding workaround with upstream #56181's
        # actual-token metadata and MLA input slicing for non-FULL execution.
        num_tokens_padded = kwargs["num_tokens_padded"]
        assert num_tokens_padded % self.num_query_per_req == 0, "Draft tokens must contain whole query groups"
        num_reqs_padded = num_tokens_padded // self.num_query_per_req

        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(
            kwargs["num_reqs"], num_reqs_padded, kwargs["step"]
        )
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            attn_metadata = super()._build_draft_attn_metadata(num_reqs_padded=num_reqs_padded, **kwargs)
        return self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        DSpark inherits DFlash's full-graph path, and upstream
        ``Speculator._build_draft_attn_metadata`` clamps ``query_start_loc`` at
        the real ``num_reqs`` to keep the cumulative series non-decreasing, so
        when a batch is padded to a capture size (``num_reqs_padded >
        num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            decode_metadata = metadata.decode if self.attn_architecture == "MLA" else metadata
            decode_metadata.actual_seq_lengths_q = query_lens_list
        return attn_metadata

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        sync_state = dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                self.max_num_tokens,
                torch.from_numpy(self.input_batch.is_prefilling_np),
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
