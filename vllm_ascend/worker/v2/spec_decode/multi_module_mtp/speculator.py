# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
import torch
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import (
    AutoRegressiveAclGraphManager,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import (
    torch_gather_wrapper,
)


class AscendMultiModuleMTPSpeculator(MultiModuleMTPSpeculator):
    """Ascend multi-module MTP speculator.

    The whole draft loop (input rolling, per-module forward, sampling) is
    inherited from the upstream GPU implementation, including its Triton
    kernels (executed by triton-ascend). Ascend replaces the attention
    metadata construction and the CUDA-graph manager (ACL FULL graphs with
    UpdatableGraph attention-parameter updates on replay).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.input_batch: InputBatch | None = None
        del self.input_buffers
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.update_stream = None

    def _fixup_draft_model_config(self) -> None:
        # Transformers validates Step3.5 layer_types against num_hidden_layers
        # and may leave only the base-layer entries. The MTP draft model builds
        # layers at num_hidden_layers + k, so make the draft config cover those
        # layer indices before the model is constructed. This can be removed when
        # v0.33.0 is supported.
        hf_config = self.draft_model_config.hf_config
        layer_types = getattr(hf_config, "layer_types", None)
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        if layer_types is not None and num_hidden_layers is not None and num_mtp_layers:
            needed = num_hidden_layers + num_mtp_layers
            if len(layer_types) < needed:
                extension = ["sliding_attention"] * (needed - len(layer_types))
                hf_config.layer_types = list(layer_types) + extension
                target_hf_config = self.vllm_config.model_config.hf_config
                target_layer_types = getattr(target_hf_config, "layer_types", None)
                if target_layer_types is not None and len(target_layer_types) < needed:
                    target_hf_config.layer_types = list(target_layer_types) + extension

    def load_model(self, target_model: torch.nn.Module) -> None:
        self._fixup_draft_model_config()
        super().load_model(target_model)

    def _create_draft_vllm_config(self) -> VllmConfig:
        parallel_config = replace(
            self.vllm_config.parallel_config,
            pipeline_parallel_size=1,
        )
        draft_vllm_config = replace(
            self.vllm_config,
            parallel_config=parallel_config,
            model_config=self.draft_model_config,
            quant_config=get_draft_quant_config(self.vllm_config),
        )
        return draft_vllm_config

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        self.draft_vllm_config = self._create_draft_vllm_config()
        return load_eagle_model(target_model, self.draft_vllm_config)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Mirror the upstream mode selection: piecewise is unsupported for
        # the multi-module loop.
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        if cudagraph_mode.has_piecewise_cudagraphs():
            cudagraph_mode = (
                CUDAGraphMode.FULL_DECODE_ONLY if cudagraph_mode.has_full_cudagraphs() else CUDAGraphMode.NONE
            )
        self.cudagraph_manager = AutoRegressiveAclGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )
        # The Ascend manager needs this speculator to rebuild draft metadata
        # and FIA graph-update params on replay (mirrors
        # AscendTargetDependentARSpeculator.init_cudagraph_manager).
        self.cudagraph_manager.speculator = self
        self.cudagraph_manager.update_stream = self.update_stream
        # Backend class used to select the replay parameter-update path.
        self.attn_backend = _get_graph_update_backend(self.attn_groups)

    def capture(self) -> None:
        assert self.cudagraph_manager is not None
        if not self.cudagraph_manager.needs_capture():
            return
        # Reset indices to zeros to prevent stale values from prior dummy
        # runs from causing out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        if self.cudagraph_manager.use_breakable_cg:
            self.cudagraph_manager.init_breakable_cg_runner(self.model)
        self.cudagraph_manager.capture(
            self._generate_drafts,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing multi-module MTP ACL graphs",
        )

    def propose(self, input_batch: InputBatch, *args, **kwargs):
        self.input_batch = input_batch
        with (
            build_attn_metadata_wrapper(),
            torch_gather_wrapper(),
        ):
            return super().propose(input_batch, *args, **kwargs)

    def build_draft_attn_metadatas(
        self,
        num_reqs_padded: int,
        num_tokens_padded: int,
        is_draft_model_prefill: bool,
    ) -> list[dict]:
        """Replay hook for AutoRegressiveAclGraphManager.run_fullgraph.

        The multi-module draft loop shares a single attention metadata
        across all module forwards (upstream contract: one metadata built at
        step 0), so the metadata propose() just built for the dispatched
        padded batch covers the whole graph -- no per-step copies needed.
        """
        assert self.input_batch is not None
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs_padded,
        )
        with build_attn_metadata_wrapper():
            attn_metadata = self._build_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=self.input_batch.query_start_loc_np,
                seq_lens_cpu_upper_bound=self.input_batch.seq_lens_cpu_upper_bound,
                step=0,
            )
        assert attn_metadata is not None
        return [{name: metadata for name, metadata in attn_metadata.items() if name in self.draft_attn_layer_names}]

    def build_fia_params(
        self,
        num_reqs_padded: int,
        draft_attn_metadata: dict,
        is_draft_model_prefill: bool,
    ) -> list[dict]:
        """Per-layer FIA parameters for the UpdatableGraph task update.

        Every attention task registered at capture time matches its update
        parameters by layer name and requires exactly one entry (see
        UpdatableGraph.resolve_tasks), so emit one entry per draft attention
        layer. All layers share the same batch-level values because the
        three module forwards reuse one metadata.
        """
        fia_params = []
        for layer_name, metadata in draft_attn_metadata.items():
            fia_params.append(
                {
                    "layer_name": layer_name,
                    "actual_seq_lengths": metadata.actual_seq_lengths_q,
                    "actual_seq_lengths_kv": metadata.seq_lens_list,
                    "block_table": metadata.block_tables,
                }
            )
        return fia_params

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc,
        query_start_loc_np,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal=True,
        dcp_local_seq_lens=None,
        slot_mappings=None,
    ):
        assert self.input_batch is not None
        assert isinstance(self.input_batch, AscendInputBatch)
        is_prefilling = torch.from_numpy(self.input_batch.is_prefilling_np)
        if bool(is_prefilling[:num_reqs].any()):
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.DecodeOnly

        seq_lens_cpu = seq_lens_cpu_upper_bound[:num_reqs]
        with build_attn_metadata_factory(
            self.input_buffers.positions,
            batch_desc.num_tokens,
            is_prefilling,
            seq_lens_cpu=seq_lens_cpu,
            attn_state=attn_state,
            parallel_config=self.vllm_config.parallel_config,
        ):
            attn_metadata = super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                slot_mappings,
            )
        return attn_metadata
