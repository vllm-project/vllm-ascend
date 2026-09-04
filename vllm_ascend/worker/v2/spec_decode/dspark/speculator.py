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
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)

from vllm_ascend.models.qwen3_dspark import process_weight
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)

logger = init_logger(__name__)

DSPARK_AUX_HIDDEN_FORMAT_RAW = "raw"
DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED = "materialized"
_VALID_DSPARK_AUX_HIDDEN_FORMATS = {
    DSPARK_AUX_HIDDEN_FORMAT_RAW,
    DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED,
}


class AscendDSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        configured_format = self._configure_target_aux_hidden_state_format(
            target_model,
            draft_hf_config,
        )
        model = super().load_draft_model(target_model, target_attn_layer_names)
        model_format = getattr(model, "dspark_aux_hidden_state_format", None)
        if configured_format is None:
            self._configure_target_aux_hidden_state_format(target_model, model)
        elif model_format is not None and model_format != configured_format:
            raise ValueError(
                "DSpark auxiliary hidden-state format mismatch between config "
                f"({configured_format!r}) and model ({model_format!r})."
            )
        # Upstream load_dspark_model overrides the drafter's quant_config with
        # get_draft_quant_config (None for a bf16 drafter), so the drafter's
        # __init__ derives rotation_path=None and its fc projection is loaded
        # unrotated. The target is QuaRot-quantized, so the aux hidden states it
        # feeds the drafter are in rotated space; fc must be rotated (W @ R) to
        # project them back to model space.
        rotation_path = get_rotation_path(self.vllm_config)
        if rotation_path is not None and hasattr(model.model, "fc"):
            rotation_weight = get_rotation_matrix(rotation_path)
            fc = model.model.fc
            with torch.no_grad():
                fc.weight.data.copy_(process_weight(fc.weight.data.cpu(), rotation_weight))
        return model

    @staticmethod
    def _configure_target_aux_hidden_state_format(
        target_model: torch.nn.Module,
        format_provider: Any,
    ) -> str | None:
        """Apply a draft-declared auxiliary-hidden-state representation."""
        aux_hidden_format = getattr(
            format_provider,
            "dspark_aux_hidden_state_format",
            None,
        )
        if aux_hidden_format is None:
            architectures = getattr(format_provider, "architectures", ()) or ()
            model_type = getattr(format_provider, "model_type", None)
            if model_type == "qwen3" or (
                "Qwen3DSparkModel" in architectures
                or "DSparkDraftModel" in architectures
            ):
                aux_hidden_format = DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED
        if aux_hidden_format is None:
            return None
        if aux_hidden_format not in _VALID_DSPARK_AUX_HIDDEN_FORMATS:
            raise ValueError(
                "Unsupported DSpark auxiliary hidden-state format "
                f"{aux_hidden_format!r}; expected one of "
                f"{sorted(_VALID_DSPARK_AUX_HIDDEN_FORMATS)}."
            )

        set_capture_mode = getattr(
            target_model,
            "set_dspark_aux_capture_materialized",
            None,
        )
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(
                    get_language_model(),
                    "set_dspark_aux_capture_materialized",
                    None,
                )
        if set_capture_mode is None:
            raise RuntimeError(
                "DSpark auxiliary hidden-state format "
                f"{aux_hidden_format!r} was resolved, but target model "
                f"{type(target_model).__name__} has no capture-mode setter."
            )

        materialized = aux_hidden_format == DSPARK_AUX_HIDDEN_FORMAT_MATERIALIZED
        set_capture_mode(materialized)
        logger.info(
            "DSpark auxiliary hidden-state contract: draft=%s, target_capture=%s.",
            aux_hidden_format,
            "materialized" if materialized else "raw",
        )
        return aux_hidden_format

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
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
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        assert self.input_batch is not None
        # The draft attention metadata is built through the generic
        # (Ascend) build_attn_metadata path; the factory forwards the draft
        # query positions that the DSA metadata builder needs for RoPE.
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                torch.from_numpy(self.input_batch.is_prefilling_np),
            ),
        ):
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        return [self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)]

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
            metadata.actual_seq_lengths_q = query_lens_list
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
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions, self.max_num_tokens, torch.from_numpy(self.input_batch.is_prefilling_np)
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
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
