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
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
)
from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.worker.v2.spec_decode.dspark.greedy import (
    sample_greedy_markov,
    scratch_shape,
)
from vllm_ascend.utils import (
    get_rotation_path,
    vllm_version_is,
)
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_wrapper,
    build_draft_attn_metadata_factory,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import prepare_replicated_pcp_config


class AscendDSparkSpeculator(DSparkSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        vllm_config, self.replicated_pcp = prepare_replicated_pcp_config(
            vllm_config
        )
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None

        additional_config = vllm_config.additional_config or {}
        self._enable_dspark_fused_greedy = validate_additional_config_bool(
            additional_config.get("enable_dspark_fused_greedy", False),
            "additional_config.enable_dspark_fused_greedy",
        )

        self._greedy_partial_values: torch.Tensor | None = None
        self._greedy_partial_indices: torch.Tensor | None = None

        if not self._enable_dspark_fused_greedy:
            return

        hf_config = self.draft_model_config.hf_config
        if getattr(hf_config, "model_type", None) != "deepseek_v4":
            raise ValueError(
                "enable_dspark_fused_greedy currently supports DeepSeek-V4 only"
            )

        if (
            self._draft_topk is not None
            or additional_config.get("deepseek_v4_dspark_topk") is not None
        ):
            raise ValueError(
                "Disable DSpark top-k before enabling full-vocabulary "
                "fused greedy reduction"
            )

        if self.draft_logits is not None:
            # Probabilistic drafting keeps the original implementation.
            return

        vocab_size = max(
            hf_config.vocab_size,
            getattr(hf_config, "draft_vocab_size", None) or 0,
        )
        shape = scratch_shape(self.max_num_reqs, vocab_size)
        self._greedy_partial_values = torch.empty(
            shape,
            dtype=torch.float32,
            device=device,
        )
        self._greedy_partial_indices = torch.empty(
            shape,
            dtype=torch.int32,
            device=device,
        )
    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
    ) -> None:
        # Current upstream _sample_logits also updates acceptance estimates.
        # Preserve that path whenever those logits consumers are enabled.
        if (
            not self._enable_dspark_fused_greedy
            or self.draft_logits is not None
            or self._draft_topk is not None
            or self.model.draft_id_to_target_id is not None
            or self.use_acceptance_estimator
            or self.draft_watermarker is not None
        ):
            super()._sample_sequential(num_reqs, head_hidden)
            return

        assert self._greedy_partial_values is not None
        assert self._greedy_partial_indices is not None

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec

        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        confidence_markov_embeds = []

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)

            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)

            # Keep the original complete Markov projection.
            bias = self.model.markov_bias(markov_embed)
            base_i = base_logits[:, i]
            output_i = self.draft_tokens[:num_reqs, i]

            can_fuse = (
                base_i.shape == bias.shape
                and base_i.dtype == bias.dtype
                and base_i.dtype
                in (torch.float16, torch.bfloat16, torch.float32)
                and base_i.stride(1) == 1
                and bias.stride(1) == 1
            )

            if can_fuse:
                sample_greedy_markov(
                    base_i,
                    bias,
                    output_i,
                    self._greedy_partial_values,
                    self._greedy_partial_indices,
                )
                draft_sampled_i = output_i
            else:
                # Retain the original promotion/broadcasting and sampling path.
                logits_i = base_i + bias
                draft_sampled_i = self._sample_logits(
                    logits_i,
                    idx_map[:, i],
                    sample_pos[:, i],
                    i,
                )
                output_i.copy_(draft_sampled_i)

            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs,
                n_spec,
            )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # Upstream replaces quant_config with None for a BF16 draft. Pass only
        # the target QuaRot path so the draft's existing load_weights can fold
        # input inverse rotation into FC (W @ R) and align fallback embedding /
        # lm_head before upstream decides weight sharing. Do not rotate again
        # after loading or replace the draft's own quantization configuration.
        draft_hf_config = self.draft_model_config.hf_config
        rotation_path = get_rotation_path(self.vllm_config)
        draft_hf_config._ascend_target_rotation_path = str(rotation_path) if rotation_path is not None else None
        model = super().load_draft_model(target_model, target_attn_layer_names)
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
        # vLLM #53694 replaced num_tokens_across_dp with the DP sync state.
        dp_sync: Any = None,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        sync_state = num_tokens_across_dp if vllm_version_is("0.28.0") else dp_sync
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
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
