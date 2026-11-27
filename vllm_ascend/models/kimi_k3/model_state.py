# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi hybrid model state for the Ascend V2 runner."""

from typing import Any

from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


class AscendKimiModelState(AscendMambaHybridModelState):
    _skip_spec_cpu_length_sync = False

    def initialize_spec_cpu_length_sync(self, runner: Any) -> None:
        """Bind the decision after set_attn, including on KV-cache reinitialization."""
        self._skip_spec_cpu_length_sync = False
        draft = runner.speculator
        if (
            not getattr(draft, "use_a3_mla_sink", False)
            or draft.attn_architecture != "MLA"
            or not draft.draft_attn_layer_names
        ):
            return
        draft_config = runner.speculative_config.draft_model_config
        if (
            runner.model_config.architecture != "KimiK31ForConditionalGeneration"
            or draft_config is None
            or "K3DSparkModel" not in (draft_config.architectures or ())
            or runner.parallel_config.pipeline_parallel_size != 1
            or runner.parallel_config.prefill_context_parallel_size != 1
            or runner.ascend_config.sparse_kv_offload_config.enabled
            or runner.speculative_config.enable_adaptive_verification
        ):
            return

        from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

        from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
        from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
        from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder
        from vllm_ascend.ops.kimi_kda import KimiKDAHostMetadataBuilder

        if self.recoverssm is not None and type(self.recoverssm) is not RecoverSSMState:
            return
        # SFA, its indexer and KDA read exact lengths from the device. Native
        # RecoverSSM also commits using device state indices after rejection.
        # Other builders may require the rejection-corrected host snapshot.
        supported = {
            AscendSFAMetadataBuilder,
            AscendSFADCPMetadataBuilder,
            AscendSFAIndexerMetadataBuilder,
            KimiKDAHostMetadataBuilder,
            AscendKimiKDARecoverSSMMetadataBuilder,
        }
        target_builders = {
            type(builder)
            for groups in runner.attn_groups
            for group in groups
            if any(name not in draft.draft_attn_layer_names for name in group.layer_names)
            for builder in group.metadata_builders
        }
        if target_builders and target_builders <= supported:
            # Drain an earlier profiling/initialization copy once before the
            # runner stops producing and waiting on its per-step event.
            runner.num_computed_tokens_stream.synchronize()
            self._skip_spec_cpu_length_sync = True

    def can_skip_spec_cpu_length_sync(self) -> bool:
        return self._skip_spec_cpu_length_sync
