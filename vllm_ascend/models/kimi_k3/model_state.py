# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

"""Kimi hybrid model state for the Ascend V2 runner."""

from typing import TYPE_CHECKING

from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class AscendKimiModelState(AscendMambaHybridModelState):
    _skip_spec_cpu_length_sync = False

    def initialize_spec_cpu_length_sync(self, runner: "NPUModelRunner") -> None:
        """Check length consumers after draft attention and KV-cache groups have been bound."""
        self._skip_spec_cpu_length_sync = False
        spec = runner.speculative_config
        if (
            spec is None
            or runner.model_config.architecture != "KimiK31ForConditionalGeneration"
            or runner.parallel_config.pipeline_parallel_size != 1
            or runner.parallel_config.prefill_context_parallel_size != 1
            or runner.ascend_config.sparse_kv_offload_config.enabled
            or spec.enable_adaptive_verification
        ):
            return

        if spec.method == "dflash":
            self._skip_spec_cpu_length_sync = self._dflash2_uses_device_lengths(runner)
        elif spec.method == "dspark" and self._a3_dspark_uses_device_lengths(runner):
            # Drain a previous profiling copy before the runner stops waiting on its event.
            runner.num_computed_tokens_stream.synchronize()
            self._skip_spec_cpu_length_sync = True

    @staticmethod
    def _dflash2_uses_device_lengths(runner: "NPUModelRunner") -> bool:
        if (
            runner.parallel_config.decode_context_parallel_size != 1
            or runner.vllm_config.kv_transfer_config is not None
        ):
            return False

        from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
        from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
        from vllm_ascend.attention.mla_v1 import AscendMLABackend
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend
        from vllm_ascend.ops.kimi_kda import KimiKDAHostMetadataBackend
        from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

        drafter = runner.speculator
        if not isinstance(drafter, AscendDFlash2Speculator):
            return False
        draft_layers = getattr(drafter, "attn_backends", {})
        # MLA prepares exact draft lengths itself; CANN FlashAttn consumes device lengths.
        if set(draft_layers.values()) not in ({AscendMLABackend}, {AscendFlashAttentionBackend}):
            return False
        # The runner's attention groups also include the draft layers.
        target_backends = {
            group.backend
            for groups in runner.attn_groups
            for group in groups
            if any(name not in draft_layers for name in group.layer_names)
        }
        return bool(target_backends) and target_backends <= {
            KimiKDAHostMetadataBackend,
            AscendSFABackend,
            AscendSFAIndexerBackend,
        }

    def _a3_dspark_uses_device_lengths(self, runner: "NPUModelRunner") -> bool:
        draft = runner.speculator
        draft_config = runner.speculative_config.draft_model_config
        if (
            not getattr(draft, "use_a3_mla_sink", False)
            or draft.attn_architecture != "MLA"
            or not draft.draft_attn_layer_names
            or draft_config is None
            or "K3DSparkModel" not in (draft_config.architectures or ())
        ):
            return False

        from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

        from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
        from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
        from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder
        from vllm_ascend.ops.kimi_kda import KimiKDAHostMetadataBuilder

        if self.recoverssm is not None and type(self.recoverssm) is not RecoverSSMState:
            return False
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
        return bool(target_builders) and target_builders <= supported

    def can_skip_spec_cpu_length_sync(self) -> bool:
        return self._skip_spec_cpu_length_sync
