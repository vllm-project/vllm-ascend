# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind Kimi's request-indexed RecoverSSM records to Ascend commit kernels."""

from vllm_ascend.attention.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.kda import configure_kda_checkpoint_builder, prepare_kda_checkpoint_prefill
from vllm_ascend.ops.recurrent_pcp import PCPRecurrentMetadataBuilderMixin
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext


class AscendKimiKDARecoverSSMMetadataBuilder(PCPRecurrentMetadataBuilderMixin, AscendKDARecoverSSMMetadataBuilder):
    recoverssm_context: KDARecoverSSMCommitContext | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configure_kda_checkpoint_builder(self)

    def _build_prefill_chunk_metadata(self, m, metadata, draft_counts):
        if not self.kv_cache_spec.num_prefill_checkpoint_blocks:
            return super()._build_prefill_chunk_metadata(m, metadata, draft_counts)
        starts = prepare_kda_checkpoint_prefill(self, m, metadata, draft_counts)
        return _build_non_spec_chunked_prefill_metadata(
            self, starts, m.query_start_loc.device, build_device_metadata=False
        )

    def _get_recoverssm_context(self) -> KDARecoverSSMCommitContext:
        forward_context = self.vllm_config.compilation_config.static_forward_context
        layers = [forward_context[name] for name in self.layer_names]
        context = self.recoverssm_context
        if context is None or not context.is_bound_to(layers):
            context = KDARecoverSSMCommitContext.create(
                layers,
                spec_query_len=1 + self.vllm_config.num_speculative_tokens,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
            self.recoverssm_context = context
        return context
