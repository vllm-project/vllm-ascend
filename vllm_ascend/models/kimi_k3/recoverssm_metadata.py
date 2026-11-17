# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind Kimi's request-indexed RecoverSSM records to Ascend commit kernels."""

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext


class AscendKimiKDARecoverSSMMetadataBuilder(AscendKDARecoverSSMMetadataBuilder):
    recoverssm_context: KDARecoverSSMCommitContext | None

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
