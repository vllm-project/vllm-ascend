# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN records with the shared RecoverSSM request/commit lifecycle."""

from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    _build_actual_seq_lengths,
    _build_non_spec_chunked_prefill_metadata,
)
from vllm_ascend.ops.triton.gdn_recoverssm import GDNRecoverSSMCommitContext


class AscendGDNRecoverSSMMetadataBuilder(AscendKDARecoverSSMMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flashinfer_prefill = False

    def _build_conv_metadata(self, m, metadata, num_decode_draft_tokens_cpu):
        super()._build_conv_metadata(m, metadata, num_decode_draft_tokens_cpu)
        if metadata.num_prefills and metadata.num_decodes:
            starts = metadata.non_spec_query_start_loc[: metadata.num_decodes + 1]
            metadata.non_spec_decode_metadata = GDNDecodeMetadata(
                causal_conv1d=GDNCausalConv1dMetadata(
                    query_start_loc=starts,
                    cache_indices=metadata.non_spec_state_indices_tensor[: metadata.num_decodes],
                    initial_state_mode=None,
                ),
                actual_seq_lengths=_build_actual_seq_lengths(starts, metadata.num_decodes),
            )

    def _build_prefill_chunk_metadata(self, m, metadata, num_decode_draft_tokens_cpu):
        # GDN's Triton prefill fallback also consumes device chunk descriptors.
        cpu_starts = self._non_spec_cpu_starts(m, metadata, num_decode_draft_tokens_cpu)
        metadata.prefill_query_start_loc = (
            metadata.non_spec_query_start_loc[metadata.num_decodes :] - metadata.num_decode_tokens
        )
        metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor[metadata.num_decodes :]
        metadata.prefill_has_initial_state = metadata.has_initial_state[metadata.num_decodes :]
        return _build_non_spec_chunked_prefill_metadata(
            self, cpu_starts[metadata.num_decodes :] - metadata.num_decode_tokens, m.query_start_loc.device
        )

    def _get_recoverssm_context(self):
        if self.recoverssm_context is None:
            layers = self.vllm_config.compilation_config.static_forward_context
            self.recoverssm_context = GDNRecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self.recoverssm_context


class AscendGDNRecoverSSMBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return AscendGDNRecoverSSMMetadataBuilder
