# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN-native RecoverSSM metadata builder.

Extends the standard GDN attention metadata builder directly, bypassing the
KDA inheritance chain. The standard builder already produces all GDN metadata
fields (spec_causal_conv1d, spec_state_indices_tensor, spec_query_start_loc)
and handles graph padding/staging internally, for both MTP and DSpark.
This subclass only adds graph-padding normalization for the input batch and
the RecoverSSM commit reference for the post-sampling lifecycle.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.ops.triton.gdn.recoverssm_gdn import GDNRecoverSSMCommitContext


@dataclass
class GDNRecoverssmCommitMetadata:
    """Per-request commit descriptors consumed by the RecoverSSM lifecycle."""

    state_indices: torch.Tensor
    query_start_loc: torch.Tensor


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """GDN-native RecoverSSM builder.

    The standard GDN builder's build flow (including _pad_spec_decode_metadata
    and _attach_spec_decode_metadata) already produces the metadata that
    gdn_recoverssm_verify consumes. This subclass only:
    1. Normalizes graph padding on the input batch (pre-build step).
    2. Attaches the RecoverSSM commit descriptors (post-build step).
    It never routes through the KDA attention builder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recoverssm_context = None
        self.use_flashinfer_prefill = False

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_actual_reqs=None,
    ):
        # Pre-build: exclude graph-padded request slots from the batch so
        # they do not become sampler rows or commit metadata entries.
        m, num_draft = self._normalize_graph_padding(
            common_attn_metadata, num_actual_reqs, num_decode_draft_tokens_cpu
        )

        # Standard GDN build: internally handles graph padding, spec staging
        # (_pad_spec_decode_metadata) and GDN metadata attachment
        # (_attach_spec_decode_metadata / _attach_non_spec_*).
        metadata = super().build(
            common_prefix_len, m, num_accepted_tokens, num_draft, fast_build
        )
        if metadata is None:
            return metadata

        # Post-build: attach the RecoverSSM commit descriptors from the
        # final (graph-stable) spec metadata produced by the standard build.
        if metadata.num_spec_decodes:
            metadata.recoverssm_commit = GDNRecoverssmCommitMetadata(
                state_indices=metadata.spec_state_indices_tensor[:, 0],
                query_start_loc=metadata.spec_query_start_loc,
            )

        # Materialize the shared commit context once for all layers.
        self._get_recoverssm_context()

        metadata.pcp_context = getattr(common_attn_metadata, "pcp_context", None)
        return metadata

    def _normalize_graph_padding(self, m, num_actual_reqs, num_decode_draft_tokens_cpu):
        """Clamp graph-padded request slots to the actual request count.

        Ported from the KDA builder; model-agnostic. Without this, graph-padded
        FIA slots would appear as sampler rows and pollute the commit metadata.
        """
        if num_actual_reqs is None or num_actual_reqs >= m.num_reqs:
            return m, num_decode_draft_tokens_cpu
        starts_cpu = m.query_start_loc_cpu.clone()
        starts_cpu[num_actual_reqs + 1 :] = starts_cpu[num_actual_reqs]
        starts = torch.minimum(m.query_start_loc, m.query_start_loc[num_actual_reqs])
        m = m.replace(
            query_start_loc_cpu=starts_cpu,
            query_start_loc=starts,
            num_actual_tokens=int(starts_cpu[num_actual_reqs]),
        )
        if num_decode_draft_tokens_cpu is not None:
            num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu.clone()
            num_decode_draft_tokens_cpu[num_actual_reqs:] = -1
        return m, num_decode_draft_tokens_cpu

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
