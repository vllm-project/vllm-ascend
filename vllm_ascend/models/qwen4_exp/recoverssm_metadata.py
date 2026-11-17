# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN verify metadata and post-sampling accepted-prefix commit."""

import logging
from dataclasses import fields

import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata

from vllm_ascend.attention.recoverssm_gdn import GDNRecoverSSMCommitContext
from vllm_ascend.models.qwen4_exp.recoverssm_config import validate_qwen_gdn_recoverssm
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder


class QwenGDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    def commit_recoverssm_state(self, num_accepted_tokens):
        if self.num_spec_decodes:
            context = self.gdn_recoverssm_context
            context.commit(
                num_accepted_tokens,
                self.gdn_recoverssm_state_indices[: self.num_spec_decodes],
                self.spec_query_start_loc[: self.num_spec_decodes + 1],
                request_indices=self.gdn_recoverssm_request_indices,
            )
            if context.commit_steps == 1:
                logging.getLogger(__name__).info(
                    "Qwen GDN RecoverSSM accepted-prefix commit submitted: rows=%d layers=%d Q=%d; "
                    "verify records consumed, rejected suffix excluded by sampler lengths",
                    self.num_spec_decodes,
                    len(context.layers),
                    context.spec_query_len,
                )
        return None  # No align-mode state migration; standard PLE acceptance stays intact.


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNHostMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        validate_qwen_gdn_recoverssm(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._commit_context = None
        self._neutral_accepted = torch.ones(self.spec_sequence_masks.shape[0], dtype=torch.int32, device=device)

    def _finalize_metadata(self, metadata, fused=False):
        metadata = super()._finalize_metadata(metadata, fused=fused)
        if not metadata.num_spec_decodes:
            return metadata
        if self._commit_context is None:
            layers = self.compilation_config.static_forward_context
            self._commit_context = GDNRecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.spec_sequence_masks.shape[0],
            )
        result = QwenGDNRecoverSSMMetadata(
            **{field.name: getattr(metadata, field.name) for field in fields(GDNAttentionMetadata)}
        )
        # Ascend descriptors are attached dynamically by the parent builder.
        result.__dict__.update(metadata.__dict__)
        result.gdn_recoverssm_context = self._commit_context
        result.gdn_recoverssm_state_indices = metadata.spec_state_indices_tensor[:, 0]
        masks_cpu = metadata.spec_sequence_masks_cpu
        if masks_cpu is None:
            masks_cpu = self.spec_sequence_masks_cpu
        # Reconstruct sampler-row mapping on CPU; no device-to-host read in the hot path.
        if masks_cpu is None:
            result.gdn_recoverssm_request_indices = None
        else:
            result.gdn_recoverssm_request_indices = torch.nonzero(masks_cpu, as_tuple=True)[0][
                : metadata.num_spec_decodes
            ].to(metadata.spec_query_start_loc.device, non_blocking=True)
        # Commit already compacted conv history. Preserve the global acceptance
        # for PLE; only this GDN convolution consumes the neutral one-based bias.
        result.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = self._neutral_accepted[
            : metadata.spec_query_start_loc.numel() - 1
        ]
        result.spec_decode_metadata.use_upstream_conv = True
        return result
