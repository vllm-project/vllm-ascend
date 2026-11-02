# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from vllm.forward_context import BatchDescriptor, create_forward_context, override_forward_context
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import (
    LiLiCorrSpeculator,
)

from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
)


class AscendLiLiCorrSpeculator(LiLiCorrSpeculator, AscendDFlash2Speculator):
    """Keep upstream LiLiCorr scoring and reuse Ascend DFlash2 runtime guards."""

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # LiLiCorr overrides DFlash2 scoring, so its compiled head needs its own
        # request-shaped context after the backbone's forward context exits.
        forward_context = create_forward_context(
            None,
            self.vllm_config,
            batch_descriptor=BatchDescriptor(num_tokens=candidate_ids.shape[0]),
        )
        with override_forward_context(forward_context):
            return super()._score_candidates(candidate_ids, unary_logits, hidden_states)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # The upstream candidate path assumes standard TP: get_top_k_tokens
        # gathers vocabulary shards only over TP, and target_embeddings receives
        # [requests, steps, topk] IDs rather than the flat token rows required by
        # Ascend's embedding exchange buffers. Reject special layouts before
        # profiling can produce partial-vocabulary candidates or fail a copy.
        # TODO: Remove this guard once LiLiCorr gathers candidates across the
        # fine-grained/PCP vocabulary shards and exchanges candidate IDs in the
        # layout expected by Ascend embeddings, with coverage for both modes.
        unsupported_modes = (VocabParallelMode.FINE_GRAINED, VocabParallelMode.PCP_X_TP)
        for name, layer in (("target embedding", self.target_embeddings), ("candidate LM head", model.lm_head)):
            if getattr(layer, "parallel_mode", None) in unsupported_modes:
                raise NotImplementedError(
                    f"LiLiCorr does not support {name} with {layer.parallel_mode.name} vocabulary sharding "
                    "on Ascend; disable fine-grained embedding/LM head TP and PCP vocabulary weight sharding. "
                    "Standard tensor parallelism is supported."
                )
        return model
