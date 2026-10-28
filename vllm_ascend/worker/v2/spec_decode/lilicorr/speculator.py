# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import (
    LiLiCorrSpeculator,
)

from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


class AscendLiLiCorrSpeculator(LiLiCorrSpeculator, AscendDFlashSpeculator):
    """LiLiCorr drafter using the shared Ascend DFlash runtime."""

    # Candidate top-k bypasses sample_draft's group-aligned row padding.
    _lmhead_tp_sample_draft_supported = False

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
        unsupported_modes = (VocabParallelMode.FINE_GRAINED, VocabParallelMode.PCP_X_TP)
        for name, layer in (("target embedding", self.target_embeddings), ("candidate LM head", model.lm_head)):
            if getattr(layer, "parallel_mode", None) in unsupported_modes:
                raise NotImplementedError(
                    f"LiLiCorr does not support {name} with {layer.parallel_mode.name} vocabulary sharding "
                    "on Ascend; disable fine-grained embedding/LM head TP and PCP vocabulary weight sharding. "
                    "Standard tensor parallelism is supported."
                )
        return model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # LiLiCorr shares DFlash2's candidate walk. Until that NPU Triton
        # kernel is ACL-graph capturable, keep the draft eager while allowing
        # the target model to retain its independently configured graph mode.
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            raise NotImplementedError(
                "LiLiCorr does not currently support graph mode on Ascend; please enable enforce_eager."
            )
        super().init_cudagraph_manager(cudagraph_mode)
