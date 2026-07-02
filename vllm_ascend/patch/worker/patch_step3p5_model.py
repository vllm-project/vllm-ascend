# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# monkey-patch: FC1 + VL layer0 residual fix for Step3p5Model.
#
# WHY: vLLM engine pre-computes inputs_embeds for multimodal models
# (gpu_model_runner.py:3336), producing full-replicated [N,H].
# FC1's SequenceRowParallelOp uses reduce_scatter, producing scatter
# [N//tp,H] attention output.  The residual ``attn_out += inputs_embeds``
# would fail because [N//tp,H] += [N,H] is a shape mismatch.
#
# This patch replaces Step3p5DecoderLayer.forward for layer 0 only:
# the replicated residual is chunked to [N//tp,H] before the addition,
# so the layer stays on the standard FC1 scatter convention.
#
# Unlike the previous "scatter inputs_embeds before model forward" approach,
# this fix does NOT require any extra condition in linear_op.py.
# The QKV column-parallel op processes replicated input natively:
#   need_all_gather = not(layer0 and VL and attn) = False
#   -> skip all-gather -> [N,H] x [H,H/8] = [N,H/8]   (correct)
#
# Deploy: cp to vllm-ascend/vllm_ascend/patch/worker/ and add
#         import line to patch/worker/__init__.py

import torch
from vllm.distributed import get_tp_group
from vllm.logger import logger

try:
    from vllm.model_executor.models.step3p5 import Step3p5DecoderLayer
except ImportError:
    logger.debug("Step3p5DecoderLayer not available, skipping FC1 layer0 patch")
    Step3p5DecoderLayer = None


if Step3p5DecoderLayer is not None:
    _original_layer_forward = Step3p5DecoderLayer.forward

    def _patched_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        # Only intervene for layer 0 of VL models running FC1 on the main model.
        # MTP draft layers (is_draft_model=True, flash_comm_v1=False) and
        # non-FC1 runs are unaffected.
        if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled and not _EXTRA_CTX.is_draft_model:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            # FC1: self_attn o_proj reduce_scatter -> scatter [(N+pad)//tp, H].
            # Text model: input already SP-sharded [N/tp,H], same as FC1 output
            #   -> no chunking needed.
            # VL model: input replicated [N,H] (engine pre-computed inputs_embeds)
            #   -> chunk residual to match FC1 scatter output.
            tp = get_tp_group().world_size
            rank = get_tp_group().rank_in_group
            n_local = hidden_states.shape[0]
            if residual.shape[0] != n_local:
                # Replicated input: pad + slice to match FC1 scatter
                pad_needed = n_local * tp - residual.shape[0]
                if pad_needed > 0:
                    residual = torch.nn.functional.pad(residual, (0, 0, 0, pad_needed))
                residual = residual[rank * n_local : (rank + 1) * n_local].contiguous()
            hidden_states = hidden_states + residual

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            if self.use_moe:
                ffn_output = self.moe(hidden_states)
            else:
                ffn_output = self.mlp(hidden_states)
            hidden_states = ffn_output + residual
            return hidden_states

        return _original_layer_forward(self, positions, hidden_states)

    Step3p5DecoderLayer.forward = _patched_forward
    logger.debug("Applied Step3p5DecoderLayer FC1 layer0 residual patch")
