# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""AscendC TextGatedResidual integration for Qwen4Exp HyperConnections.

Requires the matching CANN TextGatedResidual package and Torch extension. Keep
checkpoint names/layouts inherited from upstream; only replace HC execution.
Injection tensors in this class are final weights, unlike upstream logits.
"""

import torch
from vllm.logger import logger
from vllm.models.qwen4_exp.amd.hyperconnection import GatedResidual

from vllm_ascend.ops.triton.qwen4_exp.hc import combine_final


class AscendTextGatedResidual(GatedResidual):
    """Preserve upstream weight loading, execute read/mix/inject via ACLNN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._textgr_enabled = bool(getattr(self.config, "hc_per_branch_norm", True))
        self.register_buffer("_textgr_gamma", None, persistent=False)
        self.register_buffer("_textgr_down", None, persistent=False)
        self.register_buffer("_textgr_up", None, persistent=False)
        self.register_buffer("_textgr_gate", None, persistent=False)

    def prepare_textgr_weights(self) -> None:
        """Prepare immutable inference views after normal checkpoint loading."""
        if not getattr(self, "_textgr_enabled", True):
            return

        # Keep the upstream implementation available when the optional CANN
        # extension is absent.  The operator is registered by the import below.
        try:
            from cann_ops_transformer.ops.mhc.text_gated_residual.text_gated_residual import (
                text_gated_residual_op_builder,  # noqa: F401
            )

            registered_op = torch.ops.cann_ops_transformer.text_gated_residual
            del registered_op
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            self._textgr_enabled = False
            logger.warning_once(
                "CANN TextGatedResidual is unavailable; falling back to the upstream HC implementation: %s",
                exc,
            )
            return

        down = self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down
        # The vendor ABI consumes ND weights, not linear-layer NZ storage.
        down.skip_weight_nz_conversion = True
        self.input_mix_weight_up.skip_weight_nz_conversion = True
        gamma = self.hc_norm.weight.detach().to(dtype=torch.float32)
        self._textgr_gamma = gamma.contiguous()
        if self.use_combine:
            merged = self.input_mix_weight_down_block_inject.weight.detach()
            self._textgr_down = merged[: self.config.hc_lowrank].contiguous()
            self._textgr_gate = merged[self.config.hc_lowrank : self.config.hc_lowrank + self.hc_count].contiguous()
        else:
            self._textgr_down = self.input_mix_weight_down.weight.detach().contiguous()
            self._textgr_gate = None
        self._textgr_up = self.input_mix_weight_up.weight.detach().contiguous()

    def mix(self, hidden_states: torch.Tensor):
        if not getattr(self, "_textgr_enabled", True):
            return super().mix(hidden_states)

        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hyper_hidden_size:
            raise ValueError("TextGatedResidual model input must be [tokens, C*H]")
        if self._textgr_gamma is None:
            raise RuntimeError("Prepare HC weights after checkpoint loading, before requests")
        outputs = torch.ops.cann_ops_transformer.text_gated_residual(
            hidden_states.unsqueeze(0),
            self._textgr_down,
            self._textgr_up,
            w_gate=self._textgr_gate,
            channels=self.hc_count,
            gamma=self._textgr_gamma,
            eps=self.config.rms_norm_eps,
            need_backward=False,
        )
        mixed = outputs[0]
        injection = outputs[1].squeeze(0) if self.use_combine else None
        return hidden_states, mixed.squeeze(0), injection

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        if not getattr(self, "_textgr_enabled", True):
            return super().combine_and_mix(hidden_states, prev_block_output, prev_injection)
        hidden_states = self.combine(hidden_states, prev_block_output, prev_injection)
        return self.mix(hidden_states)

    def combine(self, hidden_states, block_output, injection):
        if not getattr(self, "_textgr_enabled", True):
            return super().combine(hidden_states, block_output, injection)
        return combine_final(hidden_states, block_output, injection, self.hc_count)


def prepare_model_textgr_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, AscendTextGatedResidual):
            module.prepare_textgr_weights()
