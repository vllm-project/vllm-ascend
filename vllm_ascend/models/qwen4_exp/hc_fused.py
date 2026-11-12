# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Controlled TextGatedResidual integration: original input shape, no padding or fallback.

The 2026-11-06 FullB operator package and matching mhc Torch extension are required. Keep
checkpoint names/layouts inherited from upstream; only replace HC execution.
Injection tensors in this class are final weights, unlike upstream logits.
"""

import torch
from vllm.models.qwen4_exp.amd.hyperconnection import GatedResidual
from vllm.triton_utils import tl, triton


@triton.jit
def _combine_final_kernel(
    residual_ptr,
    block_ptr,
    injection_ptr,
    output_ptr,
    residual_stride,
    block_stride,
    injection_stride,
    output_stride,
    HIDDEN: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1)
    inner = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = inner < HIDDEN
    offsets = channel * HIDDEN + inner
    residual = tl.load(residual_ptr + row * residual_stride + offsets, mask, 0).to(tl.float32)
    block = tl.load(block_ptr + row * block_stride + inner, mask, 0).to(tl.float32)
    if HAS_INJECTION:
        injection = tl.load(injection_ptr + row * injection_stride + channel).to(tl.float32)
        block *= injection
    tl.store(output_ptr + row * output_stride + offsets, residual + block, mask)


def combine_final(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor | None,
    channels: int,
) -> torch.Tensor:
    """Write back a pending block, without another sigmoid or channel scaling."""
    rows, dim = residual.shape
    hidden = dim // channels
    if block_output.shape != (rows, hidden):
        raise ValueError("HC block output has an incompatible shape")
    if injection is not None and injection.shape != (rows, channels):
        raise ValueError("HC final injection has an incompatible shape")
    if residual.stride(1) != 1 or block_output.stride(1) != 1:
        raise ValueError("HC combine requires unit inner strides")
    output = torch.empty_like(residual)
    if rows:
        _combine_final_kernel[(rows, channels, triton.cdiv(hidden, 1024))](
            residual,
            block_output,
            residual if injection is None else injection,
            output,
            residual.stride(0),
            block_output.stride(0),
            0 if injection is None else injection.stride(0),
            output.stride(0),
            HIDDEN=hidden,
            HAS_INJECTION=injection is not None,
            BLOCK=1024,
        )
    return output


class AscendTextGatedResidual(GatedResidual):
    """Preserve upstream weight loading, execute read/mix/inject via ACLNN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        down = self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down
        # The vendor ABI consumes ND weights, not linear-layer NZ storage.
        down.skip_weight_nz_conversion = True
        self.input_mix_weight_up.skip_weight_nz_conversion = True
        self.register_buffer("_textgr_gamma", None, persistent=False)
        self.register_buffer("_textgr_down", None, persistent=False)
        self.register_buffer("_textgr_up", None, persistent=False)
        self.register_buffer("_textgr_gate", None, persistent=False)

    def prepare_textgr_weights(self) -> None:
        """Prepare immutable inference views after normal checkpoint loading."""
        gamma = self.hc_norm.weight.detach().to(dtype=torch.float32)
        if gamma.numel() != self.hyper_hidden_size:
            raise ValueError("TextGatedResidual requires one raw norm weight per HC element")
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
        # Import only in the experimental path. The baseline has no wheel dependency.
        from cann_ops_transformer_custom.ops.mhc.text_gated_residual.text_gated_residual import (
            text_gated_residual_op_builder,  # noqa: F401 - registers the Torch operator
        )

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
        hidden_states = self.combine(hidden_states, prev_block_output, prev_injection)
        return self.mix(hidden_states)

    def combine(self, hidden_states, block_output, injection):
        return combine_final(hidden_states, block_output, injection, self.hc_count)


def prepare_model_textgr_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, AscendTextGatedResidual):
            module.prepare_textgr_weights()
