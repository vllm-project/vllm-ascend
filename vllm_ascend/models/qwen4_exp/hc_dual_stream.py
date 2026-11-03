# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Optional Qwen4Exp HC injection overlap with Attention.

The checkpoint's merged down projection has independent low-rank and injection
rows.  The low-rank rows feed the gate and next block input; injection is first
read after that block finishes.  Split those rows only when the opt-in is set.
The join lives in opaque custom ops so FULL graph capture records the stream
dependency at the actual injection consumer.
"""

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.amd import hyperconnection as upstream_hc
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    _hc_combine,
    _hc_combine_norm,
    grouped_gemma_rmsnorm,
    hc_gate_mix,
    hc_silu,
)
from vllm_ascend.utils import qwen4_exp_hc_injection_stream


def _launch_injection_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    lowrank_size: int,
    hc_count: int,
) -> torch.Tensor:
    """Queue injection after the gate path, alongside Attention work."""
    main_stream = torch.npu.current_stream()
    injection_stream = qwen4_exp_hc_injection_stream()
    gate_ready = main_stream.record_event()
    with torch.npu.stream(injection_stream):
        injection_stream.wait_event(gate_ready)
        injection = F.linear(x, weight[lowrank_size : lowrank_size + hc_count])
    x.record_stream(injection_stream)
    injection.record_stream(main_stream)
    return injection


def _launch_injection_projection_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    lowrank_size: int,
    hc_count: int,
) -> torch.Tensor:
    del weight
    return x.new_empty((x.shape[0], hc_count))


def _combine_after_injection(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    torch.npu.current_stream().wait_stream(qwen4_exp_hc_injection_stream())
    return _hc_combine(residual, block_output, injection, hc_count)


def _combine_after_injection_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection, hc_count
    return residual.new_empty(residual.shape)


def _combine_norm_after_injection(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.npu.current_stream().wait_stream(qwen4_exp_hc_injection_stream())
    return _hc_combine_norm(residual, block_output, injection, norm_weight, eps, hc_count)


def _combine_norm_after_injection_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection, norm_weight, eps, hc_count
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_hc_launch_injection_projection",
    op_func=_launch_injection_projection,
    fake_impl=_launch_injection_projection_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_after_injection",
    op_func=_combine_after_injection,
    fake_impl=_combine_after_injection_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm_after_injection",
    op_func=_combine_norm_after_injection,
    fake_impl=_combine_norm_after_injection_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


class AscendGatedResidual(upstream_hc.GatedResidual):
    """Retain upstream weights and enable the split only for NPU HC mixers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.dual_stream = get_ascend_config().multistream_qwen4_exp_hc
        except RuntimeError:
            # CPU-only model construction can run before the Ascend runtime is
            # initialized.  The opt-in is unavailable in that context.
            self.dual_stream = False

    def _project_from_normalized(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_combine:
            if self.dual_stream and xn.device.type == "npu":
                weight = self.input_mix_weight_down_block_inject.weight
                lowrank = F.linear(xn, weight[: self.hc_lowrank])
            else:
                projected = self.input_mix_weight_down_block_inject(xn)
                lowrank, injection, _ = projected.split([self.hc_lowrank, self.hc_count, self.pad_size], dim=-1)
        else:
            lowrank = self.input_mix_weight_down(xn)
            injection = None
        lowrank = hc_silu(lowrank, self.hc_count)
        gate = self.input_mix_weight_up(lowrank)
        block_input = hc_gate_mix(xn, gate, self.hc_count)
        if self.use_combine and self.dual_stream and xn.device.type == "npu":
            injection = torch.ops.vllm.qwen4_exp_hc_launch_injection_projection(
                xn, weight, self.hc_lowrank, self.hc_count
            )
        return block_input, injection

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.dual_stream:
            return super().mix(hidden_states)
        xn = grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        block_input, injection = self._project_from_normalized(xn)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.dual_stream or hidden_states.device.type != "npu":
            return super().combine_and_mix(hidden_states, prev_block_output, prev_injection)
        hidden_states, xn = torch.ops.vllm.qwen4_exp_hc_combine_norm_after_injection(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )
        block_input, injection = self._project_from_normalized(xn)
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        if not self.dual_stream or hidden_states.device.type != "npu":
            return super().combine(hidden_states, block_output, injection)
        return torch.ops.vllm.qwen4_exp_hc_combine_after_injection(
            hidden_states, block_output, injection, self.hc_count
        )
