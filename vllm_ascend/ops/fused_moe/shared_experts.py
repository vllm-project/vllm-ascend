#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0


class AscendSharedExperts:
    """Ascend-owned shared expert executor.

    Keep the original shared expert module registered on ``AscendMoERunner``
    for checkpoint compatibility while moving split/overlap execution details
    out of the runner.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config,
        quant_type: QuantType,
        quant_method: FusedMoEMethodBase,
    ):
        self.layer = layer
        self.hidden_size = moe_config.hidden_dim
        self.in_dtype = moe_config.in_dtype
        self.quant_type = quant_type
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self.validate_consistency()
                return result

            quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def validate_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        split_out = self.part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.hidden_size,
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self.layer.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self.layer.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self.layer, "expert_gate") and self.layer.expert_gate is not None:
            gate_out, _ = self.layer.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def forward(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap):
            # Only used for int quantization
            has_quantized_shared = hasattr(self.layer.gate_up_proj, "weight_scale") and hasattr(
                self.layer.down_proj, "weight_scale"
            )
            if has_quantized_shared and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.gate_up_proj.weight,
                    self.layer.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
                # Execute activation concurrently with gmm2.

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=None,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=1,
                    clamp_limit=fused_moe_evts.swiglu_limit,
                    glu_alpha=fused_moe_evts.swiglu_alpha,
                    glu_bias=fused_moe_evts.swiglu_beta,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.down_proj.weight,
                    self.layer.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation concurrently with gmm2.
                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                    hidden_states,
                    topk_weight=None,
                    group_index=None,
                    dst_type=torch.float8_e4m3fn,
                    quant_mode=2,
                    clamp_value=fused_moe_evts.swiglu_limit,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self.layer.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                part1_out = self.part1(hidden_states)
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self.part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        # NOTE: This is exactly the opposite of
        # `maybe_all_reduce_tensor_model_parallel`
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out
