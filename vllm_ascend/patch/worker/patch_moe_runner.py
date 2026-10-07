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
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _moe_runner_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invoke the fused moe layer.

    Input:
    - hidden_states
    - router_logits

    Output:
    - The new hidden_states.

    Calling sequence
    - forward
      - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
        - _forward_impl

    Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
    to the following reason:
    1. pytorch cannot handle union types in custom op signatures so
       _moe_forward and _moe_forward_shared must be split.
    """

    # Apply transform for routed experts (e.g., latent projection
    # for latent MoE)
    hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)

    # Record before `_maybe_pad_hidden_states` pads activations to match
    # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
    # so routed output can be trimmed before
    # shared+routed add / latent up proj if needed.

    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
        shared_experts_input,
        hidden_states,
    )

    result = self._forward_entry(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        self._encode_layer_name(),
        self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0,
    )

    #
    # Note: there are two all-reduce points below. They are mutually
    # exclusive, controlled by _fused_output_is_reduced
    #  - When True: the combine kernel already reduced fused_output,
    #    so we reduce shared_output here to match, then skip the
    #    all-reduce in _maybe_reduce_final_output.
    #  - When False: neither output is reduced yet, so we combine
    #    them first and all-reduce the sum in _maybe_reduce_final_output.

    # Extract outputs from result
    if isinstance(result, tuple):
        shared_output, fused_output = result
    else:
        shared_output, fused_output = None, result

    if og_hidden_dim_pre_xform is not None:
        fused_output = fused_output[..., :og_hidden_dim_pre_xform]

    # If combine kernel already reduced fused, reduce shared to match.
    # See note above re: the two all-reduce points.
    shared_output = self._maybe_reduce_shared_expert_output(shared_output)

    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)

    # Apply output transform (e.g. latent -> full dim)
    fused_output = self.apply_routed_output_transform(fused_output)

    if shared_output is not None:
        # NPU foreach path: a single fused multi-tensor add kernel,
        # independent of the tensor-list length, instead of one
        # elementwise add per combine.
        result = torch._foreach_add([shared_output], [fused_output])[0]
    else:
        result = fused_output

    result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform)

    return self._maybe_add_zero_expert_output(result)


# AscendMoERunner does not define forward, so every Ascend MoE layer picks up
# the patched method through inheritance.
MoERunner.forward = _moe_runner_forward
