# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch.nn.functional as F
import torch_npu

# Smaller groups can produce invalid scales on A3, notably T=1 TP16 shards.
_MIN_GROUP_TOKENS = 8


def shared_dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    clamp_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run clipped shared SwiGLU using the official A3 one-group kernel.

    The shared projection stores [gate | up], whereas official mode 1 expects
    interleaved gate/up channels. A non-null group index selects the path that
    honors clamp_limit. This changes only the operator input, not MoE routing.
    """
    num_tokens, width = x.shape
    if num_tokens == 0:
        return x.new_empty((0, width // 2), dtype=torch.int8), x.new_empty((0,), dtype=torch.float32)

    gate, up = x.chunk(2, dim=-1)
    x = torch.stack((gate, up), dim=-1).reshape(num_tokens, width)
    gate_scale, up_scale = weight_scale.reshape(-1).chunk(2)
    weight_scale = torch.stack((gate_scale, up_scale), dim=-1).reshape(1, width)
    activation_scale = activation_scale.reshape(num_tokens)

    padded_tokens = max(num_tokens, _MIN_GROUP_TOKENS)
    if padded_tokens != num_tokens:
        x = F.pad(x, (0, 0, 0, padded_tokens - num_tokens))
        activation_scale = F.pad(activation_scale, (0, padded_tokens - num_tokens), value=1.0)
    # Device-side construction also works when the entire adapter is captured.
    group_index = torch.full((1,), padded_tokens, dtype=torch.int64, device=x.device)
    output, scale = torch_npu.npu_dequant_swiglu_quant(
        x=x,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        group_index=group_index,
        activate_left=True,
        quant_mode=1,
        swiglu_mode=1,
        clamp_limit=clamp_limit,
        glu_alpha=1.0,
        glu_bias=0.0,
    )
    return output[:num_tokens], scale[:num_tokens]
