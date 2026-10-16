# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
import torch.nn.functional as F
from vllm.model_executor.models.lilicorr import LiLiCorrLatticeAttention
from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv


def lattice_attention_forward(
    self: LiLiCorrLatticeAttention,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the learned per-head additive bias using NPU tensor ops.

    The lattice has no causal mask or KV cache. Compute attention scores and
    softmax in fp32 so arbitrary learned biases do not depend on the NPU
    fused SDPA backend's mask or head-dimension restrictions.
    """
    batch_size, seq_len, _ = hidden_states.shape
    qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
    shape = (batch_size, seq_len, self.num_heads, self.head_dim)
    q, k, v = (tensor.reshape(shape).transpose(1, 2) for tensor in qkv.chunk(3, dim=-1))
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (self.head_dim**-0.5)
    probabilities = torch.softmax(scores + attention_bias.float(), dim=-1).to(v.dtype)
    output = torch.matmul(probabilities, v)
    return self.out_proj(output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size))


LiLiCorrLatticeAttention.forward = lattice_attention_forward


def grouped_conv_convolve(
    self: DFlashGroupedConv,
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    side: int,
) -> torch.Tensor:
    """Keep convolution inside each request's query block on NPU.

    Accumulate in fp32 as in the upstream CUDA kernel. A short draft may
    have fewer total rows than kernel taps, so omit shifts with no input.
    Keep the upstream module and quantized projection/weight loader intact.
    """
    num_rows = hidden_states.shape[0]
    if num_rows == 0:
        return hidden_states
    blocks = hidden_states.float().unflatten(-1, (self.num_groups, self.group_size))
    base = self.base_kernel[side].float().reshape(self.taps, self.num_groups, self.group_size)
    delta = delta.float().unsqueeze(-1)
    output = (base[0] + delta[:, 0]) * blocks
    positions = torch.arange(num_rows, device=hidden_states.device) % self.block_size
    for tap in range(1, min(self.taps, num_rows)):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + (base[tap] + delta[:, tap]) * shifted * (positions >= tap).reshape(-1, 1, 1)
    return output.flatten(-2).to(hidden_states.dtype)


DFlashGroupedConv._convolve = grouped_conv_convolve
