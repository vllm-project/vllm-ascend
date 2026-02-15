#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_ref_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    PyTorch reference implementation of causal_conv1d.
    
    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        initial_states: (batch, dim, width - 1)
        final_states_out: (batch, dim, width - 1)
        return_final_states: bool
        activation: str
    
    Returns:
        out: (batch, dim, seqlen)
        final_states_out: (batch, dim, width - 1) if return_final_states
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata=None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    PyTorch implementation of causal_conv1d_fn for 310P.
    
    Args:
        x: (batch, dim, seqlen) or (dim, cu_seq_len) for varlen
        weight: (dim, width)
        bias: (dim,)
        query_start_loc: (batch + 1) int32
        cache_indices: (batch) int32
        has_initial_state: (batch) bool
        conv_states: (..., dim, width - 1)
        activation: str
        pad_slot_id: int
        metadata: attention metadata
    
    Returns:
        out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b_b.append(
            causal_conv1d_ref_pytorch(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)] if has_initial_state[i] else None,
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


def causal_conv1d_update_pytorch(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    PyTorch implementation of causal_conv1d_update for 310P.
    
    Args:
        x: Input tensor
        conv_state: (..., dim, state_len)
        weight: (dim, width)
        bias: (dim,)
        activation: str
        conv_state_indices: (batch,) int32
        num_accepted_tokens: (batch,) int32
        query_start_loc: (batch + 1,) int32
        max_query_len: int
        pad_slot_id: int
        block_idx_last_scheduled_token: (batch,) int32
        initial_state_idx: (batch,) int32
        validate_data: bool
    
    Returns:
        out: same shape as x
    """
    weight = weight.transpose(0, 1).contiguous()
    conv_state = conv_state.transpose(1, 2).contiguous()
    
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)

    if query_start_loc is None:
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width, _ = weight.shape
    num_cache_lines, state_len_total, _ = conv_state.size()

    out = x

    if query_start_loc is None:
        for i in range(batch):
            x_i = x[i]
            conv_state_i = conv_state[i]
            
            if conv_state_indices is not None:
                idx = conv_state_indices[i].item()
                if idx == pad_slot_id:
                    continue
                conv_state_i = conv_state[idx]
            else:
                conv_state_i = conv_state[i]
            
            if initial_state_idx is not None:
                init_idx = initial_state_idx[i].item()
                if init_idx >= 0:
                    conv_state_i = conv_state[init_idx]
            
            if num_accepted_tokens is not None:
                accepted = num_accepted_tokens[i].item()
                if accepted > 0:
                    x_i = x_i[:accepted]
            
            out_i = causal_conv1d_ref_pytorch(
                x_i,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_state_i,
                initial_states=conv_state_i,
            )[0]
            
            out[i] = out_i
    else:
        for i in range(batch):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            seq_len = end - start
            
            if seq_len == 0:
                continue
            
            x_i = x[start:end]
            
            if conv_state_indices is not None:
                idx = conv_state_indices[i].item()
                if idx == pad_slot_id:
                    continue
                conv_state_i = conv_state[idx]
            else:
                conv_state_i = conv_state[i]
            
            if initial_state_idx is not None:
                init_idx = initial_state_idx[i].item()
                if init_idx >= 0:
                    conv_state_i = conv_state[init_idx]
            
            if num_accepted_tokens is not None:
                accepted = num_accepted_tokens[i].item()
                if accepted > 0:
                    x_i = x_i[:accepted]
            
            out_i = causal_conv1d_ref_pytorch(
                x_i,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_state_i,
                initial_states=conv_state_i,
            )[0]
            
            out[start:end] = out_i

    if unsqueeze:
        out = out.squeeze(1)

    out = out.to(original_x_dtype)
    return out
