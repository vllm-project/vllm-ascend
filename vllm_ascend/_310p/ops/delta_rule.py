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


def chunk_gated_delta_rule_pytorch(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    head_first=False,
    use_qk_l2norm_in_kernel=False,
):
    """
    PyTorch implementation of chunk_gated_delta_rule.
    This is a fallback implementation for 310P without Triton support.
    
    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        g: gating tensor
        beta: beta tensor
        initial_state: initial state
        output_final_state: whether to output final state
        cu_seqlens: cumulative sequence lengths
        head_first: whether head dimension is first
        use_qk_l2norm_in_kernel: whether to use L2 normalization in kernel
    
    Returns:
        o: output tensor
        final_state: final state (if output_final_state=True)
    """
    if cu_seqlens is None:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HV = v.shape[2]
        N = B
    else:
        N = len(cu_seqlens) - 1
        B, T, H, K = k.shape
        V = v.shape[-1]
        HV = v.shape[2]
    
    scale = K ** -0.5
    
    o = torch.zeros_like(v)
    
    if initial_state is not None:
        h = initial_state.clone()
    else:
        h = torch.zeros(N, HV, K, V, dtype=q.dtype, device=q.device)
    
    if cu_seqlens is not None:
        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            
            if seq_len == 0:
                continue
            
            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]
            g_seq = g[start:end] if g is not None else None
            beta_seq = beta[start:end] if beta is not None else None
            
            for t in range(seq_len):
                q_t = q_seq[t]
                k_t = k_seq[t]
                v_t = v_seq[t]
                
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t)) + 1e-6)
                    k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t)) + 1e-6)
                
                q_t = q_t * scale
                
                if g_seq is not None:
                    g_t = g_seq[t]
                    h[i] = h[i] * torch.exp(g_t)
                
                v_t = v_t - torch.sum(h[i] * k_t, dim=0)
                
                if beta_seq is not None:
                    beta_t = beta_seq[t]
                    v_t = v_t * beta_t
                
                h[i] = h[i] + k_t.unsqueeze(1) * v_t.unsqueeze(0)
                
                o[start + t] = torch.sum(h[i] * q_t, dim=0)
    else:
        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            
            if use_qk_l2norm_in_kernel:
                q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True)) + 1e-6)
                k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True)) + 1e-6)
            
            q_t = q_t * scale
            
            if g is not None:
                g_t = g[:, t]
                h = h * torch.exp(g_t.unsqueeze(-1).unsqueeze(-1))
            
            v_t = v_t - torch.sum(h * k_t.unsqueeze(-1), dim=1)
            
            if beta is not None:
                beta_t = beta[:, t]
                v_t = v_t * beta_t.unsqueeze(-1)
            
            h = h + k_t.unsqueeze(1) * v_t.unsqueeze.unsqueeze(0)
            
            o[:, t] = torch.sum(h * q_t.unsqueeze(-1), dim=1)
    
    if output_final_state:
        return o, h
    else:
        return o, None


def fused_recurrent_gated_delta_rule_pytorch(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    inplace_final_state=False,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """
    PyTorch implementation of fused_recurrent_gated_delta_rule.
    This is a fallback implementation for 310P without Triton support.
    """
    if cu_seqlens is None:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HV = v.shape[2]
        N = B
    else:
        N = len(cu_seqlens) - 1
        B, T, H, K = k.shape
        V = v.shape[-1]
        HV = v.shape[2]
    
    scale = K ** -0.5
    
    o = torch.zeros_like(v)
    
    if initial_state is not None:
        if inplace_final_state:
            h = initial_state
        else:
            h = initial_state.clone()
    else:
        h = torch.zeros(N, HV, K, V, dtype=q.dtype, device=q.device)
    
    if cu_seqlens is not None:
        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            
            if seq_len == 0:
                continue
            
            if ssm_state_indices is not None:
                state_idx = ssm_state_indices[i].item()
                if num_accepted_tokens is not None:
                    accepted = num_accepted_tokens[i].item()
                    seq_len = min(seq_len, accepted)
            
            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]
            g_seq = g[start:end] if g is not None else None
            beta_seq = beta[start:end] if beta is not None else None
            
            for t in range(seq_len):
                q_t = q_seq[t]
                k_t = k_seq[t]
                v_t = v_seq[t]
                
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t)) + 1e-6)
                    k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t)) + 1e-6)
                
                q_t = q_t * scale
                
                if g_seq is not None:
                    g_t = g_seq[t]
                    h[i] = h[i] * torch.exp(g_t)
                
                v_t = v_t - torch.sum(h[i] * k_t, dim=0)
                
                if beta_seq is not None:
                    beta_t = beta_seq[t]
                    v_t = v_t * beta_t
                
                h[i] = h[i] + k_t.unsqueeze(1) * v_t.unsqueeze(0)
                
                o[start + t] = torch.sum(h[i] * q_t, dim=0)
    else:
        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            
            if use_qk_l2norm_in_kernel:
                q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True)) + 1e-6)
                k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True)) + 1e-6)
            
            q_t = q_t * scale
            
            if g is not None:
                g_t = g[:, t]
                h = h * torch.exp(g_t.unsqueeze(-1).unsqueeze(-1))
            
            v_t = v_t - torch.sum(h * k_t.unsqueeze(-1), dim=1)
            
            if beta is not None:
                beta_t = beta[:, t]
                v_t = v_t * beta_t.unsqueeze(-1)
            
            h = h + k_t.unsqueeze(1) * v_t.unsqueeze(0)
            
            o[:, t] = torch.sum(h * q_t.unsqueeze(-1), dim=1)
    
    if inplace_final_state:
        return o, h
    else:
        return o, h


def fused_sigmoid_gating_delta_rule_update_pytorch(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """
    PyTorch implementation of fused_sigmoid_gating_delta_rule_update.
    This is a fallback implementation for 310P without Triton support.
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    
    if scale is None:
        scale = K ** -0.5
    
    o = torch.zeros_like(v)
    
    if initial_state_source is not None:
        h = initial_state_source.clone()
    else:
        h = torch.zeros(N, HV, K, V, dtype=q.dtype, device=q.device)
    
    if cu_seqlens is not None:
        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            
            if seq_len == 0:
                continue
            
            if initial_state_indices is not None:
                state_idx = initial_state_indices[i].item()
                if state_idx >= 0:
                    h_i = h[state_idx].clone()
                else:
                    h_i = torch.zeros(HV, K, V, dtype=q.dtype, device=q.device)
            else:
                h_i = h[i].clone()
            
            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]
            a_seq = a[start:end]
            b_seq = b[start:end]
            
            for t in range(seq_len):
                q_t = q_seq[t]
                k_t = k_seq[t]
                v_t = v_seq[t]
                a_t = a_seq[t]
                b_t = b_seq[t]
                
                x = a_t + dt_bias
                beta_x = softplus_beta * x
                softplus_x = torch.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
                    x,
                )
                g_t = -torch.exp(A_log) * softplus_x
                
                beta_t = 1.0 / (1.0 + torch.exp(-b_t))
                
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t)) + 1e-6)
                    k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t)) + 1e-6)
                
                q_t = q_t * scale
                
                h_i = h_i * torch.exp(g_t.unsqueeze(-1).unsqueeze(-1))
                
                v_t = v_t - torch.sum(h_i * k_t.unsqueeze(-1), dim=1)
                
                v_t = v_t * beta_t.unsqueeze(-1)
                
                h_i = h_i + k_t.unsqueeze(1) * v_t.unsqueeze(0)
                
                o[start + t] = torch.sum(h_i * q_t.unsqueeze(-1), dim=1)
            
            if initial_state_indices is not None:
                state_idx = initial_state_indices[i].item()
                if state_idx >= 0:
                    h[state_idx] = h_i
    else:
        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            a_t = a[:, t]
            b_t = b[:, t]
            
            x = a_t + dt_bias
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
                x,
            )
            g_t = -torch.exp(A_log) * softplus_x
            
            beta_t = 1.0 / (1.0 + torch.exp(-b_t))
            
            if use_qk_l2norm_in_kernel:
                q_t = q_t / (torch.sqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True)) + 1e-6)
                k_t = k_t / (torch.sqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True)) + 1e-6)
            
            q_t = q_t * scale
            
            h = h * torch.exp(g_t.unsqueeze(-1).unsqueeze(-1))
            
            v_t = v_t - torch.sum(h * k_t.unsqueeze(-1), dim=1)
            
            v_t = v_t * beta_t.unsqueeze(-1)
            
            h = h + k_t.unsqueeze(1) * v_t.unsqueeze(0)
            
            o[:, t] = torch.sum(h * q_t.unsqueeze(-1), dim=1)
    
    return o
