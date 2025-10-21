# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
import torch.nn.functional as F
import triton
from vllm.model_executor.layers.fla.ops.layernorm_guard import \
    layer_norm_fwd_kernel
from math import log

def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        return y.reshape(x_shape_og)


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=128,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_qk_heads, sequence_length, k_head_dim = key.shape
    num_v_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
    key = F.pad(key, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    sequence_length_padded = sequence_length + pad_size
    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=torch.bool,
                                 device=query.device),
                      diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) -
                   g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -(
        (k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    lg = int(log(chunk_size, 2))

    block_size = 1
    attn_inv = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device).repeat((tuple(attn.shape)[:-2] + (1, 1)))
    attn = attn_inv - attn
    for i in range(lg):
        block_num = chunk_size // block_size
        prod = attn @ attn_inv
        attn_inv_block = attn_inv.view(tuple(attn.shape)[:-2] + (block_num, block_size, block_num, block_size)).transpose(-2, -3)
        prod_block = prod.view(tuple(attn.shape)[:-2] + (block_num, block_size, block_num, block_size)).transpose(-2, -3)
        r0 = torch.arange(block_num // 2, device=attn.device) * 2
        r1 = r0 + 1
        attn_inv_block[:, :, :, r1, r0, :, :] = -attn_inv_block[..., r1, r1, :, :] @ prod_block[..., r1, r0, :, :]
        attn_inv = attn_inv_block.transpose(-2, -3).view(tuple(attn_inv_block.shape)[:-4] + (chunk_size, chunk_size))
        block_size *= 2
    attn = attn_inv

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (torch.zeros(batch_size, num_v_heads,
                                        k_head_dim, v_head_dim).to(value) if
                            initial_state is None else initial_state.to(value))

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size,
                                 chunk_size,
                                 dtype=torch.bool,
                                 device=query.device),
                      diagonal=1)
    query_view = query.reshape(query.shape[0], query.shape[1], -1, chunk_size, query.shape[-1])
    key_trans = key.reshape(key.shape[0], key.shape[1], -1, chunk_size, key.shape[-1]).transpose(-1, -2)
    qk = query_view @ key_trans
    attn_score = qk * decay_mask.masked_fill_(mask, 0)

    gexp = g[:, :, :, :, None].exp()
    qgexp = query * gexp

    kgexp = (g[:, :, :, -1, None] - g[:, :, :]).exp()[..., None]
    kgexp = key * kgexp
    
    k_cumdecay_qgexp = torch.cat([k_cumdecay, qgexp], dim=3)
    v_new_out = torch.zeros_like(value)
    attn_inter_out = torch.zeros_like(value)

    # for each chunk
    for i in range(0, sequence_length_padded // chunk_size):
        v_i = value[:, :, i]
        attn = attn_score[:, :, i]
        v_prime_attn_inter = (k_cumdecay_qgexp[:, :, i]) @ last_recurrent_state
        v_prime = v_prime_attn_inter[:, :, :chunk_size]
        attn_inter = v_prime_attn_inter[:, :, chunk_size:]
        v_new = v_i - v_prime
        v_new_out[:, :, i] = v_new
        attn_inter_out[:, :, i] = attn_inter
        last_recurrent_state *= gexp[:, :, i, -1, :, None]
        last_recurrent_state += (kgexp[:, :, i]).transpose(-1, -2) @ v_new
    core_attn_out = attn_inter_out + attn_score @ v_new_out

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                          core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1,
                                            2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
