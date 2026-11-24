# SPDX-License-Identifier: Apache-2.0
"""Ascend 950 full BF16 KDA on the ordinary ReplaySSM speculative decode path."""

import torch


def unpack_mx_qkv(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Expand original output-major E4M3FN weights and group-32 E8M0 scales once at load."""
    rows, cols = weight.shape
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8 or scale.shape != (rows, cols // 32):
        raise ValueError("Full KDA requires original MXFP8 QKV weights and group-32 E8M0 scales")
    out = torch.empty((rows, cols), device=weight.device, dtype=torch.bfloat16)
    for i in range(0, rows, 512):
        part = weight[i : i + 512].float().view(-1, cols // 32, 32)
        exp = scale[i : i + 512].float()
        factor = torch.where(exp == 255, float("nan"), torch.exp2(exp - 127))
        out[i : i + 512].copy_((part * factor.unsqueeze(-1)).reshape(-1, cols))
    return out


def full_kda_verify(layer, op, hidden, metadata, conv_weight):
    """Map paged states into request order and publish records expected by the A5 commit."""
    slots = metadata.spec_state_indices_tensor[:, 0]
    starts = metadata.spec_query_start_loc
    batch = slots.shape[0]
    window = layer.num_spec + 1
    steps = torch.arange(window, device=hidden.device, dtype=torch.int32)
    lengths = starts[1:] - starts[:-1]
    valid_request = (lengths > 0) & (slots > 0)
    slots = torch.where(valid_request, slots, 0).long()
    real = valid_request[:, None] & (steps < lengths[:, None])
    rows = starts[:-1, None] + steps
    packed_hidden = hidden.index_select(0, rows.flatten().clamp(0, hidden.shape[0] - 1).long())
    packed_hidden = packed_hidden.view(batch, window, -1)
    conv, checkpoint = layer.kv_cache
    channels = 3 * layer.local_num_heads * layer.head_dim
    if conv.shape[-1] != channels:
        conv = conv.transpose(-1, -2)
    original = conv.index_select(0, slots).contiguous()
    compact_conv = original.clone()
    compact_checkpoint = checkpoint.index_select(0, slots).contiguous()
    counts = metadata.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens
    counts = torch.where(valid_request, counts, 1).int().contiguous()
    dense_ids = torch.arange(batch, device=hidden.device, dtype=torch.int32)
    bfg = layer.fused_bfg_proj
    beta, _, gate = bfg.weight.split(layer._fused_bfg_output_sizes, dim=0)
    records = layer.recoverssm_records
    correction = records.correction[:batch]
    decay = layer.recoverssm_decay_cache[:batch]
    key = torch.empty_like(decay, dtype=torch.float16)
    # BSH has fixed strides, including for graph padding. Only real positions are published below.
    result = op(
        packed_hidden,
        layer.in_proj_qkvgfab._full_kda_weight,
        bfg.f_a_weight,
        bfg.f_b_weight,
        beta,
        gate,
        layer.o_norm.weight.float(),
        layer.o_proj.weight,
        conv_weight,
        compact_conv,
        compact_checkpoint,
        correction,
        key,
        decay,
        dense_ids,
        counts,
        layer.A_log.reshape(-1)[: layer.local_num_heads].float().contiguous(),
        layer.dt_bias.reshape(layer.local_num_heads, layer.head_dim).float().contiguous(),
        None,
        None,
        layer.head_dim**-0.5,
        layer.gate_lower_bound,
        layer.o_norm.eps,
        int(torch.npu.current_stream().npu_stream),
    )[0]
    layer.recoverssm_normalized_key_cache[:batch].copy_(key)
    pos = torch.arange(conv.shape[1], device=hidden.device)
    write = valid_request[:, None] & (pos < (lengths + 2)[:, None])
    conv.index_copy_(0, slots, torch.where(write[:, :, None], compact_conv, original))
    # Padded rows get distinct unused destinations and never alias a live output.
    unused = hidden.shape[0] + torch.arange(batch * window, device=hidden.device).view(batch, window)
    index = torch.where(real, rows, unused).flatten().long()
    output = hidden.new_zeros((hidden.shape[0] + batch * window, hidden.shape[1]))
    output.index_copy_(0, index, result.reshape(-1, hidden.shape[1]))
    return output[: hidden.shape[0]]
