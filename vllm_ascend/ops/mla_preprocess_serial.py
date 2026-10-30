# SPDX-License-Identifier: Apache-2.0
"""Simple MLA fallback: three Torch matmuls and three ACLNN Vector operators.

Persistent weights use logical matrices: down rows [KV,K-rope,Q], up [H*(P+64),L],
key [H,P,512]. Projection weights use persistent ND storage.
Prepare weights once when loading a model; no conversion is
hidden in this function. Only BF16/no_quant/split ND caches are supported.
"""

import torch
import torch_npu

FRACTAL_NZ = 29


def prepare_mla_preprocess_serial_weights(wdqkv, wuq, wuk):
    """Prepare model weights once; never call this in the timed forward path.

    Input down rows are [Q,KV,K-rope]; wuk is [H,512,P]. All BF16 values
    are preserved; only row order and matrix orientation change.
    """
    if any(w.dtype != torch.bfloat16 or w.device.type != "npu" for w in (wdqkv, wuq, wuk)):
        raise ValueError("MLA serial weights must be BF16 NPU tensors")
    rank = wuq.shape[1]
    down = torch.cat((wdqkv[rank:], wdqkv[:rank])).contiguous()
    return down, wuq, wuk.transpose(-1, -2).contiguous()


def prepare_mla_preprocess_single_token_weights(weight_down, weight_up):
    """Prepare extra NZ RHS weights once for the optional T=1 path.

    Call with down/up returned by prepare_mla_preprocess_serial_weights.
    Keep the result explicitly and rebuild it whenever either weight changes.
    Logical BF16 values are unchanged; no global cache or runtime flag is used.
    """
    if any(w.ndim != 2 or w.dtype != torch.bfloat16 or w.device.type != "npu" for w in (weight_down, weight_up)):
        raise ValueError("single-token weights must be two-dimensional BF16 NPU tensors")
    if weight_down.device != weight_up.device:
        raise ValueError("single-token weights must share a device")
    return tuple(torch_npu.npu_format_cast(w.t().contiguous(), FRACTAL_NZ).t() for w in (weight_down, weight_up))


def mla_preprocess_serial(
    x,
    weight_down,
    weight_up,
    weight_key,
    gamma_q,
    gamma_kv,
    cos,
    sin,
    slots,
    cache,
    cache_rope,
    *,
    single_token_weights=None,
):
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise ValueError("MLA serial currently supports BF16 NPU tensors")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be present or absent")
    if x.shape[0] == 1 and single_token_weights is not None:
        weight_down, weight_up = single_token_weights
    heads, nope, _ = weight_key.shape
    down = torch.matmul(x, weight_down.t())
    q_latent = torch.ops._C_ascend.mla_serial_norm_cache(down, gamma_q, gamma_kv, cos, sin, slots, cache, cache_rope)
    up = torch.matmul(q_latent, weight_up.t())
    if x.shape[0] == 1:
        # A singleton token axis makes the BMM result's final layout a view.
        query, q_rope = torch.ops._C_ascend.mla_serial_q_prepare(up, heads, cos, sin)
        key = torch.bmm(query.view(heads, 1, nope), weight_key)
        return key.view(1, heads, 512), q_rope, cache, cache_rope
    query = torch.ops._C_ascend.mla_serial_q_pack(up, heads)
    key = torch.bmm(query, weight_key)
    q_nope, q_rope = torch.ops._C_ascend.mla_serial_q_finalize(up, key, cos, sin)
    return q_nope, q_rope, cache, cache_rope
