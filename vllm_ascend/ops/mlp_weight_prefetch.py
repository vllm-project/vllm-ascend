# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Dense MLP prefetch with stream dependencies contained in compute ops.

Each op returns the real vector computation result consumed by the next
projection. This keeps both stream fences in the compiled graph without
aliasing an input or relying on unused, side-effect-only custom ops.
"""

import torch
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.weight_prefetch import MAX_PREFETCH_WEIGHT_SIZE, layer_idx_from_prefix, layers_of_model_instance
from vllm_ascend.utils import enable_custom_op, get_weight_prefetch_method, npu_stream_switch, prefetch_stream

MLP_PREFETCH_TOKEN_THRESHOLD = 500


def _attention_gate_up_target(prefix):
    """Resolve only the current decoder's unfused Attention output reduction."""
    from vllm_ascend.ops.linear_op import MatmulAllreduceRowParallelOp

    method = get_weight_prefetch_method()
    if method is None or not method.use_qwen_dense_mlp_prefetch:
        return None
    if not prefix.endswith((".self_attn.o_proj", ".linear_attn.out_proj")):
        return None
    idx = layer_idx_from_prefix(prefix)
    layers = layers_of_model_instance(_EXTRA_CTX.model_instance)
    if layers is None or idx is None or not 0 <= idx < len(layers):
        return None
    layer = layers[idx]
    attention = (
        getattr(layer, "self_attn", None)
        if prefix.endswith(".self_attn.o_proj")
        else getattr(layer, "linear_attn", None)
    )
    projection = (
        getattr(attention, "o_proj", None)
        if prefix.endswith(".self_attn.o_proj")
        else getattr(attention, "out_proj", None)
    )
    op = getattr(projection, "custom_op", None)
    if (
        not isinstance(op, MatmulAllreduceRowParallelOp)
        or projection.prefix != prefix
        or not op.reduce_results
        or op.tp_size <= 1
        or not hasattr(layer.mlp, "gate_up_proj")
    ):
        return None
    return layer.mlp.gate_up_proj.weight, float(method.mlp.prefetch_ratio.get("gate_up", 0))


def _next_layer_input_proj_target(prefix):
    """Resolve a down_proj to the NEXT decoder layer's input projection.

    The MLP down_proj all-reduce is the last collective of layer N, and the
    next large weight read is layer N+1's input projection, one AddRmsNorm
    later. That is the same fork/consume shape the gate_up prefetch already
    uses at o_proj, so reuse it. Measured at TP4: 22.48us idle window against
    a 17.25us MTE2 for in_proj_qkvz (40MiB).
    """
    from vllm_ascend.ops.linear_op import MatmulAllreduceRowParallelOp

    method = get_weight_prefetch_method()
    if method is None or not method.is_qwen_dense:
        return None
    if not method.attn.enable or not prefix.endswith(".mlp.down_proj"):
        return None
    idx = layer_idx_from_prefix(prefix)
    layers = layers_of_model_instance(_EXTRA_CTX.model_instance)
    if layers is None or idx is None or not 0 <= idx + 1 < len(layers):
        return None
    # Qwen3.5 alternates linear_attention and full_attention layers.
    nxt = layers[idx + 1]
    attention = getattr(nxt, "linear_attn", None) or getattr(nxt, "self_attn", None)
    projection = getattr(attention, "in_proj_qkvz", None) or getattr(attention, "qkv_proj", None)
    weight = getattr(projection, "weight", None)
    if weight is None:
        return None
    # If this down_proj does not route through matmul_and_reduce the fork never
    # runs, so report no target rather than silently doing nothing.
    down = getattr(getattr(layers[idx], "mlp", None), "down_proj", None)
    if not isinstance(getattr(down, "custom_op", None), MatmulAllreduceRowParallelOp):
        return None
    return weight, float(method.attn.prefetch_ratio.get("qkv", 0))


def _all_reduce_prefetch_target(prefix):
    """Pick what this row-parallel all-reduce should overlap a prefetch of.

    Two disjoint fork points, both landing one AddRmsNorm before their reader:
      o_proj / out_proj -> this layer's mlp.gate_up_proj
      mlp.down_proj     -> next layer's in_proj_qkvz / qkv_proj
    """
    return _attention_gate_up_target(prefix) or _next_layer_input_proj_target(prefix)


def attention_all_reduce_with_prefetch(output, prefix):
    # Called after MatMul. Down-projection submits AllReduce before prefetch;
    # attention projections retain prefetch-first ordering.
    target = _all_reduce_prefetch_target(prefix)
    if target is not None and prefix.endswith(".mlp.down_proj"):
        weight, ratio = target
        size = min(int(weight.numel() * weight.element_size() * ratio), MAX_PREFETCH_WEIGHT_SIZE)
        if size > 0 and 0 < output.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD:
            compute = torch_npu.npu.current_stream()
            stream = prefetch_stream()
            # Snapshot the Down MatMul boundary BEFORE submitting AllReduce.
            # This wait must not include the subsequent collective.
            stream.wait_stream(compute)
            output = tensor_model_parallel_all_reduce(output)
            with npu_stream_switch(stream):
                # The stream wait supplies readiness. Do not depend on the
                # reduction tensor, which the collective may update in place.
                torch_npu.npu_prefetch(weight, None, size, 0)
            _finish_prefetch(stream)
            return output
        return tensor_model_parallel_all_reduce(output)
    stream = _start_prefetch(target[0], output, target[1]) if target is not None else None
    output = tensor_model_parallel_all_reduce(output)
    _finish_prefetch(stream)
    return output


def _prefetch_size(weight: torch.Tensor, ratio: float) -> int:
    return min(int(weight.numel() * weight.element_size() * ratio), MAX_PREFETCH_WEIGHT_SIZE)


def _start_prefetch(
    weight: torch.Tensor, dependency: torch.Tensor, ratio: float, num_tokens: int | None = None
):
    # Decide inside the op: profile_run may trace at a prefill size while
    # ACL graphs are subsequently captured at smaller decode sizes.
    size = _prefetch_size(weight, ratio)
    tokens = dependency.shape[0] if num_tokens is None else num_tokens
    if size <= 0 or not 0 < tokens < MLP_PREFETCH_TOKEN_THRESHOLD:
        return None
    compute = torch_npu.npu.current_stream()
    stream = prefetch_stream()
    stream.wait_stream(compute)
    with npu_stream_switch(stream):
        torch_npu.npu_prefetch(weight, dependency, size, 0)
    return stream


def _finish_prefetch(stream):
    if stream is not None:
        torch_npu.npu.current_stream().wait_stream(stream)


def prefetch_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    ratio: float,
    attention_prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_ascend.compilation.passes.allreduce_rmsnorm_fusion_pass import ALLREDUCE_NORM_FUSE_THRESHOLD

    # Repeat the runtime eligibility check, not a Python trace-time token
    # decision. Other reductions retain the Norm-overlap fallback.
    prefetched = (
        attention_prefix != ""
        and x.shape[0] < ALLREDUCE_NORM_FUSE_THRESHOLD
        and _attention_gate_up_target(attention_prefix) is not None
    )
    stream = None if prefetched else _start_prefetch(weight, x, ratio)
    if enable_custom_op():
        output, _, residual_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, 1.0 + norm_weight, None, epsilon
        )
    else:
        output, _, residual_out = torch_npu.npu_add_rms_norm(x, residual, 1.0 + norm_weight, epsilon)
    _finish_prefetch(stream)
    return output, residual_out


def _prefetch_gemma_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    ratio: float,
    attention_prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


def prefetch_swiglu(x: torch.Tensor, weight: torch.Tensor, ratio: float) -> torch.Tensor:
    stream = _start_prefetch(weight, x, ratio)
    output = torch_npu.npu_swiglu(x)
    _finish_prefetch(stream)
    return output


def attention_gate_with_prefetch(
    attn_output: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, ratio: float
) -> torch.Tensor:
    """Apply full_attention's output gate, then join o_proj's prefetch.

    triton_split_qkv_rmsnorm_mrope() issued that fetch before the attention
    core. This is the last real computation before o_proj reads the weight, so
    it is where the fence belongs -- the same issue/join split GDN uses for
    linear_attn's out_proj. Returning the real gated result keeps the fence in
    the compiled graph. The guard must match the issue site's.
    """
    output = attn_output * torch.sigmoid(gate)
    if _prefetch_size(weight, ratio) > 0 and 0 < attn_output.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD:
        _finish_prefetch(prefetch_stream())
    return output


def _attention_gate_with_prefetch_fake(
    attn_output: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, ratio: float
) -> torch.Tensor:
    return torch.empty_like(attn_output)


def prefetch_gated_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    ratio: float,
    num_tokens: int,
) -> torch.Tensor:
    """Join out_proj prefetch launched inside the GDN core, after gated norm.

    The real norm output keeps the join in the compiled graph. Token counts
    refer to tokens, not the flattened token/head rows used by this norm.
    """
    from vllm_ascend.ops.layernorm import LayerNormFn

    out = LayerNormFn.apply(
        x, norm_weight, None, z, eps, None if group_size < 0 else group_size, norm_before_gate, True
    )
    if (
        ratio > 0
        and weight.numel() * weight.element_size() * ratio >= 1
        and 0 < num_tokens < MLP_PREFETCH_TOKEN_THRESHOLD
    ):
        _finish_prefetch(prefetch_stream())
    return out


def _prefetch_gated_norm_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    ratio: float,
    num_tokens: int,
) -> torch.Tensor:
    return torch.empty_like(x)


def _prefetch_swiglu_fake(x: torch.Tensor, weight: torch.Tensor, ratio: float) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def dense_mlp_with_prefetch(layer, hidden_states, residual, method):
    """Use this decoder's weights directly, including for PP and MTP layers."""
    mlp = layer.mlp
    norm = layer.post_attention_layernorm
    attention = getattr(layer, "self_attn", None)
    projection = getattr(attention, "o_proj", None)
    if projection is None:
        projection = getattr(getattr(layer, "linear_attn", None), "out_proj", None)
    attention_prefix = getattr(projection, "prefix", "")
    residual = torch.ops.vllm.maybe_chunk_residual(hidden_states, residual)
    hidden_states, residual = torch.ops.vllm.prefetch_gemma_rms_norm(
        hidden_states,
        residual,
        norm.weight,
        mlp.gate_up_proj.weight,
        norm.variance_epsilon,
        float(method.mlp.prefetch_ratio.get("gate_up", 0)),
        attention_prefix,
    )
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    hidden_states = torch.ops.vllm.prefetch_swiglu(
        gate_up, mlp.down_proj.weight, float(method.mlp.prefetch_ratio.get("down", 0))
    )
    hidden_states, _ = mlp.down_proj(hidden_states)
    return hidden_states, residual


direct_register_custom_op(
    op_name="prefetch_gemma_rms_norm",
    op_func=prefetch_gemma_rms_norm,
    fake_impl=_prefetch_gemma_rms_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="attention_gate_with_prefetch",
    op_func=attention_gate_with_prefetch,
    fake_impl=_attention_gate_with_prefetch_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="prefetch_gated_norm",
    op_func=prefetch_gated_norm,
    fake_impl=_prefetch_gated_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="prefetch_swiglu",
    op_func=prefetch_swiglu,
    fake_impl=_prefetch_swiglu_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
