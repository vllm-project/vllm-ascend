# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Projected MegaKDA for paged KDA states on Ascend 950."""

import ctypes
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

MEGA_KDA_HEAD_DIM = 128
MEGA_KDA_CONV_WIDTH = 4
_SPEC_CONV_PREFIX_ROWS = 2
_OP_NAMESPACE = "vllm_ascend_mega_kda"
_MX_SCALE_PAIR_SIZE = 64


@dataclass(frozen=True)
class MxQkvProjection:
    """MXFP8 activation scales and the output-major checkpoint weights."""

    input_scale: torch.Tensor
    weight: torch.Tensor
    weight_scale: torch.Tensor


@lru_cache(maxsize=1)
def load_mega_kda(package_dir: str, op_name: str = "npu_mega_kda_projected") -> torch._ops.OpOverload:
    """Register the optional wheel before the worker captures an NPU graph."""
    package = Path(package_dir).resolve() / "fla_npu"
    launcher = package / "libfla_npu_stable.so"
    vendor = package / "opp/vendors/fla_npu_transformer"
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    if not launcher.is_file() or not opapi.is_file():
        raise RuntimeError("The MegaKDA package directory must contain the A5 projected operator wheel")
    opp_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    if not any(entry and Path(entry).resolve() == vendor for entry in opp_paths):
        raise RuntimeError("Add the MegaKDA OPP vendor directory to ASCEND_CUSTOM_OPP_PATH before worker startup")

    torch.ops.load_library(str(launcher))
    register = ctypes.CDLL(str(launcher)).fla_npu_stable_register
    register.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
    register.restype = ctypes.c_char_p
    error = register(_OP_NAMESPACE.encode(), os.fsencode(opapi))
    if error:
        raise RuntimeError(f"MegaKDA registration failed: {os.fsdecode(error)}")
    try:
        return getattr(getattr(torch.ops, _OP_NAMESPACE), op_name).default
    except AttributeError as exc:
        raise RuntimeError(f"The A5 fla_npu wheel must export {op_name}") from exc


def _validate_pool_layout(tensor: torch.Tensor, slot_shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tensor.dtype != dtype or tensor.shape[1:] != slot_shape:
        raise ValueError(f"MegaKDA requires cache shape [slots, {slot_shape}] with dtype {dtype}")
    stride = 1
    for size, actual_stride in zip(reversed(slot_shape), reversed(tensor.stride()[1:])):
        if stride != actual_stride:
            raise ValueError("MegaKDA requires contiguous values within each cache slot")
        stride *= size
    if tensor.stride(0) < stride:
        raise ValueError("MegaKDA cache slots must not overlap")


def _run_projected(
    layer: "AscendKimiK3DeltaAttention",
    op: torch._ops.OpOverload,
    qkv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    conv_weight: torch.Tensor,
    prefix: torch.Tensor,
    slots: torch.Tensor,
    conv_slots: torch.Tensor,
    accepted: torch.Tensor,
    has_initial: torch.Tensor,
    mx_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    mx_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    conv, state = layer.kv_cache
    rows, width = gate.shape
    heads = layer.local_num_heads
    head_dim = MEGA_KDA_HEAD_DIM
    if width != heads * head_dim or beta.shape != (rows, heads):
        raise ValueError("Unsupported MegaKDA projection geometry")
    if any(tensor.dtype != torch.bfloat16 for tensor in (gate, beta, conv_weight)):
        raise ValueError("MegaKDA requires BF16 gate, beta and packed convolution weights")
    if mx_weights is None:
        if qkv.shape != (rows, 3 * width) or qkv.dtype != torch.bfloat16:
            raise ValueError("Projected MegaKDA requires BF16 QKV activations")
        projection_args = (qkv.contiguous(),)
    else:
        assert mx_input_scale is not None
        weight, weight_scale = mx_weights
        projection_args = (qkv.contiguous(), weight, mx_input_scale.contiguous(), weight_scale)

    history = conv.shape[1]
    _validate_pool_layout(conv, (history, 3 * width), torch.bfloat16)
    _validate_pool_layout(state, (heads, head_dim, head_dim), torch.float32)
    if slots.ndim == 1:
        slots = slots[:, None]
    if conv_slots.ndim == 2:
        conv_slots = conv_slots[:, 0]
    batch, candidates = slots.shape
    if (
        prefix.shape != (batch + 1,)
        or conv_slots.shape != (batch,)
        or accepted.shape != (batch,)
        or has_initial.shape != (batch,)
        or not 1 <= candidates <= history - _SPEC_CONV_PREFIX_ROWS
        or conv_weight.shape != (MEGA_KDA_CONV_WIDTH, 3 * width)
    ):
        raise ValueError("MegaKDA request metadata does not match the cache or token geometry")

    # Padded requests can be skipped by the device kernel. Initialize their
    # outputs before applying the same gated norm used by the native path.
    out = torch.zeros_like(gate, memory_format=torch.contiguous_format)
    stream = torch.npu.current_stream(qkv.device).npu_stream
    assert layer.gate_lower_bound is not None
    return op(
        *projection_args,
        gate.contiguous(),
        beta.contiguous(),
        conv_weight,
        conv,
        state,
        prefix.to(torch.int32).contiguous(),
        slots.to(torch.int32).contiguous(),
        conv_slots.to(torch.int32).contiguous(),
        accepted.to(torch.int32).contiguous(),
        has_initial.to(torch.int32).contiguous(),
        layer.A_log.reshape(heads).float().contiguous(),
        layer.dt_bias.reshape(heads, head_dim).float().contiguous(),
        out,
        head_dim**-0.5,
        layer.gate_lower_bound,
        int(stream),
    )


def run_mega_kda(
    layer: "AscendKimiK3DeltaAttention",
    op: torch._ops.OpOverload,
    metadata: "GDNAttentionMetadata",
    qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    output_gate: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_weight: torch.Tensor,
    mx: MxQkvProjection | None = None,
) -> None:
    """Pass replay-time metadata through to the NPU without host readbacks."""
    total = metadata.num_actual_tokens
    has_spec = metadata.num_spec_decodes > 0
    has_non_spec = metadata.num_prefills > 0 or metadata.num_decodes > 0
    if not total or not (has_spec or has_non_spec):
        core_attn_out.zero_()
        return

    qkv = qkv[:total]
    raw_gate = raw_gate[0, :total].flatten(1)
    beta = beta[0, :total]
    mx_weights = None
    mx_input_scale = None
    if mx is not None:
        columns = qkv.size(-1)
        expected_weight = (3 * raw_gate.shape[1], columns)
        if columns % _MX_SCALE_PAIR_SIZE or mx.weight.shape != expected_weight:
            raise ValueError("Mixed MegaKDA requires the original output-major MXFP8 QKV weight")
        qkv = qkv.view(torch.uint8)
        scale_pairs = columns // _MX_SCALE_PAIR_SIZE
        mx_input_scale = mx.input_scale[:total].contiguous().view(torch.uint8)
        mx_weights = (
            mx.weight.contiguous().view(torch.uint8),
            mx.weight_scale.contiguous().view(torch.uint8).reshape(expected_weight[0], scale_pairs, 2),
        )

    def run(indices, prefix, slots, conv_slots, accepted, has_initial):
        if indices is None:
            activations = qkv, raw_gate, beta
        else:
            activations = tuple(tensor.index_select(0, indices) for tensor in (qkv, raw_gate, beta))
        args = (layer, op, *activations, conv_weight, prefix, slots, conv_slots, accepted, has_initial)
        if mx_input_scale is None:
            return _run_projected(*args)
        scales = mx_input_scale if indices is None else mx_input_scale.index_select(0, indices)
        return _run_projected(*args, mx_weights=mx_weights, mx_input_scale=scales)

    spec_out = non_spec_out = None
    if has_spec:
        assert metadata.spec_decode_metadata is not None
        assert metadata.spec_state_indices_tensor is not None
        spec = metadata.spec_decode_metadata.spec_causal_conv1d
        assert spec.num_accepted_tokens is not None
        spec_out = run(
            metadata.spec_token_indx if has_non_spec else None,
            spec.query_start_loc,
            metadata.spec_state_indices_tensor,
            spec.cache_indices,
            spec.num_accepted_tokens,
            torch.ones_like(spec.num_accepted_tokens, dtype=torch.int32),
        )

    if has_non_spec:
        slots = metadata.non_spec_state_indices_tensor
        assert slots is not None
        accepted = torch.ones_like(slots, dtype=torch.int32)
        if metadata.num_prefills > 0:
            assert metadata.non_spec_prefill_metadata is not None
            non_spec = metadata.non_spec_prefill_metadata.causal_conv1d
            has_initial = non_spec.initial_state_mode
            assert has_initial is not None
        else:
            assert metadata.non_spec_decode_metadata is not None
            non_spec = metadata.non_spec_decode_metadata.causal_conv1d
            has_initial = accepted
        non_spec_out = run(
            metadata.non_spec_token_indx if has_spec else None,
            non_spec.query_start_loc,
            slots,
            non_spec.cache_indices,
            accepted,
            has_initial,
        )

    if spec_out is not None and non_spec_out is not None:
        assert metadata.spec_token_indx is not None
        assert metadata.non_spec_token_indx is not None
        output = torch.zeros_like(raw_gate)
        output.index_copy_(0, metadata.spec_token_indx, spec_out)
        output.index_copy_(0, metadata.non_spec_token_indx, non_spec_out)
    else:
        output = spec_out if spec_out is not None else non_spec_out
        assert output is not None

    norm_input = output.reshape(1, total, layer.local_num_heads, MEGA_KDA_HEAD_DIM)
    layer.o_norm.forward_oot(norm_input, output_gate[:total], out=core_attn_out)
