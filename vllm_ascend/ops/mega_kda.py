# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional KDA operators for Ascend 950 disaggregated decode."""

import ctypes
import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm_ascend.ops.mega_kda_upstream_state import prepare_upstream_state_plan, run_upstream_state_batch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

MEGA_KDA_HEAD_DIM = 128
MEGA_KDA_CONV_WIDTH = 4
_SPEC_CONV_PREFIX_ROWS = 2
_OP_NAMESPACE = "vllm_ascend_mega_kda"


@lru_cache(maxsize=1)
def load_mega_kda(package_dir: str) -> torch._ops.OpOverload:
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
        return getattr(torch.ops, _OP_NAMESPACE).npu_mega_kda_projected.default
    except AttributeError as exc:
        raise RuntimeError("The A5 fla_npu wheel must export npu_mega_kda_projected") from exc


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
) -> torch.Tensor:
    conv, state = layer.kv_cache
    rows, width = gate.shape
    heads = layer.local_num_heads
    head_dim = MEGA_KDA_HEAD_DIM
    if width != heads * head_dim or beta.shape != (rows, heads):
        raise ValueError("Unsupported MegaKDA projection geometry")
    if any(tensor.dtype != torch.bfloat16 for tensor in (gate, beta, conv_weight)):
        raise ValueError("MegaKDA requires BF16 gate, beta and packed convolution weights")
    if qkv.shape != (rows, 3 * width) or qkv.dtype != torch.bfloat16:
        raise ValueError("Projected MegaKDA requires BF16 QKV activations")

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
        qkv.contiguous(),
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

    def run(indices, prefix, slots, conv_slots, accepted, has_initial):
        if indices is None:
            activations = qkv, raw_gate, beta
        else:
            activations = tuple(tensor.index_select(0, indices) for tensor in (qkv, raw_gate, beta))
        return _run_projected(layer, op, *activations, conv_weight, prefix, slots, conv_slots, accepted, has_initial)

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


@dataclass(frozen=True)
class UpstreamMegaKDA:
    forward: Callable
    metadata: Callable


@lru_cache(maxsize=1)
def load_upstream_mega_kda(package_dir: str) -> UpstreamMegaKDA:
    """Resolve both public wrappers from the configured upstream operator wheel."""
    package = Path(package_dir).resolve() / "fla_npu"
    vendor = package / "opp/vendors/fla_npu_transformer"
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    if not (package / "libfla_npu_stable.so").is_file() or not opapi.is_file():
        raise RuntimeError("Install the Ascend 950 MegaRecurrentKda operator wheel in operator_package_dir")
    opp_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    if not any(entry and Path(entry).resolve() == vendor for entry in opp_paths):
        raise RuntimeError("Add the configured operator wheel's vendor to ASCEND_CUSTOM_OPP_PATH before startup")
    # Use the upstream wheel-layer registration protocol. The image's base FLA
    # package still owns all unrelated operators and the process-wide CANN setup.
    alias = "_vllm_ascend_upstream_mega_kda"
    spec = importlib.util.spec_from_file_location(
        alias, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    if spec is None:
        raise RuntimeError("Invalid MegaRecurrentKda Python package")
    provider = importlib.util.module_from_spec(spec)
    provider._WHEEL_LAYER_CONTEXT = {"namespace": alias.lstrip("_"), "libraries": [str(opapi)]}
    active = sys.modules.setdefault(alias, provider)
    if active.__file__ != provider.__file__:
        raise RuntimeError("A different MegaRecurrentKda wheel was already registered in this worker")
    ascendc = importlib.import_module(f"{alias}.ops.ascendc")
    try:
        forward = ascendc.npu_mega_recurrent_kda
        metadata = ascendc.npu_mega_recurrent_kda_metadata
    except AttributeError as exc:
        raise RuntimeError("Install the Ascend 950 MegaRecurrentKda wheel with MXFP8 and metadata support") from exc
    if not all(Path(inspect.getfile(fn)).resolve().is_relative_to(package) for fn in (forward, metadata)):
        raise RuntimeError("The imported MegaRecurrentKda wrappers do not belong to operator_package_dir")
    if "qkv_projection_weight_scale" not in inspect.signature(forward).parameters:
        raise RuntimeError("The MegaRecurrentKda wheel does not provide the MXFP8 weight interface")
    return UpstreamMegaKDA(forward, metadata)


def run_upstream_mega_kda(
    layer: "AscendKimiK3DeltaAttention",
    op: UpstreamMegaKDA,
    metadata: "GDNAttentionMetadata",
    hidden_states: torch.Tensor,
    weights: dict,
    *,
    only_spec: bool = False,
) -> torch.Tensor:
    """Return the rank-local output projection. The caller applies the TP collective."""
    from vllm.forward_context import get_forward_context

    output = torch.zeros_like(hidden_states)
    total = metadata.num_actual_tokens
    if not total:
        return output
    hidden = hidden_states[:total]
    conv, checkpoint = layer.kv_cache
    plan_cache = get_forward_context().additional_kwargs.setdefault("mega_kda_state_plans", {})

    def launch(x, conv_state, checkpoint_state, replay_u, replay_k, replay_decay, state_indices, accepted, **packed):
        return op.forward(
            hidden_states=x,
            **weights,
            conv_state=conv_state,
            checkpoint_state=checkpoint_state,
            replay_u=replay_u,
            replay_k=replay_k,
            replay_decay=replay_decay,
            state_indices=state_indices,
            conv_num_accepted_tokens=accepted,
            scale=layer.head_dim**-0.5,
            lower_bound=layer.gate_lower_bound,
            rms_norm_eps=layer.o_norm.eps,
            **packed,
        )

    def run_group(indices, conv_meta, slots, accepted, initial, width, records, group):
        tokens = hidden if indices is None else hidden.index_select(0, indices)
        prefix = conv_meta.query_start_loc
        conv_slots = conv_meta.cache_indices
        # One plan per batch, shared only by layers with the same metadata and page geometry.
        key = (id(metadata), group, tokens.shape[0], width, checkpoint.shape[0], conv.shape[0])
        if key not in plan_cache:
            plan_cache[key] = prepare_upstream_state_plan(
                prefix,
                slots,
                conv_slots,
                accepted,
                initial,
                num_tokens=tokens.shape[0],
                max_query_len=width,
                num_state_slots=checkpoint.shape[0],
                num_conv_slots=conv.shape[0],
                recover=records is not None,
                build_metadata=op.metadata,
            )
        result = run_upstream_state_batch(
            tokens,
            conv,
            checkpoint,
            prefix,
            slots,
            conv_slots,
            accepted,
            initial,
            launch,
            max_query_len=width,
            plan=plan_cache[key],
            records=records,
        )
        if indices is None:
            output[: result.shape[0]].copy_(result)
        else:
            output.index_copy_(0, indices.long(), result)

    spec = metadata.num_spec_decodes > 0
    non_spec = metadata.num_prefills > 0 or metadata.num_decodes > 0
    if not only_spec and metadata.num_prefills and metadata.num_prefill_tokens > metadata.num_prefills:
        raise ValueError("Long KDA prefill must be routed before entering the decode operator")
    if spec:
        spec_conv = metadata.spec_decode_metadata.spec_causal_conv1d
        accepted = spec_conv.num_accepted_tokens
        records = layer.recoverssm_records if layer.use_recoverssm else None
        if layer.use_recoverssm and records is None:
            raise RuntimeError("KDA replay records must be allocated before verification")
        run_group(
            metadata.spec_token_indx if non_spec else None,
            spec_conv,
            metadata.spec_state_indices_tensor,
            accepted,
            torch.ones_like(accepted),
            layer.num_spec + 1,
            records,
            "spec",
        )
    if non_spec and not only_spec:
        slots = metadata.non_spec_state_indices_tensor
        accepted = torch.ones(slots.shape[0], dtype=torch.int32, device=slots.device)
        if metadata.num_prefills:
            conv_meta = metadata.non_spec_prefill_metadata.causal_conv1d
            initial = conv_meta.initial_state_mode
        else:
            conv_meta = metadata.non_spec_decode_metadata.causal_conv1d
            initial = accepted
        run_group(
            metadata.non_spec_token_indx if spec else None,
            conv_meta,
            slots,
            accepted,
            initial,
            1,
            None,
            "decode",
        )
    return output
