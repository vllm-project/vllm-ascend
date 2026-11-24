# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional KDA operators for Ascend 950 disaggregated decode."""

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

from vllm_ascend.ops.mega_kda_upstream_state import (
    prepare_token_layout,
    prepare_upstream_state_plan,
    run_upstream_state_batch,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

MEGA_KDA_HEAD_DIM = 128
MEGA_KDA_CONV_WIDTH = 4


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


def _tensor_view_key(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    # Keep the storage alive for this forward. Inference-mode views may have
    # no _base; two differently offset slices must never share a state plan.
    return (tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride(), tensor.dtype)


def run_upstream_mega_kda(
    layer: "AscendKimiK3DeltaAttention",
    op: UpstreamMegaKDA,
    metadata: "GDNAttentionMetadata",
    hidden_states: torch.Tensor,
    weights: dict,
    *,
    output: torch.Tensor,
    only_spec: bool = False,
) -> None:
    """Write the rank-local projection into the caller's output. Padding stays zero."""
    from vllm.forward_context import get_forward_context

    output.zero_()
    total = metadata.num_actual_tokens
    if not total:
        return
    hidden = hidden_states[:total]
    conv, checkpoint = layer.kv_cache
    context = get_forward_context().additional_kwargs
    layouts = context.setdefault("mega_kda_token_layouts", {})
    plans = context.setdefault("mega_kda_state_plans", {})

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

    def run_group(indices, conv_meta, slots, accepted, initial, width):
        num_tokens = hidden.size(0) if indices is None else indices.numel()
        if num_tokens == 0:
            return
        prefix, conv_slots = conv_meta.query_start_loc, conv_meta.cache_indices
        layout_key = (_tensor_view_key(prefix), num_tokens, width)
        if layout_key not in layouts:
            layouts[layout_key] = prepare_token_layout(prefix, num_tokens, width, op.metadata)
        key = (
            layout_key,
            _tensor_view_key(slots),
            _tensor_view_key(conv_slots),
            _tensor_view_key(accepted),
            _tensor_view_key(initial),
            _tensor_view_key(indices),
            checkpoint.size(0),
            conv.size(0),
        )
        if key not in plans:
            plans[key] = prepare_upstream_state_plan(
                layouts[layout_key],
                slots,
                conv_slots,
                accepted,
                initial,
                group_indices=indices,
                max_query_len=width,
                num_state_slots=checkpoint.size(0),
                num_conv_slots=conv.size(0),
            )
        run_upstream_state_batch(hidden, conv, checkpoint, plans[key], launch, output)

    spec = metadata.num_spec_decodes > 0
    non_spec = metadata.num_prefills > 0 or metadata.num_decodes > 0
    if not only_spec and metadata.num_prefills and metadata.num_prefill_tokens > metadata.num_prefills:
        raise ValueError("Long KDA prefill must be routed before entering the decode operator")
    if spec:
        spec_conv = metadata.spec_decode_metadata.spec_causal_conv1d
        accepted = spec_conv.num_accepted_tokens
        run_group(
            metadata.spec_token_indx if non_spec else None,
            spec_conv,
            metadata.spec_state_indices_tensor,
            accepted,
            None,
            layer.num_spec + 1,
        )
    if non_spec and not only_spec:
        slots = metadata.non_spec_state_indices_tensor
        if metadata.num_prefills:
            conv_meta = metadata.non_spec_prefill_metadata.causal_conv1d
            initial = conv_meta.initial_state_mode
        else:
            conv_meta = metadata.non_spec_decode_metadata.causal_conv1d
            initial = None
        run_group(
            metadata.non_spec_token_indx if spec else None,
            conv_meta,
            slots,
            None,
            initial,
            1,
        )
