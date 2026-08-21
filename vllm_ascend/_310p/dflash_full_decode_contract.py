# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Explicit retained-input inventory for 310P DFlash full-decode graphs.

The ACL graph call signature exposes only model arguments. Full graphs also
retain attention metadata, FIA graph-task parameters, and runner/proposer
buffers. This module deliberately lists those sources by semantic role. It
does not recursively walk arbitrary process state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_ascend._310p.graph_input_contract import GraphInputSource


class FullDecodeContractInventoryError(RuntimeError):
    """Raised when graph-visible state is outside the explicit inventory."""


_ATTENTION_TENSOR_FIELDS = (
    "attn_mask",
    "seq_lens",
    "query_start_loc",
    "block_tables",
    "slot_mapping",
    "positions",
    "group_len",
    "group_key_idx",
    "group_key_cache_idx",
    "dcp_mtp_attn_mask",
    # Upstream GDNAttentionMetadata. The 310P builder binds these to
    # descriptor-bounded buffers before a FULL graph is captured.
    "has_initial_state",
    "spec_query_start_loc",
    "non_spec_query_start_loc",
    "spec_state_indices_tensor",
    "non_spec_state_indices_tensor",
    "spec_sequence_masks",
    "spec_token_indx",
    "non_spec_token_indx",
    "num_accepted_tokens",
    "chunk_indices",
    "chunk_offsets",
    "prefill_query_start_loc",
    "prefill_state_indices",
    "prefill_has_initial_state",
    "batch_ptr",
    "token_chunk_offset_ptr",
)
_ATTENTION_NESTED_FIELDS = (
    "decode_meta",
    "prefill",
    "kvcomp_metadata",
)
_HOST_ONLY_TENSOR_FIELDS = (
    "query_lens_cpu",
    "seq_lens_cpu",
    "positions_cpu",
    "num_computed_tokens_cpu",
)
_GRAPH_PARAM_TENSOR_ROLES = {
    0: "query",
    1: "key_cache",
    2: "value_cache",
    3: "block_table",
    4: "attention_mask",
    11: "output",
    12: "softmax_lse",
    16: "key_antiquant_scale",
    17: "key_antiquant_offset",
    18: "value_antiquant_scale",
    19: "value_antiquant_offset",
}


def _source(
    role: str,
    tensor: torch.Tensor,
    *,
    ownership: str,
    bounded_view: bool,
) -> GraphInputSource:
    # NPU graph inputs are required to retain at least the allocator ABI's
    # 16-byte address alignment. The element size is also part of the rule for
    # unusual dtypes with a wider natural alignment.
    required_alignment = max(16, tensor.element_size())
    return GraphInputSource(
        role=role,
        tensor=tensor,
        ownership=ownership,
        required_alignment=required_alignment,
        alignment_source="Ascend-NPU-allocator-ABI-minimum-16-bytes",
        mutable=True,
        bounded_view=bounded_view,
    )


def _object_items(value: Any):
    if isinstance(value, Mapping):
        return value.items()
    try:
        return vars(value).items()
    except TypeError:
        return ()


def _metadata_sources(
    *,
    component: str,
    metadata_by_layer: Mapping[str, Any],
    step: int | None,
) -> list[GraphInputSource]:
    prefix = f"{component}.attention"
    if step is not None:
        prefix += f".step{step}"
    sources: list[GraphInputSource] = []

    for layer_name, metadata in metadata_by_layer.items():
        known_tensor_names: set[str] = set()
        for field_name in _ATTENTION_TENSOR_FIELDS:
            tensor = getattr(metadata, field_name, None)
            if isinstance(tensor, torch.Tensor):
                known_tensor_names.add(field_name)
                sources.append(
                    _source(
                        f"{prefix}.{layer_name}.{field_name}",
                        tensor,
                        ownership="attention-metadata",
                        bounded_view=True,
                    )
                )

        for nested_name in _ATTENTION_NESTED_FIELDS:
            nested = getattr(metadata, nested_name, None)
            if nested is None:
                continue
            for field_name, tensor in _object_items(nested):
                if isinstance(tensor, torch.Tensor):
                    known_tensor_names.add(f"{nested_name}.{field_name}")
                    sources.append(
                        _source(
                            f"{prefix}.{layer_name}.{nested_name}.{field_name}",
                            tensor,
                            ownership="attention-metadata",
                            bounded_view=True,
                        )
                    )

        ignored_names = set(_HOST_ONLY_TENSOR_FIELDS)
        ignored_names.update(_ATTENTION_NESTED_FIELDS)
        for field_name, tensor in _object_items(metadata):
            if not isinstance(tensor, torch.Tensor):
                continue
            if field_name in _ATTENTION_TENSOR_FIELDS:
                continue
            if field_name in ignored_names and tensor.device.type == "cpu":
                continue
            raise FullDecodeContractInventoryError(
                "unregistered graph-visible attention tensor: "
                f"component={component}, layer={layer_name}, "
                f"field={field_name}"
            )
    return sources


def _graph_param_sources(
    *,
    component: str,
    graph_params: Any,
    descriptor_num_tokens: int,
) -> list[GraphInputSource]:
    if graph_params is None:
        raise FullDecodeContractInventoryError(f"{component} graph parameters are not initialized")
    sources: list[GraphInputSource] = []
    params_for_size = graph_params.attn_params.get(descriptor_num_tokens, ())
    workspace = graph_params.workspaces.get(descriptor_num_tokens)
    if params_for_size and not isinstance(workspace, torch.Tensor):
        raise FullDecodeContractInventoryError(
            f"{component} graph workspace is missing for descriptor {descriptor_num_tokens}"
        )
    if isinstance(workspace, torch.Tensor):
        sources.append(
            _source(
                f"{component}.graph.workspace",
                workspace,
                ownership="acl-graph-parameter-cache",
                bounded_view=False,
            )
        )

    for attention_index, params in enumerate(params_for_size):
        for param_index, tensor in enumerate(params):
            if not isinstance(tensor, torch.Tensor):
                continue
            semantic_name = _GRAPH_PARAM_TENSOR_ROLES.get(
                param_index,
                f"tensor_{param_index}",
            )
            sources.append(
                _source(
                    f"{component}.graph.attention.{attention_index}.{semantic_name}",
                    tensor,
                    ownership="acl-graph-task-parameter",
                    bounded_view=False,
                )
            )
    return sources


def _buffer_sources(
    *,
    component: str,
    owner_name: str,
    buffers: Mapping[str, Any],
) -> list[GraphInputSource]:
    sources: list[GraphInputSource] = []
    for name, tensor in buffers.items():
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise FullDecodeContractInventoryError(f"{component}.{owner_name}.{name} is not a tensor")
        sources.append(
            _source(
                f"{component}.{owner_name}.{name}",
                tensor,
                ownership=f"{owner_name}-persistent-buffer",
                bounded_view=True,
            )
        )
    return sources


def _forward_sources(component: str, forward_context: Any) -> list[GraphInputSource]:
    sources: list[GraphInputSource] = []
    for name in ("input_ids", "num_tokens_across_dp", "mc2_mask"):
        tensor = getattr(forward_context, name, None)
        if isinstance(tensor, torch.Tensor):
            sources.append(
                _source(
                    f"{component}.forward.{name}",
                    tensor,
                    ownership="forward-context",
                    bounded_view=True,
                )
            )
    return sources


def build_target_full_decode_contract_sources(
    *,
    forward_context: Any,
    graph_params: Any,
    runner_buffers: Mapping[str, Any],
    descriptor_num_tokens: int,
) -> tuple[GraphInputSource, ...]:
    """Return the complete explicit target FULL retained-input inventory."""
    sources = _forward_sources("target", forward_context)
    metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(metadata, Mapping):
        raise FullDecodeContractInventoryError("target forward context has no attention metadata mapping")
    sources.extend(
        _metadata_sources(
            component="target",
            metadata_by_layer=metadata,
            step=None,
        )
    )
    sources.extend(
        _buffer_sources(
            component="target",
            owner_name="runner",
            buffers=runner_buffers,
        )
    )
    sources.extend(
        _graph_param_sources(
            component="target",
            graph_params=graph_params,
            descriptor_num_tokens=descriptor_num_tokens,
        )
    )
    return tuple(sources)


def build_draft_full_decode_contract_sources(
    *,
    forward_context: Any,
    graph_params: Any,
    proposer_buffers: Mapping[str, Any],
    descriptor_num_tokens: int,
) -> tuple[GraphInputSource, ...]:
    """Return the complete explicit merged-DFlash FULL retained inventory."""
    sources = _forward_sources("draft", forward_context)
    metadata_steps = getattr(forward_context, "draft_attn_metadatas", None)
    if not isinstance(metadata_steps, (list, tuple)) or not metadata_steps:
        raise FullDecodeContractInventoryError("draft forward context has no per-step attention metadata")
    for step, metadata in enumerate(metadata_steps):
        if not isinstance(metadata, Mapping):
            raise FullDecodeContractInventoryError(f"draft attention metadata step {step} is not a mapping")
        sources.extend(
            _metadata_sources(
                component="draft",
                metadata_by_layer=metadata,
                step=step,
            )
        )
    sources.extend(
        _buffer_sources(
            component="draft",
            owner_name="proposer",
            buffers=proposer_buffers,
        )
    )
    sources.extend(
        _graph_param_sources(
            component="draft",
            graph_params=graph_params,
            descriptor_num_tokens=descriptor_num_tokens,
        )
    )
    return tuple(sources)
