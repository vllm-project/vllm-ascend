#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Patch Qwen3MoE expert pruning support into vLLM.

The pruning table is passed by:
--additional-config '{"expert_list_file":"/path/to/experts.jsonl","expert_prune_ratio":0.5}'
"""

import json
from functools import lru_cache, wraps

import torch
from vllm.model_executor.models import qwen3_moe as qm


@lru_cache
def _load_ordered_expert_map(expert_list_file: str) -> dict[int, tuple[int, ...]]:
    ordered_map: dict[int, tuple[int, ...]] = {}

    with open(expert_list_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                layer_entry = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid Qwen3MoE expert pruning JSONL at line {line_num}.") from err

            if not isinstance(layer_entry, dict) or len(layer_entry) != 1:
                raise ValueError(
                    "Qwen3MoE expert pruning JSONL entries must be single-key objects, "
                    f"got {type(layer_entry).__name__} at line {line_num}."
                )

            layer_idx, expert_ids = next(iter(layer_entry.items()))
            try:
                parsed_layer_idx = int(layer_idx)
            except ValueError as err:
                raise ValueError(f"Invalid Qwen3MoE expert pruning layer id: {layer_idx!r}") from err

            if parsed_layer_idx in ordered_map:
                raise ValueError(f"Qwen3MoE expert pruning layer {parsed_layer_idx} appears more than once.")
            if not isinstance(expert_ids, list) or not all(isinstance(expert_id, int) for expert_id in expert_ids):
                raise ValueError(
                    f"Qwen3MoE expert pruning entry must contain integer expert ids for layer {layer_idx}."
                )
            if len(set(expert_ids)) != len(expert_ids):
                raise ValueError(f"Qwen3MoE expert pruning layer {layer_idx} contains duplicate expert ids.")

            ordered_map[parsed_layer_idx] = tuple(expert_ids)

    if not ordered_map:
        raise ValueError("Qwen3MoE expert pruning JSONL file is empty.")

    return ordered_map


def _get_expert_list_file(vllm_config) -> str | None:
    return vllm_config.additional_config.get("expert_list_file")


def _get_expert_prune_ratio(vllm_config) -> float:
    ratio = vllm_config.additional_config.get("expert_prune_ratio")
    if ratio is None:
        raise ValueError("Qwen3MoE expert pruning requires additional_config['expert_prune_ratio'].")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise ValueError("Qwen3MoE expert_prune_ratio must be a number.")
    ratio = float(ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("Qwen3MoE expert_prune_ratio must be in [0, 1).")
    return ratio


def _get_pruned_expert_count(total_num_experts: int, top_k: int, prune_ratio: float) -> int:
    pruned_expert_count = int(total_num_experts * (1.0 - prune_ratio))
    if pruned_expert_count <= 0:
        raise ValueError(f"Qwen3MoE expert_prune_ratio={prune_ratio} prunes all {total_num_experts} experts.")
    if top_k > pruned_expert_count:
        raise ValueError(
            f"Qwen3MoE keeps {pruned_expert_count} experts with expert_prune_ratio={prune_ratio}, "
            f"but num_experts_per_tok={top_k}."
        )
    return pruned_expert_count


def _get_pruned_expert_ids(
    vllm_config,
    layer_idx: int,
    total_num_experts: int,
    top_k: int,
) -> tuple[int, ...] | None:
    expert_list_file = _get_expert_list_file(vllm_config)
    if not expert_list_file:
        return None

    ordered_map = _load_ordered_expert_map(expert_list_file)
    if layer_idx not in ordered_map:
        raise ValueError(f"Qwen3MoE expert pruning file is missing an entry for layer {layer_idx}.")

    prune_ratio = _get_expert_prune_ratio(vllm_config)
    pruned_expert_count = _get_pruned_expert_count(total_num_experts, top_k, prune_ratio)
    ordered_expert_ids = ordered_map[layer_idx]
    if len(ordered_expert_ids) < pruned_expert_count:
        raise ValueError(
            f"Qwen3MoE layer {layer_idx} provides {len(ordered_expert_ids)} ordered experts, "
            f"but expert_prune_ratio={prune_ratio} requires {pruned_expert_count}."
        )

    expert_ids = ordered_expert_ids[:pruned_expert_count]
    if any(expert_id < 0 or expert_id >= total_num_experts for expert_id in expert_ids):
        raise ValueError(
            f"Qwen3MoE layer {layer_idx} expert ids must be in [0, {total_num_experts}). Got {expert_ids}."
        )

    return expert_ids


def _get_expert_pruning_map(vllm_config) -> dict[int, tuple[int, ...]] | None:
    expert_list_file = _get_expert_list_file(vllm_config)
    if expert_list_file is None:
        return None

    config = vllm_config.model_config.hf_text_config
    ordered_map = _load_ordered_expert_map(expert_list_file)
    return {
        layer_idx: _get_pruned_expert_ids(
            vllm_config,
            layer_idx,
            config.num_experts,
            config.num_experts_per_tok,
        )
        for layer_idx in ordered_map
    }


def _maybe_prune_gate_weight(
    name: str,
    loaded_weight: torch.Tensor,
    expert_pruning_map: dict[int, tuple[int, ...]] | None,
) -> torch.Tensor:
    if expert_pruning_map is None or not name.endswith(".mlp.gate.weight"):
        return loaded_weight

    layer_idx = qm.extract_layer_index(name)
    expert_ids = expert_pruning_map.get(layer_idx)
    if expert_ids is None:
        return loaded_weight

    if max(expert_ids) >= loaded_weight.shape[0]:
        raise ValueError(
            f"Cannot prune gate weight {name}: expert id {max(expert_ids)} "
            f"is out of bounds for loaded shape {tuple(loaded_weight.shape)}."
        )

    index = torch.tensor(expert_ids, dtype=torch.long, device=loaded_weight.device)
    return loaded_weight.index_select(0, index)


def _patched_sparse_moe_block_init(self, vllm_config, prefix: str = ""):
    config = vllm_config.model_config.hf_text_config
    original_num_experts = config.num_experts
    layer_idx = qm.extract_layer_index(prefix)
    pruned_expert_ids = _get_pruned_expert_ids(
        vllm_config,
        layer_idx,
        original_num_experts,
        config.num_experts_per_tok,
    )

    if pruned_expert_ids is None:
        original_init = qm.Qwen3MoeSparseMoeBlock._ascend_original_init
        return original_init(self, vllm_config, prefix)

    self.original_num_experts = original_num_experts
    self.layer_idx = layer_idx
    self.pruned_expert_ids = pruned_expert_ids

    try:
        config.num_experts = len(pruned_expert_ids)
        original_init = qm.Qwen3MoeSparseMoeBlock._ascend_original_init
        return original_init(self, vllm_config, prefix)
    finally:
        config.num_experts = original_num_experts


def _patched_get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
    if getattr(self, "expert_pruning_map", None) is not None:
        base_layer = "base_layer." if any(".base_layer." in name for name, _ in self.named_parameters()) else ""
        remapped_suffixes = ("weight", "weight_scale")
        return [
            (
                f"layers.{layer_idx}.mlp.experts.{base_layer}w13_{suffix}"
                if weight_name in ["gate_proj", "up_proj"]
                else f"layers.{layer_idx}.mlp.experts.{base_layer}w2_{suffix}",
                f"layers.{layer_idx}.mlp.experts.{original_expert_id}.{weight_name}.{base_layer}{suffix}",
                pruned_expert_id,
                shard_id,
            )
            for layer_idx, expert_ids in sorted(self.expert_pruning_map.items())
            for pruned_expert_id, original_expert_id in enumerate(expert_ids)
            for suffix in remapped_suffixes
            for shard_id, weight_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]

    original_get_expert_mapping = qm.Qwen3MoeModel._ascend_original_get_expert_mapping
    return original_get_expert_mapping(self)


def _patch_qwen3_moe_pruning() -> None:
    if hasattr(qm, "_load_ordered_expert_map"):
        return
    if getattr(qm.Qwen3MoeModel, "_ascend_expert_pruning_patched", False):
        return

    qm.Qwen3MoeSparseMoeBlock._ascend_original_init = qm.Qwen3MoeSparseMoeBlock.__init__
    qm.Qwen3MoeSparseMoeBlock.__init__ = _patched_sparse_moe_block_init

    original_model_init = qm.Qwen3MoeModel.__init__
    original_load_weights = qm.Qwen3MoeModel.load_weights
    qm.Qwen3MoeModel._ascend_original_get_expert_mapping = qm.Qwen3MoeModel.get_expert_mapping

    @wraps(original_model_init)
    def patched_model_init(self, *args, **kwargs):
        original_model_init(self, *args, **kwargs)
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        self.expert_pruning_map = _get_expert_pruning_map(vllm_config)

    @wraps(original_load_weights)
    def patched_load_weights(self, weights):
        def pruned_weights():
            for name, loaded_weight in weights:
                if getattr(self, "expert_pruning_map", None) is not None and ".mlp.experts." in name and (
                    ".weight_offset" in name or "_weight_offset" in name
                ):
                    continue
                yield name, _maybe_prune_gate_weight(name, loaded_weight, getattr(self, "expert_pruning_map", None))

        return original_load_weights(self, pruned_weights())

    qm.Qwen3MoeModel.__init__ = patched_model_init
    qm.Qwen3MoeModel.get_expert_mapping = _patched_get_expert_mapping
    qm.Qwen3MoeModel.load_weights = patched_load_weights
    qm.Qwen3MoeModel._ascend_expert_pruning_patched = True

_patch_qwen3_moe_pruning()
