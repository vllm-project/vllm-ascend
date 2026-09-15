# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _load_pruning_module():
    module_path = Path(__file__).parents[4] / "vllm_ascend" / "patch" / "worker" / "patch_qwen3_moe_pruning.py"
    spec = importlib.util.spec_from_file_location("patch_qwen3_moe_pruning_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pruning = _load_pruning_module()


def _write_expert_list(tmp_path, expert_table):
    expert_list_file = tmp_path / "expert_list.jsonl"
    lines = [json.dumps({str(layer_idx): expert_ids}) for layer_idx, expert_ids in expert_table.items()]
    expert_list_file.write_text("\n".join(lines), encoding="utf-8")
    pruning._load_ordered_expert_map.cache_clear()
    return str(expert_list_file)


def _make_vllm_config(expert_list_file=None, num_experts=8, top_k=2, prune_ratio=0.5):
    additional_config = {}
    if expert_list_file is not None:
        additional_config = {
            "expert_list_file": expert_list_file,
            "expert_prune_ratio": prune_ratio,
        }
    return SimpleNamespace(
        additional_config=additional_config,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_experts=num_experts,
                num_experts_per_tok=top_k,
            )
        ),
    )


def test_load_ordered_expert_map_parses_jsonl_layer_ids(tmp_path):
    expert_list_file = _write_expert_list(
        tmp_path,
        {
            "0": [3, 1, 5],
            "2": [4, 0],
        },
    )

    assert pruning._load_ordered_expert_map(expert_list_file) == {
        0: (3, 1, 5),
        2: (4, 0),
    }


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"layer0": [0]}', "Invalid Qwen3MoE expert pruning layer id"),
        ("[0, 1]", "single-key objects"),
        ('{"0": [0], "1": [1]}', "single-key objects"),
        ('{"0": ["0"]}', "integer expert ids"),
        ('{"0": [1, 1]}', "duplicate expert ids"),
        ("", "JSONL file is empty"),
    ],
)
def test_load_ordered_expert_map_rejects_invalid_jsonl(tmp_path, content, match):
    expert_list_file = tmp_path / "expert_list.jsonl"
    expert_list_file.write_text(content, encoding="utf-8")
    pruning._load_ordered_expert_map.cache_clear()

    with pytest.raises(ValueError, match=match):
        pruning._load_ordered_expert_map(str(expert_list_file))


@pytest.mark.parametrize(
    ("prune_ratio", "match"),
    [
        (None, "requires additional_config"),
        ("0.5", "must be a number"),
        (True, "must be a number"),
        (-0.1, r"must be in \[0, 1\)"),
        (1.0, r"must be in \[0, 1\)"),
    ],
)
def test_get_expert_prune_ratio_validates_config(prune_ratio, match):
    additional_config = {"expert_list_file": "experts.jsonl"}
    if prune_ratio is not None:
        additional_config["expert_prune_ratio"] = prune_ratio
    vllm_config = SimpleNamespace(additional_config=additional_config)

    with pytest.raises(ValueError, match=match):
        pruning._get_expert_prune_ratio(vllm_config)


def test_get_pruned_expert_count_uses_prune_ratio():
    assert pruning._get_pruned_expert_count(total_num_experts=8, top_k=2, prune_ratio=0.5) == 4
    assert pruning._get_pruned_expert_count(total_num_experts=8, top_k=2, prune_ratio=0.25) == 6

    with pytest.raises(ValueError, match="num_experts_per_tok=3"):
        pruning._get_pruned_expert_count(total_num_experts=8, top_k=3, prune_ratio=0.75)


def test_get_pruned_expert_ids_returns_none_when_disabled():
    vllm_config = _make_vllm_config(expert_list_file=None)

    assert pruning._get_pruned_expert_ids(vllm_config, layer_idx=0, total_num_experts=8, top_k=2) is None


def test_get_pruned_expert_ids_validates_layer_topk_and_bounds(tmp_path):
    expert_list_file = _write_expert_list(tmp_path, {"0": [7, 3, 5, 1]})
    vllm_config = _make_vllm_config(expert_list_file, num_experts=8, top_k=2, prune_ratio=0.5)

    assert pruning._get_pruned_expert_ids(vllm_config, layer_idx=0, total_num_experts=8, top_k=2) == (7, 3, 5, 1)

    with pytest.raises(ValueError, match="missing an entry for layer 1"):
        pruning._get_pruned_expert_ids(vllm_config, layer_idx=1, total_num_experts=8, top_k=2)

    short_expert_list_file = _write_expert_list(tmp_path, {"0": [7, 3]})
    short_vllm_config = _make_vllm_config(short_expert_list_file, num_experts=8, top_k=2, prune_ratio=0.5)
    with pytest.raises(ValueError, match="requires 4"):
        pruning._get_pruned_expert_ids(short_vllm_config, layer_idx=0, total_num_experts=8, top_k=2)

    bounds_expert_list_file = tmp_path / "bounds_expert_list.jsonl"
    bounds_expert_list_file.write_text(json.dumps({"0": [7, 3, 5, 1]}), encoding="utf-8")
    pruning._load_ordered_expert_map.cache_clear()
    bounds_vllm_config = _make_vllm_config(str(bounds_expert_list_file), num_experts=8, top_k=2, prune_ratio=0.5)
    with pytest.raises(ValueError, match=r"expert ids must be in \[0, 7\)"):
        pruning._get_pruned_expert_ids(bounds_vllm_config, layer_idx=0, total_num_experts=7, top_k=2)


def test_maybe_prune_gate_weight_selects_configured_expert_rows():
    gate_weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    pruned = pruning._maybe_prune_gate_weight(
        "model.layers.0.mlp.gate.weight",
        gate_weight,
        {0: (7, 3, 1)},
    )

    torch.testing.assert_close(pruned, gate_weight[[7, 3, 1]])


def test_maybe_prune_gate_weight_ignores_non_gate_or_unlisted_layer():
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    assert pruning._maybe_prune_gate_weight("model.layers.0.mlp.down_proj.weight", weight, {0: (3, 1)}) is weight
    assert pruning._maybe_prune_gate_weight("model.layers.1.mlp.gate.weight", weight, {0: (3, 1)}) is weight
    assert pruning._maybe_prune_gate_weight("model.layers.0.mlp.gate.weight", weight, None) is weight


def test_maybe_prune_gate_weight_rejects_out_of_bounds_expert_id():
    gate_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    with pytest.raises(ValueError, match="out of bounds"):
        pruning._maybe_prune_gate_weight(
            "model.layers.0.mlp.gate.weight",
            gate_weight,
            {0: (4,)},
        )


def test_sparse_moe_block_init_temporarily_uses_pruned_num_experts(tmp_path, monkeypatch):
    expert_list_file = _write_expert_list(tmp_path, {"0": [7, 3, 1, 5]})
    vllm_config = _make_vllm_config(expert_list_file, num_experts=8, top_k=2, prune_ratio=0.5)
    block = SimpleNamespace()
    observed_num_experts = []

    def fake_original_init(self, config, prefix):
        observed_num_experts.append(config.model_config.hf_text_config.num_experts)
        self.original_init_prefix = prefix
        return "initialized"

    monkeypatch.setattr(pruning.qm, "extract_layer_index", lambda prefix: 0)
    monkeypatch.setattr(pruning.qm.Qwen3MoeSparseMoeBlock, "_ascend_original_init", fake_original_init, raising=False)

    result = pruning._patched_sparse_moe_block_init(block, vllm_config, "model.layers.0.mlp")

    assert result == "initialized"
    assert observed_num_experts == [4]
    assert vllm_config.model_config.hf_text_config.num_experts == 8
    assert block.original_num_experts == 8
    assert block.pruned_expert_ids == (7, 3, 1, 5)


def test_sparse_moe_block_init_delegates_when_pruning_disabled(monkeypatch):
    vllm_config = _make_vllm_config(expert_list_file=None, num_experts=8, top_k=2)
    block = SimpleNamespace()
    observed_num_experts = []

    def fake_original_init(self, config, prefix):
        observed_num_experts.append(config.model_config.hf_text_config.num_experts)
        return "original"

    monkeypatch.setattr(pruning.qm, "extract_layer_index", lambda prefix: 0)
    monkeypatch.setattr(pruning.qm.Qwen3MoeSparseMoeBlock, "_ascend_original_init", fake_original_init, raising=False)

    assert pruning._patched_sparse_moe_block_init(block, vllm_config, "model.layers.0.mlp") == "original"
    assert observed_num_experts == [8]
    assert not hasattr(block, "pruned_expert_ids")


def test_get_expert_mapping_remaps_original_experts_to_pruned_slots():
    model = SimpleNamespace(
        expert_pruning_map={1: (7, 3)},
        named_parameters=lambda: [],
    )

    mapping = pruning._patched_get_expert_mapping(model)

    assert mapping == [
        ("layers.1.mlp.experts.w13_weight", "layers.1.mlp.experts.7.gate_proj.weight", 0, "w1"),
        ("layers.1.mlp.experts.w2_weight", "layers.1.mlp.experts.7.down_proj.weight", 0, "w2"),
        ("layers.1.mlp.experts.w13_weight", "layers.1.mlp.experts.7.up_proj.weight", 0, "w3"),
        ("layers.1.mlp.experts.w13_weight_scale", "layers.1.mlp.experts.7.gate_proj.weight_scale", 0, "w1"),
        ("layers.1.mlp.experts.w2_weight_scale", "layers.1.mlp.experts.7.down_proj.weight_scale", 0, "w2"),
        ("layers.1.mlp.experts.w13_weight_scale", "layers.1.mlp.experts.7.up_proj.weight_scale", 0, "w3"),
        ("layers.1.mlp.experts.w13_weight", "layers.1.mlp.experts.3.gate_proj.weight", 1, "w1"),
        ("layers.1.mlp.experts.w2_weight", "layers.1.mlp.experts.3.down_proj.weight", 1, "w2"),
        ("layers.1.mlp.experts.w13_weight", "layers.1.mlp.experts.3.up_proj.weight", 1, "w3"),
        ("layers.1.mlp.experts.w13_weight_scale", "layers.1.mlp.experts.3.gate_proj.weight_scale", 1, "w1"),
        ("layers.1.mlp.experts.w2_weight_scale", "layers.1.mlp.experts.3.down_proj.weight_scale", 1, "w2"),
        ("layers.1.mlp.experts.w13_weight_scale", "layers.1.mlp.experts.3.up_proj.weight_scale", 1, "w3"),
    ]
    assert not any("weight_offset" in weight_name for _, weight_name, _, _ in mapping)


def test_get_expert_mapping_delegates_without_pruning(monkeypatch):
    model = MagicMock(expert_pruning_map=None)
    original_get_expert_mapping = MagicMock(return_value=["original"])
    monkeypatch.setattr(
        pruning.qm.Qwen3MoeModel,
        "_ascend_original_get_expert_mapping",
        original_get_expert_mapping,
        raising=False,
    )

    assert pruning._patched_get_expert_mapping(model) == ["original"]
    original_get_expert_mapping.assert_called_once_with(model)
