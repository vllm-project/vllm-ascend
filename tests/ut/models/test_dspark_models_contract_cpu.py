# SPDX-License-Identifier: Apache-2.0
"""Execute model-side DSpark contracts without importing the NPU stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn


def load_model(filename, class_name, methods, parent=nn.Module, functions=(), **extra):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    selected.append(definition)
    namespace = {"torch": torch, "nn": nn, "Path": Path, "Parent": parent, **extra}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile("from __future__ import annotations\n" + ast.unparse(module), str(source), "exec"), namespace)
    return namespace[class_name], namespace


@pytest.mark.parametrize("wrapped", [False, True])
def test_gqa_selects_materialized_aux_without_changing_graph(wrapped):
    model, _ = load_model("qwen3_dspark.py", "AscendQwen3DSparkForCausalLM", {"configure_target_aux_hidden_capture"})
    setter = MagicMock()
    target = SimpleNamespace(set_dspark_aux_capture_materialized=setter, enforce_eager=False)
    outer = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    model().configure_target_aux_hidden_capture(outer)
    setter.assert_called_once_with(True)
    assert target.enforce_eager is False
    model().configure_target_aux_hidden_capture(SimpleNamespace())


@pytest.mark.parametrize("owned", [(), ("embed_tokens",), ("lm_head",), ("embed_tokens", "lm_head")])
def test_gqa_rotation_is_applied_once_and_preserves_owned_vocab_weights(owned):
    class Upstream(nn.Module):
        def load_weights(self, weights):
            self.loaded = dict(weights)
            self.has_own_embed_tokens = "embed_tokens.weight" in self.loaded
            self.has_own_lm_head = "lm_head.weight" in self.loaded
            return set(self.loaded)

    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotation_loader = MagicMock(return_value=rotation)
    vocab_loader = MagicMock()
    model, _ = load_model(
        "qwen3_dspark.py",
        "AscendQwen3DSparkForCausalLM",
        {"load_weights"},
        parent=Upstream,
        functions=("process_weight",),
        get_rotation_matrix=rotation_loader,
        load_quarot_target_layer=vocab_loader,
        TARGET_EMBED_WEIGHT_NAMES=("embed_tokens.weight",),
        TARGET_LM_HEAD_WEIGHT_NAMES=("lm_head.weight",),
    )
    draft = model()
    draft.config = SimpleNamespace(_ascend_target_rotation_path="target-rotation")
    draft.rotation_path = Path("draft-rotation")
    draft.target_model_path = Path("target")
    draft.model = SimpleNamespace(embed_tokens=object())
    draft.lm_head = object()
    fc = torch.arange(8, dtype=torch.float32).view(2, 4)
    own_weights = [(name + ".weight", torch.full((2, 2), 7.0)) for name in owned]
    loaded = draft.load_weights(iter([("fc.weight", fc), *own_weights]))
    expected = torch.cat((fc[:, :2] @ rotation, fc[:, 2:] @ rotation), dim=1)
    torch.testing.assert_close(draft.loaded["fc.weight"], expected)
    torch.testing.assert_close(fc, torch.arange(8, dtype=torch.float32).view(2, 4))
    rotation_loader.assert_called_once_with(Path("target-rotation"))
    assert vocab_loader.call_count == 2 - len(owned)
    assert draft.has_own_embed_tokens and draft.has_own_lm_head
    for name, value in own_weights:
        assert draft.loaded[name] is value
    assert loaded == {"fc.weight", *(name for name, _ in own_weights)}


def test_gqa_without_rotation_keeps_weights_and_upstream_sharing():
    class Upstream(nn.Module):
        def load_weights(self, weights):
            return dict(weights)

    rotation_loader = MagicMock(side_effect=AssertionError("unexpected rotation"))
    model, _ = load_model(
        "qwen3_dspark.py",
        "AscendQwen3DSparkForCausalLM",
        {"load_weights"},
        parent=Upstream,
        get_rotation_matrix=rotation_loader,
    )
    draft = model()
    draft.config, draft.rotation_path = SimpleNamespace(), None
    weight = torch.randn(2, 4)
    assert draft.load_weights(iter([("fc.weight", weight)]))["fc.weight"] is weight
    assert not hasattr(draft, "has_own_embed_tokens")
    rotation_loader.assert_not_called()


def mla_draft(**fields):
    model, _ = load_model("kimi_k3_dspark.py", "AscendK3DSparkForCausalLM", {"configure_target_aux_hidden_capture"})
    draft = model()
    draft.config = SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
    for name, value in fields.items():
        setattr(draft.config, name, value)
    return draft


def raw_target():
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=4, hidden_size=4), aux_hidden_state_layers=(1, 3)
        ),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_mla_selects_validated_raw_capture(wrapped, alias):
    target = raw_target()
    draft = mla_draft(dspark_target_layer_ids=[0, 2]) if alias else mla_draft()
    outer = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    draft.configure_target_aux_hidden_capture(outer)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(False)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_layer_ids", [], "requires target_layer_ids"),
        ("target_layer_ids", [0, 0], "Invalid"),
        ("target_layer_ids", [-1, 2], "Invalid"),
        ("target_layer_ids", [0, 9], "Invalid"),
        ("target_layer_ids", [1, 2], "boundaries do not match"),
        ("target_hidden_size", 8, "hidden sizes"),
        ("num_target_layers", 3, "num_target_layers"),
    ],
)
def test_mla_rejects_incompatible_aux_contract_before_changing_capture(field, value, message):
    target = raw_target()
    with pytest.raises(ValueError, match=message):
        mla_draft(**{field: value}).configure_target_aux_hidden_capture(target)
    target.set_dspark_aux_capture_materialized.assert_not_called()


def test_mla_requires_explicit_raw_capture_capability():
    with pytest.raises(ValueError, match="supporting raw-prefix-sum"):
        mla_draft().configure_target_aux_hidden_capture(SimpleNamespace())


@pytest.mark.parametrize(
    "direct,preserved,expected", [("target", "fallback", "target"), (None, "fallback", "fallback"), (None, None, None)]
)
def test_mla_recovers_target_rotation_after_quant_config_is_cleared(direct, preserved, expected):
    _, namespace = load_model(
        "kimi_k3_dspark.py",
        "AscendK3DSparkForCausalLM",
        {"get_draft_attn_causal"},
        functions=("_get_target_rotation_path",),
        get_rotation_path=lambda _: direct,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(_ascend_target_rotation_path=preserved))
        )
    )
    assert namespace["_get_target_rotation_path"](config) == expected


def test_mla_cache_cleanup_reaches_owning_attention_layer():
    model, _ = load_model("kimi_k3.py", "AscendKimiMLAAttention", {"kv_cache"})
    attention = model()
    original_cache = [torch.ones(2, 4)]
    attention._attention_layer = nn.Module()
    attention._attention_layer.kv_cache = original_cache
    assert attention.kv_cache is original_cache
    replacement = [torch.empty(0)]
    attention.kv_cache = replacement
    assert attention._attention_layer.kv_cache is replacement


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_main_dense_sp_keeps_one_tp_reduction(rank, sequence_parallel):
    x = torch.arange(12, dtype=torch.float32).view(4, 3)
    weight_shards = [torch.eye(3), 2 * torch.eye(3)]
    calls = []

    class Upstream(nn.Module):
        def __init__(self, *, reduce_results, **kwargs):
            super().__init__()
            self.reduce_results = reduce_results

        def forward(self, value):
            calls.append("mlp")
            return value @ (sum(weight_shards) if self.reduce_results else weight_shards[rank])

    def gather(local):
        calls.append("gather")
        torch.testing.assert_close(local, x.chunk(2)[rank])
        return x

    def reduce_scatter(partial):
        calls.append("reduce_scatter")
        torch.testing.assert_close(partial, x @ weight_shards[rank])
        return (partial + x @ weight_shards[1 - rank]).chunk(2)[rank]

    model, _ = load_model(
        "kimi_k3.py",
        "AscendKimiMLP",
        {"__init__", "forward"},
        parent=Upstream,
        sp_all_gather=gather,
        sp_reduce_scatter=reduce_scatter,
    )
    mlp = model(3, 6, "silu", use_sequence_parallel=sequence_parallel)
    given = x.chunk(2)[rank] if sequence_parallel else x
    torch.testing.assert_close(mlp(given), given @ sum(weight_shards))
    assert mlp.reduce_results is (not sequence_parallel)
    assert calls == (["gather", "mlp", "reduce_scatter"] if sequence_parallel else ["mlp"])


def test_gqa_captures_requested_final_materialized_state():
    def materialize(hidden_states, residual, projection, norm, num_blocks):
        return hidden_states + 100 * num_blocks

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.prev_valid_blocks = index
            self.self_attention_res_proj = None
            self.self_attention_res_norm = None

        def forward(self, *, positions, hidden_states, residual):
            return materialize(hidden_states, residual, None, None, self.prev_valid_blocks) + 10, residual

    model_class, _ = load_model(
        "kimi_k3.py",
        "AscendKimiLinearModel",
        {"forward"},
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
        _apply_ascend_attn_res=materialize,
    )
    model = model_class()
    model.config = SimpleNamespace(attn_res_block_size=1)
    model.start_layer, model.end_layer = 0, 2
    model.layers = nn.ModuleList([Layer(0), Layer(1)])
    model.use_sequence_parallel = False
    model.dspark_aux_capture_materialized = True
    model.aux_hidden_state_layers = (1, 2)
    model.output_attn_res_proj = None
    model.output_attn_res_norm = None

    output, auxiliary = model(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0]]),
    )
    assert len(auxiliary) == 2
    torch.testing.assert_close(auxiliary[0], torch.tensor([[111.0]]))
    torch.testing.assert_close(auxiliary[1], torch.tensor([[321.0]]))
    torch.testing.assert_close(auxiliary[1], output)
