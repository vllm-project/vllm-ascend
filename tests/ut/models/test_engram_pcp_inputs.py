# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production Engram orchestration with hash/lookup device kernels mocked.

AST isolates the model methods from NPU/vLLM module initialization. The tests
execute the original method bodies; they do not validate hash kernels, table
offload, distributed collectives, or the full model on NPU.
"""

import ast
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _load_model_methods():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    names = {"prepare_engram", "prepare_engram_inputs", "prepare_engram_graph_inputs", "forward"}
    methods = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in names]
    module = ModuleType("v41_pcp_model_methods")
    module.__dict__.update(torch=torch)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


model_mod = _load_model_methods()


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(model_mod, "engram_enabled", lambda config: True, raising=False)
    monkeypatch.setattr(
        model_mod, "engram_dead_mask", lambda ids, image, pad: (ids == image) | (ids == pad), raising=False
    )
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        model_mod, "gather_engram_hashes", MagicMock(side_effect=lambda hashes, **kwargs: hashes), raising=False
    )
    table = SimpleNamespace(
        dim=1,
        n_hash_cols=6,
        embed_gathered=MagicMock(side_effect=lambda hashes, count: hashes[:count].to(torch.bfloat16).unsqueeze(-1)),
    )
    hash_state = MagicMock(return_value=torch.arange(24, dtype=torch.int32).reshape(4, 1, 6))
    hash_state.ensure_cache.return_value = True
    hash_state.lookback_depth = 3
    hash_state.dummy_hashes.side_effect = lambda ids: (
        torch.full((ids.numel(), 1, 6), -1, dtype=torch.int32),
        torch.zeros(ids.numel(), dtype=torch.bool),
    )
    obj = SimpleNamespace(
        config=SimpleNamespace(
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=1,
            engram_layer_ids=[0],
            image_token_id=99,
            image_pad_token_id=98,
        ),
        layers=[SimpleNamespace(engram=SimpleNamespace(embed_tokens=table))],
        engram_hash=hash_state,
        engram_dp_shared_memory=False,
        engram_rotation=torch.eye(32),
        _engram_max_tokens=8,
        _engram_input_buffers=None,
    )
    for name in ("prepare_engram", "prepare_engram_inputs", "prepare_engram_graph_inputs"):
        setattr(obj, name, MethodType(getattr(model_mod, name), obj))
    return obj


def _coordinates():
    return dict(
        query_start_loc=torch.tensor([0, 4]),
        lookback_token_ids=torch.tensor([[5, 99, -1]]),
        slot_mapping=torch.arange(4),
        block_table=torch.tensor([[0, 1]]),
    )


def test_hashes_global_then_selects_local_rows_before_lookup(model):
    inputs = torch.tensor([0, 1, 99, 3])
    output = model.prepare_engram_inputs(
        inputs, torch.arange(4), 3, local_token_indices=torch.tensor([0, 3]), **_coordinates()
    )
    assert model.engram_hash.call_args.args[0] is inputs
    assert model.engram_hash.call_args.args[3].tolist() == [False, False, True, False]
    assert model.engram_hash.call_args.args[5].tolist() == [[False, True, False]]
    assert model_mod.gather_engram_hashes.call_args.args[0].tolist() == [[list(range(6))], [list(range(18, 24))]]
    assert model.layers[0].engram.embed_tokens.embed_gathered.call_args.args[1] == 2
    assert output["engram_mask"][:3].tolist() == [True, True, False]
    assert torch.count_nonzero(output["engram_lookups"][0][2:]) == 0
    previous_pointer = output["engram_lookups"][0].data_ptr()
    output = model.prepare_engram_inputs(
        inputs, torch.arange(4), 3, local_token_indices=torch.tensor([2]), **_coordinates()
    )
    assert output["engram_lookups"][0].data_ptr() == previous_pointer
    assert not output["engram_mask"][:3].any()
    assert torch.count_nonzero(output["engram_lookups"][0][1:3]) == 0


@pytest.mark.parametrize("shared", [False, True])
def test_empty_local_rank_preserves_new_lookup_contract(model, shared):
    model.engram_dp_shared_memory = shared
    output = model.prepare_engram_inputs(
        torch.arange(4), torch.arange(4), 2, local_token_indices=torch.empty(0, dtype=torch.long), **_coordinates()
    )
    model.engram_hash.assert_called_once()
    assert model_mod.gather_engram_hashes.call_args.kwargs == {"dp_shared_memory": shared, "pre_forward": False}
    assert model_mod.gather_engram_hashes.call_args.args[0].shape == (0, 1, 6)
    assert model.layers[0].engram.embed_tokens.embed_gathered.call_args.args[1] == 0
    assert not output["engram_mask"].any()


@pytest.mark.parametrize("dp_size,shared,participates", [(1, False, False), (2, False, True), (2, True, False)])
@pytest.mark.parametrize("pre_forward", [False, True])
def test_dummy_retains_dp_and_shared_memory_behavior(model, monkeypatch, dp_size, shared, participates, pre_forward):
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: dp_size)
    model.engram_dp_shared_memory = shared
    result = model.prepare_engram_inputs(torch.arange(2), torch.arange(2), 2, pre_forward=pre_forward)
    model.engram_hash.assert_not_called()
    assert model.engram_hash.dummy_hashes.called == participates
    assert model_mod.gather_engram_hashes.called == participates
    if participates:
        assert model.engram_hash.dummy_hashes.call_args.args[0].numel() == (0 if pre_forward else 2)
        assert model_mod.gather_engram_hashes.call_args.kwargs["pre_forward"] == pre_forward
    assert not result["engram_mask"].any()


def test_pcp1_keeps_all_hash_rows_and_explicit_history(model):
    coordinates = _coordinates()
    model.prepare_engram_inputs(torch.arange(4), torch.arange(4), 4, **coordinates)
    assert model.engram_hash.call_args.args[4] is coordinates["lookback_token_ids"]
    assert model_mod.gather_engram_hashes.call_args.args[0].shape == (4, 1, 6)
    assert model.layers[0].engram.embed_tokens.embed_gathered.call_args.args[1] == 4


def test_capture_only_binds_buffers(model):
    result = model.prepare_engram_graph_inputs(4)
    assert result["engram_lookups"][0].shape == (8, 6)
    model.engram_hash.assert_not_called()
    model_mod.gather_engram_hashes.assert_not_called()


@pytest.mark.parametrize("rotated", [False, True])
def test_sequence_parallel_slices_step_extent_before_sharding(monkeypatch, rotated):
    monkeypatch.setattr(model_mod, "envs", SimpleNamespace(VLLM_MOE_SKIP_PADDING=False), raising=False)
    monkeypatch.setattr(model_mod, "sp_shard", lambda value: value.chunk(2, dim=0)[1], raising=False)
    monkeypatch.setattr(model_mod, "sp_all_gather", lambda value: torch.cat((value, value)), raising=False)
    engram = MagicMock(side_effect=lambda hidden, *args: hidden)

    class Layer:
        layer_idx = 0

        def __call__(self, positions, hidden, pre_mix, scaling, **kwargs):
            return hidden, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

    layer = Layer()
    layer.engram = engram
    model = SimpleNamespace(
        use_sequence_parallel=True,
        embed_input_ids=lambda ids: torch.zeros((ids.numel(), 2)),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        _mtp_hidden_buffer=None,
        hc_mult=2,
        needs_moe_input_ids=False,
        aux_hidden_state_layers=(),
        layers=[layer],
        config=SimpleNamespace(hidden_size=2, rms_norm_eps=1e-6),
        engram_rotation=torch.eye(2),
        engram_rotated=rotated,
        norm=lambda value: value,
    )
    lookups = torch.arange(48).reshape(8, 6).float()
    model_mod.forward(
        model,
        torch.arange(6),
        torch.arange(6),
        None,
        engram_lookups={0: lookups},
        engram_mask=torch.tensor([True] * 6 + [False] * 2),
    )
    torch.testing.assert_close(engram.call_args.args[1], lookups[3:6])
    assert engram.call_args.args[2].tolist() == [True, True, True]
    assert engram.call_args.args[3] is (model.engram_rotation if rotated else None)
