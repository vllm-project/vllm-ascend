# SPDX-License-Identifier: Apache-2.0
"""CPU checks for QSA preparation without importing the NPU model stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def _load_definitions(path, names, namespace):
    """Execute production definitions while mocking only device boundaries."""
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in selected:
        if isinstance(node, ast.ClassDef):
            node.bases = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


@pytest.fixture
def indexer_class():
    namespace = _load_definitions(
        "vllm_ascend/models/qwen4_exp/qsa.py",
        {"AscendQSAIndexer"},
        {"torch": torch, "logger": Mock()},
    )
    return namespace["AscendQSAIndexer"]


def _make_indexer(indexer_class, skip_topk):
    indexer = indexer_class()
    indexer.skip_topk = skip_topk
    indexer.index_n_heads = 2
    indexer.index_kv_heads = 1
    indexer.index_head_dim = 128
    projected = torch.arange(2 * 384, dtype=torch.float32).reshape(2, 384)
    indexer.index_qk_proj = Mock(return_value=(projected, None))
    indexer.q_layernorm = object()
    indexer._fused_norm_rope_eligible = Mock(return_value=True)
    indexer._fused_norm_rope = Mock(side_effect=lambda q, norm, positions: q + 1000)
    indexer._metadata = Mock(return_value=(SimpleNamespace(num_actual_tokens=2), object()))
    indexer._update_and_compress = Mock()
    indexer._select = Mock(return_value=torch.full((2, 3), 7))
    return indexer, projected


@pytest.mark.parametrize("skip_topk", [False, True])
def test_forward_keeps_key_updates_and_only_skips_unused_query(indexer_class, skip_topk):
    indexer, projected = _make_indexer(indexer_class, skip_topk)
    hidden = torch.zeros((4, 16))
    positions = torch.arange(4)
    out = torch.full((2, 3), 11)
    actual = indexer.forward(hidden, positions, out)

    projected_hidden = indexer.index_qk_proj.call_args.args[0]
    torch.testing.assert_close(projected_hidden, hidden[:2])
    assert projected_hidden.shape == (2, 16)
    indexer._update_and_compress.assert_called_once()
    key, used_positions, raw, compressed = indexer._update_and_compress.call_args.args
    torch.testing.assert_close(key, projected[:, 256:].reshape(2, 1, 128))
    torch.testing.assert_close(used_positions, positions[:2])
    assert (raw, compressed) == indexer._metadata.return_value
    if skip_topk:
        assert actual is out
        torch.testing.assert_close(out, torch.full_like(out, 11))
        indexer._fused_norm_rope.assert_not_called()
        indexer._select.assert_not_called()
    else:
        indexer._fused_norm_rope.assert_called_once()
        query = indexer._select.call_args.args[0]
        torch.testing.assert_close(query, projected[:, :256].reshape(2, 2, 128) + 1000)
        assert actual is indexer._select.return_value


def test_public_projection_still_returns_normalized_query_when_topk_is_reused(indexer_class):
    indexer, projected = _make_indexer(indexer_class, True)
    query, key = indexer.project_qk(torch.zeros((2, 16)), torch.arange(2))
    indexer._fused_norm_rope.assert_called_once()
    torch.testing.assert_close(query, projected[:, :256].reshape(2, 2, 128) + 1000)
    torch.testing.assert_close(key, projected[:, 256:].reshape(2, 1, 128))


def test_topk_reuse_requires_output_before_updating_state(indexer_class):
    indexer, _ = _make_indexer(indexer_class, True)
    with pytest.raises(RuntimeError, match="requires an output buffer"):
        indexer.forward(torch.zeros((2, 16)), torch.arange(2))
    indexer.index_qk_proj.assert_not_called()
    indexer._update_and_compress.assert_not_called()


def test_missing_metadata_preserves_reused_selection(indexer_class):
    indexer, _ = _make_indexer(indexer_class, True)
    indexer._metadata.return_value = None
    out = torch.full((2, 3), 11)
    assert indexer.forward(torch.zeros((2, 16)), torch.arange(2), out) is out
    indexer.index_qk_proj.assert_not_called()
    indexer._update_and_compress.assert_not_called()


def test_indexer_dispatches_persistent_cache_and_strided_positions(indexer_class, monkeypatch):
    indexer = indexer_class()
    cache = MagicMock()
    cache.__getitem__.side_effect = AssertionError("must not gather a temporary rotary tensor")
    indexer.rotary_emb = SimpleNamespace(
        _match_cos_sin_cache_dtype=Mock(return_value=cache),
        mrope_section=[11, 11, 10],
        rotary_dim=64,
    )
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    tensor = torch.zeros((2, 1, 128), dtype=torch.bfloat16)
    positions = torch.arange(6).reshape(2, 3).transpose(0, 1)
    op = Mock(return_value=tensor)
    monkeypatch.setattr(torch.ops.vllm, "triton_qsa_rmsnorm_mrope_from_cache", op, raising=False)
    assert indexer._fused_norm_rope(tensor, norm, positions) is tensor
    assert op.call_args.kwargs["cos_sin_cache"] is cache
    assert op.call_args.kwargs["positions"] is positions
    cache.__getitem__.assert_not_called()


@pytest.fixture
def cache_op():
    kernel = MagicMock()
    namespace = _load_definitions(
        "vllm_ascend/ops/triton/linearnorm/qsa_rmsnorm_mrope.py",
        {"triton_qsa_rmsnorm_mrope_from_cache"},
        {
            "torch": torch,
            "triton": SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
            "_qsa_rmsnorm_mrope_kernel": kernel,
        },
    )
    return namespace["triton_qsa_rmsnorm_mrope_from_cache"], kernel


def _cache_op_inputs(tokens=2):
    return dict(
        tensor=torch.empty((tokens, 2, 128), dtype=torch.bfloat16),
        weight=torch.ones(128, dtype=torch.bfloat16),
        cos_sin_cache=torch.empty((32, 64), dtype=torch.bfloat16),
        positions=torch.arange(tokens * 3).reshape(tokens, 3).transpose(0, 1),
        eps=1e-6,
        mrope_section=[11, 11, 10],
        rope_dim=64,
    )


@pytest.mark.parametrize("axis_count", [1, 3])
def test_cache_op_passes_position_strides_without_materialization(cache_op, axis_count):
    op, kernel = cache_op
    inputs = _cache_op_inputs()
    if axis_count == 1:
        inputs["positions"] = inputs["positions"][0]
    result = op(**inputs)
    args = kernel.__getitem__.return_value.call_args.args
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert result.shape == inputs["tensor"].shape
    assert args[2] is inputs["cos_sin_cache"]
    assert args[4] is inputs["positions"]
    assert args[11] == (inputs["positions"].stride(0) if axis_count == 3 else 0)
    assert args[12] == inputs["positions"].stride(-1)
    assert kwargs["gather_positions"] is True
    assert kwargs["has_mrope"] is (axis_count == 3)


def test_empty_token_input_does_not_launch(cache_op):
    op, kernel = cache_op
    assert op(**_cache_op_inputs(tokens=0)).shape == (0, 2, 128)
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("positions", torch.zeros((2, 2), dtype=torch.int64), "three position axes"),
        ("positions", torch.zeros((3, 3), dtype=torch.int64), "rows must match"),
        ("positions", torch.zeros((3, 2), dtype=torch.float32), "integer tensor"),
        ("cos_sin_cache", torch.empty((0, 64), dtype=torch.bfloat16), "nonempty matching"),
        ("rope_dim", 65, "even rotary width"),
        ("rope_dim", 96, "power-of-two rotary width"),
        ("weight", torch.empty(256, dtype=torch.bfloat16)[::2], "contiguous matching weight"),
    ],
)
def test_cache_op_rejects_invalid_inputs_before_launch(cache_op, field, value, error):
    op, kernel = cache_op
    inputs = _cache_op_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match=error):
        op(**inputs)
    kernel.__getitem__.assert_not_called()
