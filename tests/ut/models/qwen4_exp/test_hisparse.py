# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp"


@pytest.fixture
def ops():
    scope = {"torch": torch}
    for filename in ("gbsa.py", "hisparse.py"):
        source = MODEL_DIR / filename
        module = ast.parse(source.read_text())
        module.body = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
        exec(compile(module, str(source), "exec"), scope)
    return scope


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_resolver_keeps_partial_group_after_invalid_lanes(ops, length_dtype):
    compact = torch.full((4, 515), -1, dtype=torch.int32)
    compact[:, :512] = torch.arange(512)
    compact[1, 0] = -1
    visible = torch.tensor([2049, 2051, 2048, 0], dtype=length_dtype)
    logical = torch.empty((4, 2048), dtype=torch.int32)
    ops["prepare_resolver_indices"](compact, visible, logical)
    assert logical[0, :2044].tolist() == list(range(2044))
    assert logical[0, 2044:].tolist() == [2048, -1, -1, -1]
    assert logical[1].tolist() == [*range(4, 2051), -1]
    assert logical[2].tolist() == list(range(2048))
    assert (logical[3] == -1).all()


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_physical_attention_reads_selected_rows_per_kv_head(ops, kv_heads):
    rng = torch.Generator().manual_seed(17)
    heads, dim = 4, 8
    query = torch.randn(2, heads, dim, generator=rng)
    packed = torch.randn(3, 4, kv_heads * 2 * dim, generator=rng)
    packed[0, 0] = torch.nan
    physical = torch.tensor([[9, -1, 2, 6], [-1, -1, -1, -1]], dtype=torch.int32)

    def reference_gbsa(q, key, value, compact, table, requests, positions, lengths, output):
        assert table.tolist() == [[0], [1]]
        assert positions.tolist() == [2, -1]
        assert lengths.tolist() == [3, 0]
        assert requests.tolist() == [0, 1]
        assert compact[:, 0].tolist() == [0, 0]
        for head in range(heads):
            kv_head = head * kv_heads // heads
            scores = (key[0, :3, kv_head] @ q[0, head]) * dim**-0.5
            output[0, head] = scores.softmax(0) @ value[0, :3, kv_head]
        output[1].fill_(torch.nan)

    ops["qsa_gbsa_attention"] = reference_gbsa
    actual = ops["attend_physical"](query, packed, physical, kv_heads, torch.empty_like(query))
    expected = torch.zeros_like(query)
    rows = packed.reshape(-1, kv_heads, 2 * dim)[[9, 2, 6]]
    for head in range(heads):
        key, value = rows[:, head * kv_heads // heads].split(dim, dim=-1)
        expected[0, head] = ((key @ query[0, head]) * dim**-0.5).softmax(0) @ value
    torch.testing.assert_close(actual, expected)
