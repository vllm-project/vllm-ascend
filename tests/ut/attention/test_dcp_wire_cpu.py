# SPDX-License-Identifier: Apache-2.0
"""Check raw-wire alias geometry without launching device kernels."""

import ast
from pathlib import Path

import pytest
import torch


def load(name, filename="sfa_dcp_exchange.py", **extra):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", "") == name]
    scope = {"torch": torch, **extra}
    exec(compile(tree, str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("tokens", [4, 8, 16, 32, 64])
def test_direct_wire_view_preserves_payload_and_lse_bits(tokens):
    storage = torch.arange(96 * tokens * 272, dtype=torch.int32).reshape(96, tokens, 272)
    output = storage.view(torch.bfloat16)[..., :512].transpose(0, 1)
    actual = load("_view_direct_dcp_wire")(output)
    assert actual.data_ptr() == storage.data_ptr()
    torch.testing.assert_close(actual, storage.reshape(8, 12, tokens, 272), atol=0, rtol=0)


def test_contiguous_output_is_not_a_direct_wire():
    assert load("_view_direct_dcp_wire")(torch.empty(4, 96, 512, dtype=torch.bfloat16)) is None


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fake_preserves_existing_fp32_lse_contract(scatter_dim, return_lse):
    fake = load("sfa_dcp_a2a_fused_fake", "sfa_cp.py", can_use_raw_dcp_exchange=lambda *a, **k: False)
    result = fake(torch.empty(16, 96, 512), torch.empty(16, 96, 1), 8, scatter_dim, "unused", return_lse=return_lse)
    assert result.shape == ((2, 96, 512 + return_lse) if scatter_dim == 0 else (16, 12, 512 + return_lse))


@pytest.mark.parametrize("row_words", [257, 272])
def test_fake_deferred_raw_wire(row_words):
    fake = load("sfa_dcp_a2a_fused_fake", "sfa_cp.py", can_use_raw_dcp_exchange=lambda *a, **k: True)
    result = fake(
        torch.empty(16, 96, 512, dtype=torch.bfloat16),
        torch.empty(16, 96, 1),
        8,
        1,
        "unused",
        defer_combine=True,
        raw_row_words=row_words,
    )
    assert result.shape == (8, 12, 16, row_words)
    assert result.dtype == torch.int32
