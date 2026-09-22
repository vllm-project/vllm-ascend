# SPDX-License-Identifier: Apache-2.0
"""CPU checks of launch metadata; numerical kernels are tested on NPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def load():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/flash_attention_output.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if getattr(node, "name", "") in ("flash_attention_gate", "flash_attention_output")
    ]
    gate, output = Kernel(), Kernel()
    scope = dict(
        torch=torch,
        triton=SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
        init_device_properties_triton=lambda: None,
        get_vectorcore_num=lambda: 40,
        _flash_attention_gate_kernel=gate,
        _flash_attention_output_kernel=output,
    )
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope), gate, output


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("tokens,hidden,block", [(4, 128, 1024), (32, 1536, 2048), (64, 1536, 2048)])
def test_gate_launch_keeps_strides_and_optional_mask(masked, tokens, hidden, block):
    ops, kernel, _ = load()
    projected = torch.empty(tokens, hidden + 16, dtype=torch.bfloat16)[:, :hidden]
    gate = torch.empty(tokens, hidden + 8, dtype=torch.bfloat16)[:, :hidden]
    mask = torch.ones(tokens, dtype=torch.int32) if masked else None
    assert ops.flash_attention_gate(projected, gate, mask) is projected
    grid, args, kwargs = kernel.calls[0]
    assert args[2] is (mask if masked else projected)
    assert args[3:] == (tokens, hidden + 16, hidden + 8)
    assert kwargs["HAS_MASK"] == masked
    assert kwargs["BLOCK"] == block
    assert grid == (min(tokens * ((hidden + block - 1) // block), 40),)


def test_empty_gate_does_not_launch():
    ops, kernel, _ = load()
    value = torch.empty(0, 128)
    assert ops.flash_attention_gate(value, value) is value
    assert not kernel.calls


def test_output_limits_live_rows_and_preserves_output_stride():
    ops, _, kernel = load()
    result = torch.empty(3, 128)
    mask = torch.ones(2, dtype=torch.int32)
    output = torch.empty(4, 136)[:, :128]
    assert ops.flash_attention_output(result, mask, output) is output
    _, args, kwargs = kernel.calls[0]
    assert args[3:] == (2, 4, 128, 136)
    assert kwargs["HIDDEN"] == 128


def test_empty_mask_zeros_padded_shard_without_launch():
    ops, _, kernel = load()
    output = torch.full((4, 128), torch.nan)
    assert ops.flash_attention_output(torch.empty(0, 128), torch.empty(0), output) is output
    assert torch.count_nonzero(output) == 0
    assert not kernel.calls
