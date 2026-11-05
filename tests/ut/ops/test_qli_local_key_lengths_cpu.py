# SPDX-License-Identifier: Apache-2.0
"""Execute the Triton arithmetic on CPU; compare with explicit token ownership."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Pointer:
    def __init__(self, tensor):
        self.tensor = tensor

    def __add__(self, offsets):
        return self.tensor, offsets


@pytest.mark.parametrize("interleave", [1, 8, 768])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_request_lengths_match_explicit_owner_counts(interleave, rank):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sharded_indexer.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_local_key_lengths"
    )
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    block = [0]

    def load(pointer, mask, other):
        tensor, indices = pointer
        return torch.where(mask, tensor[indices.clamp(max=tensor.numel() - 1)], other)

    def store(pointer, values, mask):
        tensor, indices = pointer
        tensor[indices[mask]] = values[mask]

    tl = SimpleNamespace(
        program_id=lambda _: block[0],
        arange=torch.arange,
        int32=torch.int32,
        load=load,
        store=store,
        minimum=lambda x, limit: x.clamp_max(limit),
        maximum=lambda x, limit: x.clamp_min(limit),
    )
    scope = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), scope)
    cycle = 8 * interleave
    lengths = torch.tensor(([0, 1, interleave - 1, interleave, cycle - 1, cycle, cycle + 1, 2 * cycle + 13] * 17)[:129])
    out = torch.full((129,), -1, dtype=torch.int32)
    for block[0] in range(2):
        scope["_local_key_lengths"](Pointer(lengths), Pointer(out), 129, 8, rank, interleave, 128)
    expected = [sum((token // interleave) % 8 == rank for token in range(int(length))) for length in lengths]
    torch.testing.assert_close(out, torch.tensor(expected, dtype=torch.int32))
