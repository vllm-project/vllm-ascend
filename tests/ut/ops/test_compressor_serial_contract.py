# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
import torch

from tests.ut.ops.compressor_serial_reference import compressor_reference


def test_exported_shapes_are_preserved():
    data = json.loads(Path(__file__).with_name("compressor_serial_shapes.json").read_text())
    assert len(data["cases"]) == 6
    for phase in ("prefill", "decode"):
        assert sum(c["instances"] for c in data["cases"] if c["phase"] == phase) == 62
    for case in data["cases"]:
        shapes = case["inputs"]
        d = shapes[5][0]
        assert shapes[1] == shapes[2] == [case["coff"] * d, 4096]
        assert case["outputs"][0] == [shapes[6][0], d]
        assert case["outputs"][1] == shapes[3]


@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("rotary", [1, 2])
def test_constant_groups_and_full_cache_write(coff, rotary):
    # Constant values and zero scores give an analytic softmax/RMSNorm answer.
    dim, ratio = 128, 4
    x = torch.ones(8, 1)
    wk = torch.ones(coff * dim, 1)
    wg = torch.zeros_like(wk)
    state = torch.full((2, 32, 2 * coff * dim), -7.0)
    args = (
        x,
        wk,
        wg,
        state,
        torch.zeros(ratio, coff * dim),
        torch.ones(dim),
        torch.zeros(3, 64),
        torch.ones(3, 64),
        torch.tensor([[1]]),
        torch.tensor([0, 8]),
        None,
        torch.tensor([0]),
    )
    out, updated = compressor_reference(args, ratio=ratio, coff=coff, rotary=rotary)
    torch.testing.assert_close(out[:2], torch.full((2, dim), (1 + 1e-6) ** -0.5))
    assert out[2].count_nonzero() == 0
    assert torch.equal(updated[0], state[0])
    assert torch.equal(updated[1, :8, : coff * dim], torch.ones(8, coff * dim))
    assert updated[1, :8, coff * dim :].count_nonzero() == 0
    assert torch.equal(updated[1, 8:], state[1, 8:])


def test_huge_scores_remain_finite():
    dim = 128
    args = (
        torch.tensor([[1.0], [2.0]]),
        torch.ones(dim, 1),
        torch.full((dim, 1), 1e4),
        torch.zeros(1, 8, 2 * dim),
        torch.zeros(2, dim),
        torch.ones(dim),
        torch.zeros(1, 64),
        torch.ones(1, 64),
        None,
        torch.tensor([0, 2]),
        None,
        None,
    )
    out, _ = compressor_reference(args, ratio=2, coff=1)
    assert out.isfinite().all()
    torch.testing.assert_close(out, torch.full_like(out, 2 / (4 + 1e-6) ** 0.5))
