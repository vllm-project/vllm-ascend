# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest
import torch


def _load_validation():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/mla_read_slots.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_slot_metadata"]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_validate_slot_metadata"]


@pytest.mark.parametrize(
    "lengths,offsets,total",
    [
        ([1], [0], 1),
        ([3], [-1], 3),
        ([3, 2], [1, 4], 5),
        ([0, 1], [999999, 0], 1),
        ([1, 0], [0, -999999], 1),
        ([1], [0, 999999], 1),
    ],
)
def test_valid_metadata_accepts_empty_requests_and_negative_indices(lengths, offsets, total):
    table = torch.arange(len(lengths) * 2).reshape(len(lengths), 2)
    _load_validation()(table, torch.tensor(lengths), torch.tensor(offsets), total, 4)


@pytest.mark.parametrize(
    "lengths,offsets,total",
    [
        ([1], [8], 1),
        ([1], [-9], 1),
        ([3], [7], 3),
        ([-1, 2], [0, 0], 1),
        ([1, 1], [0, 0], 3),
    ],
)
def test_invalid_metadata_is_not_silently_accepted(lengths, offsets, total):
    table = torch.arange(len(lengths) * 2).reshape(len(lengths), 2)
    with pytest.raises((IndexError, RuntimeError)):
        _load_validation()(table, torch.tensor(lengths), torch.tensor(offsets), total, 4)


def test_rejects_wrapped_end_position_even_if_both_wrapped_endpoints_look_valid():
    table = torch.arange(4).reshape(1, 4)
    lengths = torch.tensor([2], dtype=torch.int64)
    offsets = torch.tensor([2**63 - 1], dtype=torch.int64)
    with pytest.raises((IndexError, RuntimeError)):
        _load_validation()(table, lengths, offsets, 2, 2**61)


def test_rejects_lengths_whose_int64_sum_wraps_to_output_size():
    table = torch.arange(10).reshape(5, 2)
    lengths = torch.tensor([2**62, 2**62, 2**62, 2**62, 1], dtype=torch.int64)
    with pytest.raises((IndexError, RuntimeError)):
        _load_validation()(table, lengths, None, 1, 2**61)
