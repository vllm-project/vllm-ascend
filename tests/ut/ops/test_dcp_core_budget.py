# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for budget forwarding and launch limits (kernels are emulated)."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/dcp/dcp_a2a.py"


def load(name, scope):
    node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("budget,expected", [(None, 56), (28, 28), (16, 16), (64, 56), (1, 1)])
@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_pack_launch_respects_budget_without_losing_rows(budget, expected, tokens):
    batched, scalar = MagicMock(), MagicMock()
    output = torch.empty(96, tokens, 512, dtype=torch.bfloat16).transpose(0, 1)
    lse = torch.empty(96, tokens, dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
    scope = dict(
        torch=torch,
        triton=SimpleNamespace(next_power_of_2=lambda n: n),
        is_950=lambda: True,
        init_device_properties_triton=lambda: None,
        get_vectorcore_num=lambda: 56,
        _lse_pack_dim=lambda dtype: 4,
        _validate_dcp_inputs=lambda *args: (tokens, 96, 512, 12, tokens),
        _pack_dcp_output_lse_batched_kernel=batched,
        _pack_dcp_output_lse_kernel=scalar,
    )
    send = load("pack_dcp_output_lse", scope)(output, lse, 8, 1, max_vector_cores=budget)
    assert send.shape == (8, 12, tokens, 516)
    kernel = batched if tokens * 96 >= 8 * expected else scalar
    kernel.__getitem__.assert_called_once_with((expected,))
    # Every row remains part of the grid-stride loop, regardless of grid size.
    assert kernel.__getitem__.return_value.call_args.args[-1] == tokens * 96


@pytest.mark.parametrize("cap", [None, 28])
@pytest.mark.parametrize("scatter_size", [1, 8])
def test_exchange_forwards_budget_but_preserves_collective(cap, scatter_size):
    send = torch.empty(scatter_size, 12, 8, 516)
    pack = Mock(return_value=send)
    collective = Mock()
    scope = dict(
        torch=torch,
        dist=SimpleNamespace(ProcessGroup=object, all_to_all_single=collective),
        GroupCoordinator=object,
        pack_dcp_output_lse=pack,
    )
    fn = load("dcp_a2a_fused_combine", scope)
    recv = fn(
        torch.empty(8, 96, 512),
        torch.empty(8, 96, 1),
        scatter_size,
        1,
        object(),
        defer_combine=True,
        pack_vector_cores=cap,
    )
    assert recv.shape == send.shape
    assert pack.call_args.kwargs["max_vector_cores"] == cap
    assert collective.call_count == int(scatter_size > 1)


def test_fake_schema_accepts_budget_and_keeps_layout():
    scope = dict(torch=torch, _lse_pack_dim=lambda dtype: 4)
    fn = load("dcp_a2a_fused_fake", scope)
    output = torch.empty(8, 96, 512, device="meta", dtype=torch.bfloat16)
    lse = torch.empty(8, 96, 1, device="meta")
    baseline = fn(output, lse, 8, 1, "dcp", defer_combine=True)
    budgeted = fn(output, lse, 8, 1, "dcp", defer_combine=True, pack_vector_cores=28)
    assert baseline.shape == budgeted.shape == (8, 12, 8, 516)
    assert baseline.stride() == budgeted.stride()
