# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for HiSparse prefill representative selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm_ascend.attention import sfa_hisparse
from vllm_ascend.ops.triton import hisparse


def test_prefill_dispatches_npu_amin_without_cpu_fallback(monkeypatch):
    representatives = SimpleNamespace(device=SimpleNamespace(type="npu"))
    indices = torch.tensor([0, 0, 1])
    values = torch.tensor([7, 3, 5])
    scatter = Mock(return_value=representatives)
    monkeypatch.setattr(sfa_hisparse, "scatter_min_representatives", scatter)

    output = sfa_hisparse._PrefillSourceSelection().__torch_function__(
        torch.Tensor.scatter_reduce_,
        (),
        (representatives, 0, indices, values),
        {"reduce": "amin", "include_self": True},
    )

    assert output is representatives
    scatter.assert_called_once_with(representatives, indices, values)


def test_prefill_leaves_other_scatter_reductions_in_torch():
    representatives = torch.tensor([11, 13])
    with sfa_hisparse._PrefillSourceSelection():
        representatives.scatter_reduce_(0, torch.tensor([0, 0]), torch.tensor([17, 3]), reduce="amax")
    assert representatives.tolist() == [17, 13]


def test_empty_representatives_scatter_keeps_output_without_launch(monkeypatch):
    representatives = torch.tensor([11, 13])
    empty = torch.empty(0, dtype=torch.int64)
    kernel = Mock()
    monkeypatch.setattr(hisparse, "_SCATTER_MIN_REPRESENTATIVES", kernel)

    assert hisparse.scatter_min_representatives(representatives, empty, empty) is representatives
    assert representatives.tolist() == [11, 13]
    kernel.assert_not_called()
