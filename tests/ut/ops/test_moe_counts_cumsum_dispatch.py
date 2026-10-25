# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import moe_utils


def npu_counts():
    counts = MagicMock(spec=torch.Tensor)
    counts.device = SimpleNamespace(type="npu")
    counts.dtype = torch.int64
    counts.ndim = 1
    counts.is_contiguous.return_value = True
    counts.numel.return_value = 32
    return counts


@pytest.mark.parametrize("n", [1, 32, 1024, 4096])
def test_int64_counts_dispatches_to_custom_scan(n):
    counts = npu_counts()
    counts.numel.return_value = n
    with (
        patch.object(moe_utils, "enable_custom_op", return_value=True),
        patch.object(torch.ops._C_ascend, "moe_counts_cumsum", create=True) as scan,
    ):
        result = moe_utils.cumsum_group_list(counts, 1, 0)
        scan.assert_called_once_with(counts)
        assert result is scan.return_value
        counts.cumsum.assert_not_called()


@pytest.mark.parametrize("case", ["disabled", "empty", "too_large", "strided", "int32", "matrix"])
def test_unsupported_counts_keep_existing_conversion(case):
    counts = npu_counts()
    if case == "empty":
        counts.numel.return_value = 0
    elif case == "too_large":
        counts.numel.return_value = moe_utils._MOE_COUNTS_CUMSUM_MAX_EXPERTS + 1
    elif case == "strided":
        counts.is_contiguous.return_value = False
    elif case == "int32":
        counts.dtype = torch.int32
    elif case == "matrix":
        counts.ndim = 2
    with (
        patch.object(moe_utils, "enable_custom_op", return_value=case != "disabled"),
        patch.object(torch.ops._C_ascend, "moe_counts_cumsum", create=True) as scan,
    ):
        result = moe_utils.cumsum_group_list(counts, 1, 0)
        scan.assert_not_called()
        counts.cumsum.assert_called_once_with(dim=0)
        assert result is counts.cumsum.return_value


def test_missing_custom_scan_keeps_existing_conversion():
    counts = npu_counts()
    with (
        patch.object(moe_utils, "enable_custom_op", return_value=True),
        patch.object(torch.ops, "_C_ascend", SimpleNamespace()),
    ):
        result = moe_utils.cumsum_group_list(counts, 1, 0)
        counts.cumsum.assert_called_once_with(dim=0)
        assert result is counts.cumsum.return_value


def test_cpu_int32_conversion_preserves_int64_output():
    counts = torch.tensor([2, 3, 1], dtype=torch.int32)
    result = moe_utils.cumsum_group_list(counts, 1, 0)
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.tensor([2, 5, 6]))


def test_prefix_to_prefix_returns_original_tensor():
    counts = npu_counts()
    with patch.object(moe_utils, "enable_custom_op") as enabled:
        assert moe_utils.cumsum_group_list(counts, 0, 0) is counts
        enabled.assert_not_called()
