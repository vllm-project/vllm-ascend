# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_patch,
)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([10.0, 20.0, 30.0, 40.0])
        )


def test_apply_sparse_patch_matches_upstream_signature():
    assert "verify" not in inspect.signature(apply_sparse_patch).parameters


def test_apply_sparse_patch_updates_only_selected_values():
    model = DummyModel()
    apply_sparse_patch(
        model,
        SparseWeightPatch(
            "weight",
            torch.tensor([1, 3], dtype=torch.int32),
            torch.tensor([200.0, 400.0]),
        ),
        expected_shape=[4],
    )
    assert torch.equal(
        model.weight,
        torch.tensor([10.0, 200.0, 30.0, 400.0]),
    )


@pytest.mark.parametrize("indices", [[-1], [4]])
def test_apply_sparse_patch_rejects_out_of_bounds_indices(indices):
    with pytest.raises((IndexError, RuntimeError)):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch(
                "weight",
                torch.tensor(indices, dtype=torch.int32),
                torch.tensor([1.0]),
            ),
        )


def test_apply_sparse_patch_rejects_declared_shape_mismatch():
    with pytest.raises(ValueError, match="declared shape"):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch(
                "weight",
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1.0]),
            ),
            expected_shape=[2, 2],
        )


def test_apply_sparse_patch_empty_update_is_noop():
    model = DummyModel()
    before = model.weight.detach().clone()
    apply_sparse_patch(
        model,
        SparseWeightPatch(
            "weight",
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float32),
        ),
    )
    assert torch.equal(model.weight, before)


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1.0]),
            "require int32 indices",
        ),
        (
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1.0]),
            "must be 1D flattened updates",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([[1.0]]),
            "must be 1D flattened updates",
        ),
        (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1.0]),
            "matching lengths",
        ),
        (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            "does not match parameter dtype",
        ),
    ],
)
def test_apply_sparse_patch_rejects_invalid_patch(indices, values, message):
    with pytest.raises(ValueError, match=message):
        apply_sparse_patch(
            DummyModel(),
            SparseWeightPatch("weight", indices, values),
        )


def test_apply_sparse_patch_rejects_noncontiguous_parameter():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    )
    assert not model.weight.is_contiguous()

    with pytest.raises(NotImplementedError, match="require contiguous params"):
        apply_sparse_patch(
            model,
            SparseWeightPatch(
                "weight",
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1.0]),
            ),
        )
