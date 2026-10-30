# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts for sparse-index capacity and the known anchor query width."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices
from vllm_ascend.ops.triton import dspark_swa_indices as fused


def inputs(query_width=9):
    return (
        torch.tensor([[10, 21, 32, 43, 54]], dtype=torch.int32),
        8,
        120,
        32,
        torch.tensor([0, query_width], dtype=torch.int32),
        torch.tensor([128 + query_width], dtype=torch.int32),
    )


def test_explicit_width_rejects_missing_last_nonanchor_slot():
    with pytest.raises(ValueError, match="index_width"):
        build_dspark_swa_indices(*inputs(), num_decode_tokens=9, index_width=128)


@pytest.mark.parametrize("query_width", [8, 9])
def test_explicit_width_preserves_known_anchor_and_nonanchor(query_width):
    table, *args = inputs(query_width)
    width = 128 if query_width == 8 else 129
    slots, lengths = build_dspark_swa_indices(
        table, *args, num_decode_tokens=query_width, index_width=width, num_query_per_req=query_width
    )
    expected = [int(table[0, position // 32]) * 32 + position % 32 for position in range(8, 128 + query_width)]
    assert slots.shape == (query_width, 1, width)
    assert slots[:, 0].tolist() == [expected] * query_width
    assert lengths.tolist() == [len(expected)] * query_width


@pytest.mark.parametrize("query_width", [8, 9])
def test_default_width_keeps_every_visible_slot(query_width):
    slots, lengths = build_dspark_swa_indices(*inputs(query_width), num_decode_tokens=query_width)
    assert slots.shape[-1] == 256
    assert (slots >= 0).sum(-1).flatten().tolist() == lengths.tolist() == [120 + query_width] * query_width


class CaptureKernel:
    def __init__(self, *, forbidden=False):
        self.calls = []
        self.forbidden = forbidden

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert not self.forbidden, "Invalid index width reached the native launch boundary"
            self.calls.append((grid, args, kwargs))

        return launch


def test_fused_explicit_width_rejects_before_native_launch():
    kernel = CaptureKernel(forbidden=True)
    with (
        patch.object(fused, "_dspark_swa_indices_kernel", kernel, create=True),
        pytest.raises(ValueError, match="index_width"),
    ):
        fused.build_dspark_swa_slot_indices_triton(*inputs(), num_decode_tokens=9, index_width=128)


@pytest.mark.parametrize("query_width", [8, 9])
def test_fused_known_query_width_uses_host_capacity_without_tensor_read(query_width):
    kernel = CaptureKernel()
    width = 120 + query_width
    with (
        patch.object(fused, "_dspark_swa_indices_kernel", kernel, create=True),
        patch.object(fused, "init_device_properties_triton"),
        patch.object(fused, "get_vectorcore_num", return_value=1),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("Unexpected device value read")),
    ):
        slots, lengths = fused.build_dspark_swa_slot_indices_triton(
            *inputs(query_width), num_decode_tokens=query_width, index_width=width, num_query_per_req=query_width
        )
    assert slots.shape == (query_width, 1, width)
    assert lengths.shape == (query_width,)
    assert len(kernel.calls) == 1
    assert kernel.calls[0][2]["INDEX_W"] == width
    assert kernel.calls[0][1][10] == query_width


@pytest.mark.parametrize("query_width", [0, 10])
@pytest.mark.parametrize("implementation", ["eager", "fused"])
def test_invalid_host_query_width_is_rejected(query_width, implementation):
    function = build_dspark_swa_indices if implementation == "eager" else fused.build_dspark_swa_slot_indices_triton
    with pytest.raises(ValueError, match="num_query_per_req"):
        function(*inputs(), num_decode_tokens=9, index_width=256, num_query_per_req=query_width)
