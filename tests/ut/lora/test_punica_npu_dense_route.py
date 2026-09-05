# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""The dense single-adapter LoRA route used by UNO's gated draft forward.

Two properties are load-bearing and neither is visible in generated text:

1. rows routed to the base model must receive a *bit-identical* output. The
   PyTorch fallback punica selects at ``max_lora_rank >= 128`` gets this wrong
   -- it fancy-indexes the stacked weights with ``-1``, which wraps to the last
   slot and applies the adapter to base rows;
2. the delta on adapted rows must match the per-row semantics ``add_lora_linear``
   documents.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

_BASE_ROW = -1


def _make_wrapper(*, slot: int, row_mask: torch.Tensor | None) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._dense_lora_slot = slot
    wrapper._dense_row_mask = row_mask
    return wrapper


def _select(stacked: torch.Tensor, slot: int) -> torch.Tensor:
    weight = stacked[slot]
    while weight.dim() > 2:
        weight = weight.squeeze(0)
    return weight


def _per_row_reference(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, token_lora_indices, slot):
    """``add_lora_linear``'s documented semantics, written out one row at a time."""
    out = y.clone()
    offset = 0
    for slice_idx in range(len(lora_a_stacked)):
        weight_a = _select(lora_a_stacked[slice_idx], slot).to(torch.float32)
        weight_b = _select(lora_b_stacked[slice_idx], slot)
        slice_size = output_slices[slice_idx]
        for row in range(x.shape[0]):
            if token_lora_indices[row] != slot:
                continue
            shrunk = (x[row].to(torch.float32) @ weight_a.t()) * scale
            out[row, offset : offset + slice_size] += shrunk.to(out.dtype) @ weight_b.to(out.dtype).t()
        offset += slice_size
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "rank, in_features, output_slices",
    [
        (128, 64, (32, 16, 16)),  # qkv_proj
        (128, 64, (48, 48)),  # gate_up_proj
        (64, 96, (40,)),  # o_proj / down_proj
    ],
)
def test_dense_route_matches_per_row_reference(dtype, rank, in_features, output_slices):
    torch.manual_seed(0)
    forward_width, num_reqs, slot, scale = 4, 3, 0, 1.0
    num_rows = forward_width * num_reqs

    lora_a_stacked = tuple(torch.randn(2, 1, rank, in_features, dtype=dtype) for _ in output_slices)
    lora_b_stacked = tuple(torch.randn(2, 1, size, rank, dtype=dtype) for size in output_slices)
    x = torch.randn(num_rows, in_features, dtype=dtype)
    y = torch.randn(num_rows, sum(output_slices), dtype=dtype)

    # UNO's routing: the seed row of every request is base, the noise rows carry
    # the adapter.
    token_lora_indices = torch.tensor(
        ([_BASE_ROW] + [slot] * (forward_width - 1)) * num_reqs,
        dtype=torch.long,
    )
    row_mask = (token_lora_indices == slot).to(torch.float32).unsqueeze(1)
    wrapper = _make_wrapper(slot=slot, row_mask=row_mask)

    buffers = tuple(torch.zeros((num_rows, rank), dtype=torch.float32) for _ in output_slices)
    actual = y.clone()
    wrapper._dense_shrink(buffers, x, lora_a_stacked, scale)
    wrapper._dense_expand(actual, buffers, lora_b_stacked, output_slices, 0, True)

    expected = _per_row_reference(y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, token_lora_indices, slot)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    base_rows = (token_lora_indices != slot).nonzero().flatten()
    assert torch.equal(actual[base_rows], y[base_rows]), (
        "base rows must be bit-identical; a nonzero delta there breaks UNO's KV reuse at the frontier"
    )
    adapted_rows = (token_lora_indices == slot).nonzero().flatten()
    assert not torch.equal(actual[adapted_rows], y[adapted_rows]), "the adapter had no effect on the noise rows"


def test_dense_route_all_rows_adapted_needs_no_mask():
    torch.manual_seed(1)
    rank, in_features, output_slices, slot = 32, 24, (16,), 0
    lora_a_stacked = (torch.randn(2, 1, rank, in_features),)
    lora_b_stacked = (torch.randn(2, 1, output_slices[0], rank),)
    x = torch.randn(5, in_features)
    y = torch.zeros(5, output_slices[0])

    wrapper = _make_wrapper(slot=slot, row_mask=None)
    buffers = (torch.zeros((5, rank), dtype=torch.float32),)
    wrapper._dense_shrink(buffers, x, lora_a_stacked, 1.0)
    wrapper._dense_expand(y, buffers, lora_b_stacked, output_slices, 0, True)

    expected = (x @ lora_a_stacked[0][slot, 0].t()) @ lora_b_stacked[0][slot, 0].t()
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def test_dense_expand_honours_offset_and_add_inputs():
    torch.manual_seed(2)
    rank, output_slices, slot = 8, (4, 6), 0
    lora_b_stacked = tuple(torch.randn(2, 1, size, rank) for size in output_slices)
    buffers = tuple(torch.randn(3, rank) for _ in output_slices)
    y = torch.randn(3, 2 + sum(output_slices))
    original = y.clone()

    wrapper = _make_wrapper(slot=slot, row_mask=None)
    wrapper._dense_expand(y, buffers, lora_b_stacked, output_slices, 2, False)

    # Columns before the offset are untouched; each slice is overwritten.
    torch.testing.assert_close(y[:, :2], original[:, :2])
    torch.testing.assert_close(y[:, 2:6], buffers[0] @ lora_b_stacked[0][slot, 0].t(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y[:, 6:], buffers[1] @ lora_b_stacked[1][slot, 0].t(), rtol=1e-4, atol=1e-4)


def _route_for(index_mapping, lora_index_to_id, *, prefers_dense, max_batches):
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._prefers_dense_single_adapter = prefers_dense
    wrapper._max_batches = max_batches
    mapping = SimpleNamespace(index_mapping=tuple(index_mapping), is_prefill=True)
    return wrapper._dense_route_slot(mapping, lora_index_to_id)


def test_dense_route_selected_when_kernels_fell_back_to_torch():
    # UNO: 2 requests x (1 base row + 3 adapted rows), one adapter with id 7.
    mapping = ([0, 7, 7, 7]) * 2
    assert _route_for(mapping, [7], prefers_dense=True, max_batches=64) == 0


def test_dense_route_selected_when_segments_overflow_the_sgmv_buffers():
    # Per-row routing yields two unique_consecutive runs per request, so a full
    # batch produces 2 * max_batches segments and overflows _seq_start_locs.
    mapping = ([0, 7, 7, 7]) * 64
    assert _route_for(mapping, [7], prefers_dense=False, max_batches=64) == 0
    # ... but a per-request mapping with the same batch size fits, and must keep
    # using the fused kernels.
    per_request = [7] * 256
    assert _route_for(per_request, [7], prefers_dense=False, max_batches=64) is None


def test_dense_route_declined_for_multi_adapter_batches():
    mapping = [0, 7, 7, 0, 9, 9]
    assert _route_for(mapping, [7, 9], prefers_dense=True, max_batches=64) is None


def test_dense_route_declined_for_base_only_batches():
    assert _route_for([0, 0, 0], [7], prefers_dense=True, max_batches=64) is None


def test_dense_route_resolves_the_slot_not_the_id():
    # lora_index_to_id maps stacked-weight slot -> adapter id; UNO's adapter can
    # land in any slot and the dense path must index by slot.
    mapping = [0, 5, 5, 5]
    assert _route_for(mapping, [None, 5], prefers_dense=True, max_batches=64) == 1


def _base_only_wrapper():
    """A wrapper with the dense route already installed from a previous step."""
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._prefers_dense_single_adapter = True
    wrapper._max_batches = 64
    wrapper._dense_lora_slot = 0
    wrapper._dense_row_mask = torch.ones(4, 1)
    wrapper._dense_sampler_mask = torch.ones(4, 1)
    wrapper.no_lora = False
    wrapper.is_prefill = True
    wrapper._token_lora_indices = torch.full((8,), _BASE_ROW, dtype=torch.long)
    wrapper._sampler_indices = torch.full((8,), _BASE_ROW, dtype=torch.long)
    wrapper._dense_row_mask_buffer = torch.empty(8, 1)
    wrapper._dense_sampler_mask_buffer = torch.empty(8, 1)
    wrapper._base_calls = []
    wrapper._update_base_metadata = lambda *args: wrapper._base_calls.append(args)
    return wrapper


def test_dense_masks_keep_capture_addresses_across_base_and_smaller_gated_batches():
    wrapper = _base_only_wrapper()
    wrapper._token_lora_indices[:4] = torch.tensor([-1, 0, 0, 0])
    wrapper._sampler_indices[:4] = torch.tensor([-1, 0, 0, 0])
    gated = SimpleNamespace(index_mapping=(0, 7, 7, 7), prompt_mapping=(0, 7, 7, 7), is_prefill=True)
    wrapper.update_metadata(gated, [7], 1, 32000)
    captured_row_mask = wrapper._dense_row_mask
    captured_sampler_mask = wrapper._dense_sampler_mask
    assert captured_row_mask[:, 0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    base = SimpleNamespace(index_mapping=(0,) * 4, prompt_mapping=(0,) * 4, is_prefill=False)
    wrapper.update_metadata(base, [7], 1, 32000)
    assert wrapper.no_lora and wrapper._dense_row_mask is None
    smaller = SimpleNamespace(index_mapping=(0, 7), prompt_mapping=(0, 7), is_prefill=True)
    wrapper.update_metadata(smaller, [7], 1, 32000)
    assert wrapper._dense_row_mask is captured_row_mask
    assert wrapper._dense_sampler_mask is captured_sampler_mask
    assert captured_row_mask[:, 0].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert torch.equal(captured_sampler_mask, captured_row_mask)


def test_a_base_only_mapping_skips_the_sgmv_segment_metadata():
    """`compute_meta` blocks the host on two `.item()` reads.

    UNO runs a base-only mapping once per decode step (the verify forward), so
    building segment metadata nothing reads puts two device syncs on the
    critical path.
    """
    wrapper = _base_only_wrapper()
    mapping = SimpleNamespace(index_mapping=(0, 0, 0, 0), prompt_mapping=(0, 0, 0, 0), is_prefill=False)

    wrapper.update_metadata(mapping, [7], 1, 32000)

    assert wrapper.no_lora is True
    assert wrapper._dense_lora_slot is None
    assert wrapper._dense_row_mask is None and wrapper._dense_sampler_mask is None
    # The per-token indices still have to be refreshed: they are what the
    # decode ops would read if anything did run.
    assert len(wrapper._base_calls) == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda w, y, x: w._shrink_decode(y, x, torch.randn(1, 1, 4, 8), 1.0),
        lambda w, y, x: w._expand_decode(y, x, torch.randn(1, 1, 16, 8), True),
        lambda w, y, x: w._expand_slice_decode(y, x, torch.randn(1, 1, 16, 8), 0, 16, True),
        lambda w, y, x: w.add_lora_logits(y, x, torch.randn(1, 1, 4, 8), torch.randn(1, 1, 16, 4), 1.0),
    ],
)
def test_decode_entry_points_are_inert_on_a_base_only_forward(call):
    """`weights[-1]` wraps round in the PyTorch reference ops.

    At `max_lora_rank >= 128` those ops are what punica selects, so a decode
    that still called them would apply the adapter to every base row rather
    than skipping it.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("a LoRA op ran on a base-only forward")

    wrapper = _base_only_wrapper()
    wrapper.no_lora = True
    wrapper._dense_lora_slot = None
    wrapper.bgmv_shrink = _boom
    wrapper.bgmv_expand = _boom
    wrapper.bgmv_expand_slice = _boom

    y = torch.randn(4, 16)
    x = torch.randn(4, 8)
    before = y.clone()

    call(wrapper, y, x)

    assert torch.equal(y, before)


def test_the_no_lora_guards_do_not_disable_the_adapted_path():
    ran = []
    wrapper = _base_only_wrapper()
    wrapper.no_lora = False
    wrapper._dense_lora_slot = None
    wrapper.bgmv_shrink = lambda *a, **k: ran.append("shrink")
    wrapper.bgmv_expand = lambda *a, **k: ran.append("expand")
    wrapper.bgmv_expand_slice = lambda *a, **k: ran.append("expand_slice")

    y = torch.randn(4, 16)
    x = torch.randn(4, 8)
    wrapper._shrink_decode(y, x, torch.randn(1, 1, 4, 8), 1.0)
    wrapper._expand_decode(y, x, torch.randn(1, 1, 16, 8), True)
    wrapper._expand_slice_decode(y, x, torch.randn(1, 1, 16, 8), 0, 16, True)

    assert ran == ["shrink", "expand", "expand_slice"]
