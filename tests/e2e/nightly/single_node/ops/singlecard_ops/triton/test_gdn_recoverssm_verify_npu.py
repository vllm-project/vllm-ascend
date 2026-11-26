# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-operator GDN RecoverSSM verify tests."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (  # noqa: E402
    WINDOW_CASES,
    ascendc_reference,
    check_accuracy,
    check_records,
    cpu_reference,
    model_inputs,
    npu_device,
    packed_layer,
    padded_layer,
    trace_launches,
    verify,
)
from vllm_ascend.ops.triton.gdn import gdn_recoverssm_verify as gdn  # noqa: E402

npu = pytest.fixture(npu_device, name="npu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_gates(dtype, record_property):
    expected = torch.zeros(100, dtype=torch.float64)
    actual = torch.zeros(100, dtype=dtype)
    actual[:1] = 0.005 if dtype == torch.float32 else 0.05
    check_accuracy(actual, expected, record_property, "99_percent")
    actual[:2] = 0.005 if dtype == torch.float32 else 0.05
    with pytest.raises(AssertionError, match="mixed tolerance"):
        check_accuracy(actual, expected, record_property, "98_percent")
    actual.zero_()
    actual[-1] = 4
    with pytest.raises(AssertionError, match="maximum error"):
        check_accuracy(actual, expected, record_property, "outlier")


def test_verify_preserves_head_padding(npu, record_property):
    hk, hv, key, value, spec_query_len = 4, 8, 128, 128, 3
    layer, checkpoint_storage, replay_storage = padded_layer(npu, hv, key, value, spec_query_len)
    _, checkpoint, replay = layer.kv_cache
    indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)[::2]
    offsets = torch.tensor([0, spec_query_len], dtype=torch.int32, device=npu)
    inputs = model_inputs(npu, spec_query_len, hk, hv, key, value)
    expected, _, records = cpu_reference(inputs, checkpoint, [0, spec_query_len], [1])
    before = checkpoint_storage.clone()
    output = verify(inputs, layer, offsets, indices, spec_query_len)
    check_accuracy(output, expected, record_property, "head_padding_verify")
    check_records(replay, records, value, record_property, "head_padding_records")
    torch.testing.assert_close(checkpoint_storage, before, atol=0, rtol=0)
    assert (replay_storage[..., -16:] == -7).all()


@pytest.mark.parametrize(
    "key,value,amplitude,normalize",
    [
        (7, 5, 1, False),
        (7, 8, 1, True),
        (8, 5, 1, True),
        (128, 128, 1e-3, True),
        (128, 128, 1e-8, True),
        (128, 128, 0, True),
    ],
)
def test_verify_ragged_tiles_and_small_norms(npu, record_property, key, value, amplitude, normalize):
    hk, hv, spec_query_len = 2, 6, 4
    layer, _ = packed_layer(npu, 3, hk, hv, key, value, spec_query_len, torch.float32, False)
    _, state, replay = layer.kv_cache
    a_log, a, b, bias, q, k, v = model_inputs(npu, spec_query_len, hk, hv, key, value)
    a[0], a[1] = -8, 8
    inputs = (a_log, a, b, bias, q * amplitude, k * amplitude, v)
    offsets = torch.tensor([0, -1, spec_query_len], device=npu, dtype=torch.int64)[::2]
    pages = torch.tensor([1], device=npu, dtype=torch.int64)
    expected, _, records = cpu_reference(
        inputs,
        state,
        [0, spec_query_len],
        [1],
        normalize=normalize,
        sp_beta=0.5,
        threshold=2.0,
        scale=0.25,
    )
    before = state.clone()
    out = torch.empty(1, spec_query_len * 2, hv, value, dtype=v.dtype, device=npu)[:, ::2]
    actual = verify(
        inputs,
        layer,
        offsets,
        pages,
        spec_query_len,
        out=out,
        scale=0.25,
        softplus_beta=0.5,
        softplus_threshold=2.0,
        use_qk_l2norm_in_kernel=normalize,
    )
    assert actual is out
    check_accuracy(actual, expected, record_property, "ragged_or_small_norm")
    check_records(replay, records, value, record_property, "replay")
    torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("amplitude", [1.0, 1e-3, 1e-8, 0.0])
def test_verify_small_norms_against_ascendc(npu, record_property, amplitude):
    inputs, layer, span, page, expected, _ = ascendc_reference(npu, amplitude)
    actual = verify(inputs, layer, span, page, layer.kv_cache[2].shape[2], use_qk_l2norm_in_kernel=False)
    check_accuracy(actual, expected, record_property, "ascendc_verify")


@pytest.mark.parametrize("offsets,pages", [([0], []), ([0, 3], [0]), ([0, 3], [-1])])
def test_verify_empty_and_null_rows(npu, offsets, pages):
    layer, backing = packed_layer(npu, 3, 2, 4, 16, 16, 3, torch.float32, True)
    before = backing.clone()
    inputs = model_inputs(npu, offsets[-1], 2, 4, 16, 16)
    starts = torch.tensor(offsets, dtype=torch.int32, device=npu)
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    out = verify(inputs, layer, starts, indices, 3)
    torch.testing.assert_close(out, torch.zeros_like(inputs[-1]), atol=0, rtol=0)
    torch.testing.assert_close(backing, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "hk,hv,key,value,spec_query_len,state_dtype",
    [case[:6] for case in WINDOW_CASES],
)
def test_verify_configured_window_dispatch(
    npu, monkeypatch, record_property, hk, hv, key, value, spec_query_len, state_dtype
):
    # Exercise the 128-wide value tiles on 950 using the production shapes.
    batch = max(3, math.ceil(96 / hv)) if key == value == 128 and hv % 2 == 0 else 3
    layer, _ = packed_layer(npu, batch + 1, hk, hv, key, value, spec_query_len, state_dtype, True)
    _, state, _ = layer.kv_cache
    pages = list(range(1, batch + 1))
    offsets = [row * spec_query_len for row in range(batch + 1)]
    starts = torch.tensor(offsets, dtype=torch.int32, device=npu)
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    inputs = model_inputs(npu, offsets[-1], hk, hv, key, value)
    name = "_gdn_recoverssm_verify_kernel"
    launches = trace_launches(monkeypatch, gdn, name)
    golden, _, _ = cpu_reference(inputs, state, offsets, pages)
    actual = verify(inputs, layer, starts, indices, spec_query_len)
    check_accuracy(actual, golden, record_property, "dispatch_verify")

    [(selected, grid, args)] = launches
    assert selected == name
    value_tile = 128 if gdn.is_950() and batch * hv >= 96 else 64
    assert args["BV"] == min(1 << (value - 1).bit_length(), value_tile)
    assert grid == (math.ceil(value / args["BV"]), hv, batch)
