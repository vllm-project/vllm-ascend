# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU scatter writes checked against an independent, whole-cache CPU oracle.

The kernel is never mocked. Without torch_npu and an available Ascend NPU,
these tests skip explicitly. A standalone operator run can use --noconftest;
no model, server, or distributed process group is required.
"""

import importlib
import math

import pytest
import torch


@pytest.fixture(scope="module")
def scatter():
    torch_npu = pytest.importorskip("torch_npu", reason="suffix KV scatter requires real torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("suffix KV scatter requires an available Ascend NPU")
    # Import the real native kernel only after the device dependency is present.
    return importlib.import_module("vllm_ascend.ops.triton.spec_decode.suffix_kv_cache").scatter


def _values(shape, offset):
    values = torch.arange(math.prod(shape), dtype=torch.float32, device="cpu")
    return (values.remainder(251).sub_(125).div_(16).add_(offset)).to(torch.bfloat16).reshape(shape)


def _input_tensor(values, step):
    if step == 1:
        return values.to("npu")
    backing = torch.full(tuple(size * step for size in values.shape), -91, dtype=torch.bfloat16, device="npu")
    view = backing[::step, ::step, ::step]
    view.copy_(values.to("npu"))
    assert not view.is_contiguous()
    return view


def _inputs(*, head_major=False, tokens=8, heads=8, dim=128, block_size=128, blocks=3, strided=False):
    shape = (tokens, heads, dim)
    cache_shape = (blocks, heads, block_size, dim) if head_major else (blocks, block_size, heads, dim)
    return {
        "key": _input_tensor(_values(shape, 0), 2 if strided else 1),
        "value": _input_tensor(_values(shape, 4), 3 if strided else 1),
        "key_cache": _values(cache_shape, -16).to("npu"),
        "value_cache": _values(cache_shape, -32).to("npu"),
        "slots": torch.arange(tokens, dtype=torch.int64, device="npu"),
        "head_major": head_major,
    }


def _logical_cache(cache, head_major):
    cpu = cache.cpu()
    return cpu.permute(0, 2, 1, 3).contiguous() if head_major else cpu


def _assert_scatter_matches_cpu(scatter, inputs):
    expected_key = _logical_cache(inputs["key_cache"], inputs["head_major"])
    expected_value = _logical_cache(inputs["value_cache"], inputs["head_major"])
    keys_before = inputs["key"].cpu()
    values_before = inputs["value"].cpu()
    slots_before = inputs["slots"].cpu()
    blocks, block_size = expected_key.shape[:2]
    # This logical [block, token, head, dim] write does not reuse kernel strides
    # or native helper functions. Unselected cells retain their initial values.
    for token, slot in enumerate(slots_before.tolist()):
        if 0 <= slot < blocks * block_size:
            block, position = divmod(slot, block_size)
            expected_key[block, position].copy_(keys_before[token])
            expected_value[block, position].copy_(values_before[token])

    assert scatter(**inputs) is None
    torch.npu.synchronize()
    torch.testing.assert_close(_logical_cache(inputs["key_cache"], inputs["head_major"]), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(
        _logical_cache(inputs["value_cache"], inputs["head_major"]), expected_value, rtol=0, atol=0
    )
    torch.testing.assert_close(inputs["key"].cpu(), keys_before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["value"].cpu(), values_before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["slots"].cpu(), slots_before, rtol=0, atol=0)


@pytest.mark.parametrize("head_major", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous-inputs", "strided-inputs"])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64], ids=["int32-slots", "int64-slots"])
def test_scatter_cross_block_and_padding_matches_whole_cache(scatter, head_major, strided, slot_dtype):
    inputs = _inputs(head_major=head_major, strided=strided)
    # Both sides of block boundaries, first/last valid slots, and padding whose
    # source values must not overwrite any destination, including slot zero.
    inputs["slots"] = torch.tensor([0, 127, 128, -1, 255, 256, -1, 383], dtype=slot_dtype, device="npu")
    _assert_scatter_matches_cpu(scatter, inputs)


@pytest.mark.parametrize("head_major", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize(
    "slots",
    [
        pytest.param([-1, -1, -1, -1], id="all-padding"),
        pytest.param([-2, -(2**40), 384, 2**40], id="masked-out-of-range-slots"),
    ],
)
def test_scatter_invalid_slots_leave_every_cache_cell_unchanged(scatter, head_major, slots):
    inputs = _inputs(head_major=head_major, tokens=len(slots), strided=True)
    inputs["slots"] = torch.tensor(slots, dtype=torch.int64, device="npu")
    _assert_scatter_matches_cpu(scatter, inputs)


@pytest.mark.parametrize("head_major", [False, True], ids=["NHD", "HND"])
def test_scatter_empty_inputs_leave_every_cache_cell_unchanged(scatter, head_major):
    _assert_scatter_matches_cpu(scatter, _inputs(head_major=head_major, tokens=0))


@pytest.mark.parametrize(
    "replace",
    [
        pytest.param(lambda data: {"key": data["key"].flatten(1)}, id="input-rank"),
        pytest.param(lambda data: {"value": data["value"][:-1]}, id="input-shape-mismatch"),
        pytest.param(lambda data: {"key_cache": data["key_cache"].flatten(1, 2)}, id="cache-rank"),
        pytest.param(lambda data: {"value_cache": data["value_cache"][:-1]}, id="cache-pair-mismatch"),
        pytest.param(lambda data: {"slots": data["slots"].unsqueeze(0)}, id="slots-rank"),
        pytest.param(lambda data: {"slots": data["slots"].repeat(2)[::2]}, id="slots-not-dense"),
        pytest.param(lambda data: {"slots": data["slots"][:-1]}, id="slot-count"),
        pytest.param(lambda data: {"slots": data["slots"].to(torch.float32)}, id="slot-dtype"),
        pytest.param(lambda data: {"slots": data["slots"].cpu()}, id="slots-on-cpu"),
        pytest.param(lambda data: {"key": data["key"].to(torch.float16)}, id="key-not-bf16"),
        pytest.param(lambda data: {"value": data["value"].to(torch.float32)}, id="value-not-bf16"),
        pytest.param(lambda data: {"key_cache": data["key_cache"].to(torch.float16)}, id="cache-not-bf16"),
        pytest.param(lambda data: {"value": data["value"].cpu()}, id="mixed-device-inputs"),
        pytest.param(
            lambda data: {
                "key_cache": data["key_cache"].transpose(1, 2),
                "value_cache": data["value_cache"].transpose(1, 2),
            },
            id="noncontiguous-caches",
        ),
        pytest.param(
            lambda data: {
                "key_cache": data["key_cache"][:, :, :-1, :].contiguous(),
                "value_cache": data["value_cache"][:, :, :-1, :].contiguous(),
            },
            id="cache-head-count-mismatch",
        ),
        pytest.param(
            lambda data: {
                "key_cache": data["key_cache"][:, :, :, :-1].contiguous(),
                "value_cache": data["value_cache"][:, :, :, :-1].contiguous(),
            },
            id="cache-head-dimension-mismatch",
        ),
    ],
)
def test_scatter_rejects_malformed_tensors_before_writing(scatter, replace):
    inputs = _inputs()
    inputs.update(replace(inputs))
    key_cache_before = inputs["key_cache"].cpu()
    value_cache_before = inputs["value_cache"].cpu()
    with pytest.raises(ValueError):
        scatter(**inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(inputs["key_cache"].cpu(), key_cache_before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["value_cache"].cpu(), value_cache_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param({"heads": 0}, id="zero-heads"),
        pytest.param({"dim": 0}, id="zero-head-dimension"),
        pytest.param({"block_size": 0}, id="zero-block-size"),
        pytest.param({"blocks": 0}, id="zero-cache-blocks"),
        pytest.param({"heads": 3}, id="non-power-of-two-heads"),
        pytest.param({"dim": 96}, id="non-power-of-two-head-dimension"),
    ],
)
def test_scatter_rejects_unsupported_dimensions_before_writing(scatter, geometry):
    inputs = _inputs(**geometry)
    key_cache_before = inputs["key_cache"].cpu()
    value_cache_before = inputs["value_cache"].cpu()
    with pytest.raises(ValueError):
        scatter(**inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(inputs["key_cache"].cpu(), key_cache_before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["value_cache"].cpu(), value_cache_before, rtol=0, atol=0)
