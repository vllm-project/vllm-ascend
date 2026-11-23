# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device acceptance tests; CPU mocks do not cover these checks.

Set VLLM_ASCEND_DSV41_TEST_CONFIG to the config.json used in deployment to
include real model dimensions. Default dimensions below are interface cases,
not a claim that all deployed model configurations have been covered.
"""

import json
import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.compressor.compressor_triton import compressor_from_projected
from vllm_ascend.ops.triton.indexer_k_fusion import indexer_k_norm_rope_int8_store

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Ordinary A2/A3 INT8 fusion; test A5 fallback separately",
)


def _dimensions():
    path = os.environ.get("VLLM_ASCEND_DSV41_TEST_CONFIG")
    if path is None:
        return 128, 64, 512, 1e-6
    with open(path) as file:
        config = json.load(file)
    config = config.get("text_config", config)
    return (
        int(config["index_head_dim"]), int(config["qk_rope_head_dim"]),
        int(config["head_dim"]), float(config["rms_norm_eps"]),
    )


def _assert_float_accuracy(actual, expected, dtype):
    # Both gates are required by triton_operators.md. FP32 subtraction avoids
    # concealing differences through low-precision error calculation.
    limits = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }
    rtol, atol, maximum = limits[dtype]
    actual, expected = actual.float(), expected.float()
    if actual.numel() == 0:
        return
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(expected).all())
    error = (actual - expected).abs()
    passed = error <= atol + rtol * expected.abs()
    assert passed.float().mean().item() >= 0.99
    assert error.max().item() <= maximum


def _native_reference(k_pre, weight, eps, cos, sin):
    # Match the production baseline, not a translation of the Triton algorithm.
    normalized, _ = torch_npu.npu_rms_norm(k_pre.contiguous(), weight.contiguous(), eps)
    rotary_dim = cos.shape[-1]
    key = normalized.clone()
    rotated = torch_npu.npu_rotary_mul(
        normalized[:, None, None, -rotary_dim:].contiguous(),
        cos.reshape(-1, 1, 1, rotary_dim).contiguous(),
        sin.reshape(-1, 1, 1, rotary_dim).contiguous(),
        rotary_mode="interleave",
    )
    key[:, -rotary_dim:] = rotated.reshape(-1, rotary_dim)
    quantized, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
    return quantized, scale.reshape(-1).half()


# Correlated cases avoid a large redundant Cartesian product. Counts cover
# empty/single-row, 16-row boundaries and a larger batch; the layout cases
# exercise the stride contract without inventing a new production shape.
@pytest.mark.parametrize("tokens,layout,slot_dtype,weight_dtype", [
    (0, "dense", torch.int32, torch.bfloat16),
    (1, "dense", torch.int32, torch.bfloat16),
    (15, "padded", torch.int64, torch.float32),
    (16, "four_dimensional", torch.int32, torch.bfloat16),
    (17, "padded", torch.int64, torch.bfloat16),
    (257, "dense", torch.int32, torch.bfloat16),
])
@torch.no_grad()
def test_indexer_matches_native_chain(tokens, layout, slot_dtype, weight_dtype):
    torch.manual_seed(1234)
    width, rotary_dim, _, eps = _dimensions()
    device = "npu"
    row_width = width + 16 if layout == "padded" else width
    projection_base = torch.randn(tokens, row_width, dtype=torch.bfloat16, device=device)
    projection = projection_base[:, :width]
    if tokens:
        projection[0].zero_()  # Native zero-scale semantics are part of acceptance.
    weight_base = torch.randn(width * 2, dtype=weight_dtype, device=device)
    weight = weight_base[::2] if layout == "padded" else weight_base[:width]
    angles = torch.randn(tokens, rotary_dim // 2, device=device)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    page_size = 16
    num_blocks = (tokens + page_size - 1) // page_size + 1
    key_storage = torch.full(
        (num_blocks, page_size + 1, 1, width + 16), -91, dtype=torch.int8, device=device,
    )
    scale_storage = torch.full(
        (num_blocks, page_size + 1, 1, 2), -7.0, dtype=torch.float16, device=device,
    )
    keys = key_storage[:, :page_size, :, :width]
    scales = scale_storage[:, :page_size, :, :1]
    row_ids = torch.arange(tokens, device=device)
    slots = torch.stack((row_ids // page_size, row_ids % page_size), dim=1).to(slot_dtype)
    # Include all four invalid-coordinate classes, not just the negative sentinel.
    if tokens >= 15:
        invalid = torch.tensor(
            [[-1, 0], [0, -1], [num_blocks, 0], [0, page_size]], device=device, dtype=slot_dtype,
        )
        slots[-4:] = invalid
    before_inputs = [t.clone() for t in (projection, weight, cos, sin, slots)]
    key_before, scale_before = key_storage.clone(), scale_storage.clone()
    call_cos = cos[:, None, None, :] if layout == "four_dimensional" else cos
    call_sin = sin[:, None, None, :] if layout == "four_dimensional" else sin
    indexer_k_norm_rope_int8_store(projection, weight, eps, call_cos, call_sin, slots, keys, scales)
    torch.npu.synchronize()
    for actual, expected in zip((projection, weight, cos, sin, slots), before_inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if tokens == 0:
        torch.testing.assert_close(key_storage, key_before, rtol=0, atol=0)
        torch.testing.assert_close(scale_storage, scale_before, rtol=0, atol=0)
        return
    reference_keys, reference_scales = _native_reference(projection, weight, eps, cos, sin)
    # Host reads are allowed in acceptance tests; never copy this into the wrapper.
    coordinates = slots.cpu().tolist()
    expected_keys, expected_scales = key_before.clone(), scale_before.clone()
    valid_rows = []
    for row, (block, offset) in enumerate(coordinates):
        if 0 <= block < num_blocks and 0 <= offset < page_size:
            valid_rows.append(row)
            expected_keys[block, offset, 0, :width] = reference_keys[row]
            expected_scales[block, offset, 0, 0] = reference_scales[row]
    # Integer quantization output uses an exact gate. If this fails, investigate
    # native rounding/division/RMS reduction instead of widening INT8 tolerance.
    torch.testing.assert_close(key_storage, expected_keys, rtol=0, atol=0)
    written_scales = torch.zeros_like(scale_storage, dtype=torch.bool)
    for row in valid_rows:
        block, offset = coordinates[row]
        written_scales[block, offset, 0, 0] = True
    torch.testing.assert_close(
        scale_storage[~written_scales], expected_scales[~written_scales], rtol=0, atol=0,
    )
    _assert_float_accuracy(
        scale_storage[written_scales], expected_scales[written_scales], torch.float16,
    )


@pytest.mark.parametrize("magnitude", [1.0, 2**-20, 2**10])
@torch.no_grad()
def test_indexer_quantization_ties_and_scale_ranges(magnitude):
    width, rotary_dim, _, eps = _dimensions()
    if width < 8:
        pytest.skip("Tie vector requires at least eight columns")
    projection = torch.ones(1, width, dtype=torch.bfloat16, device="npu")
    weight = torch.zeros(width, dtype=torch.bfloat16, device="npu")
    weight[:8] = torch.tensor(
        [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127], dtype=torch.bfloat16, device="npu",
    ) * magnitude
    cos = torch.ones(1, rotary_dim, device="npu")
    sin = torch.zeros_like(cos)
    slots = torch.tensor([[0, 0]], dtype=torch.int32, device="npu")
    keys = torch.empty(1, 1, 1, width, dtype=torch.int8, device="npu")
    scales = torch.empty(1, 1, 1, 1, dtype=torch.float16, device="npu")
    indexer_k_norm_rope_int8_store(projection, weight, eps, cos, sin, slots, keys, scales)
    reference_keys, reference_scales = _native_reference(projection, weight, eps, cos, sin)
    torch.testing.assert_close(keys[0, 0, 0], reference_keys[0], rtol=0, atol=0)
    _assert_float_accuracy(scales.flatten(), reference_scales, torch.float16)


@pytest.mark.parametrize("start,length,use_gather", [
    (0, 1, False), (0, 2, False), (1, 1, False), (1, 3, True), (31, 3, True),
])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_c2_fused_norm_matches_separate_norm(start, length, use_gather, weight_dtype):
    torch.manual_seed(17)
    _, _, width, eps = _dimensions()
    kv = torch.randn(length, width, device="npu", dtype=torch.float32)
    scores = torch.randn_like(kv)
    weight = torch.randn(width, device="npu", dtype=weight_dtype)
    # Block 0 is reserved; batch 1 is padding. The first request uses block 1.
    metadata = torch.tensor(
        [[start, 0], [length, 0], [0, 0], [0, 0], [1, 0]], device="npu", dtype=torch.int32,
    )
    initial_state = torch.randn(3, 32, 2 * width, device="npu", dtype=torch.float32)
    reference_state, fused_state = initial_state.clone(), initial_state.clone()
    prev_rows = torch.randn(1, 2 * width, device="npu") if use_gather else None
    prev_indices = torch.tensor([0, -1], device="npu", dtype=torch.int64) if use_gather else None
    kwargs = dict(
        max_query_len=length, num_cores=1, prev_rows=prev_rows, prev_row_indices=prev_indices,
    )
    readonly = (kv, scores, weight, metadata)
    if use_gather:
        readonly += (prev_rows, prev_indices)
    readonly_before = [tensor.clone() for tensor in readonly]
    raw = compressor_from_projected(
        kv, scores, reference_state, metadata, torch.empty_like(kv, dtype=torch.bfloat16),
        **kwargs, norm_w=None,
    )
    expected, _ = torch_npu.npu_rms_norm(raw, weight, eps)
    output = torch.full_like(kv, 5, dtype=torch.bfloat16)
    result = compressor_from_projected(
        kv, scores, fused_state, metadata, output, **kwargs, norm_w=weight, rms_eps=eps,
    )
    torch.npu.synchronize()
    assert result.data_ptr() == output.data_ptr()
    for actual, before in zip(readonly, readonly_before):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    _assert_float_accuracy(result, expected, torch.bfloat16)
    torch.testing.assert_close(fused_state, reference_state, rtol=0, atol=0)
    if use_gather:
        torch.testing.assert_close(fused_state, initial_state, rtol=0, atol=0)
    # Incomplete pairs remain zero; they must not be normalized a second time.
    completion_rows = [row for row in range(length) if (start + row + 1) % 2 == 0]
    untouched_rows = [row for row in range(length) if row not in completion_rows]
    if untouched_rows:
        torch.testing.assert_close(result[untouched_rows], torch.zeros_like(result[untouched_rows]), rtol=0, atol=0)
