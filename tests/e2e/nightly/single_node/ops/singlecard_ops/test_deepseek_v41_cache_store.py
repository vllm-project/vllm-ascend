# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate V4.1 fused stores against the PyTorch cache-write reference."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from tests.deepseek_v41_reference import scatter_cache
from vllm_ascend.attention.dsa_v41 import scatter_cache_sk
from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _slot_mapping_2d(slots, block_size):
    valid = slots >= 0
    physical = slots.clamp_min(0)
    indices = torch.stack(
        (
            torch.div(physical, block_size, rounding_mode="floor"),
            physical.remainder(block_size),
        ),
        dim=-1,
    ).to(torch.int32)
    indices[~valid] = -1
    return indices


def _reference_scatter(cache, slots, values):
    valid = slots >= 0
    scatter_cache(cache, slots[valid], values[valid])


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_dspark_context_store_uses_slot_backed_view_without_target_corruption(stage):
    config = make_cache_config(17, draft_layers=3)
    expected_raw, expected = allocate_cache_views(config, "npu")
    actual_raw, actual = allocate_cache_views(config, "npu")
    name = f"mtp.{stage}.self_attn.swa_cache"
    target = f"model.layers.{(2, 8, 14)[stage]}.self_attn.long_kv_cache"
    actual[target][2].fill_(7)
    expected[target][2].fill_(7)
    slots = torch.tensor([-1, 7 * 128 + 127, 13 * 128], dtype=torch.int64, device="npu")
    values = torch.randn(3, 512, dtype=torch.bfloat16, device="npu")
    model = SimpleNamespace(vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="bfloat16")))
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                kv_cache=[actual[name]],
                block_size=128,
            )
        )
    )
    DeepseekV4DSparkModel._store_standard_swa_kv(model, values.unsqueeze(1), slots, attn)
    _reference_scatter(expected[name], slots, values)
    torch.npu.synchronize()
    for actual_buffer, expected_buffer in zip(actual_raw, expected_raw):
        torch.testing.assert_close(actual_buffer.cpu(), expected_buffer.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,rows,width,dtype",
    [
        ("model.layers.3.self_attn.swa_cache", 128, 512, torch.bfloat16),
        ("model.layers.2.self_attn.long_kv_cache", 64, 512, torch.bfloat16),
        ("model.layers.20.self_attn.long_kv_cache", 128, 512, torch.bfloat16),
    ],
)
def test_fused_store_matches_reference_in_layer_slots(name, rows, width, dtype):
    torch.manual_seed(47)
    config = make_cache_config(7)
    expected_backing, expected = allocate_cache_views(config, "npu")
    actual_backing, actual = allocate_cache_views(config, "npu")
    slots = torch.tensor(
        [-1, rows + 3, 5 * rows + rows - 1],
        dtype=torch.int64,
        device="npu",
    )
    values = torch.randn(3, width, dtype=dtype, device="npu")

    _reference_scatter(expected[name], slots, values)
    indices = _slot_mapping_2d(slots, rows)
    scatter_cache_sk(actual[name], indices, values)
    torch.npu.synchronize()

    for expected_raw, actual_raw in zip(expected_backing, actual_backing):
        torch.testing.assert_close(actual_raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["random", "zero", "tiny"])
def test_indexer_dynamic_quant_and_fused_store_match_reference(kind):
    torch.manual_seed(53)
    config = make_cache_config(7)
    expected_backing, expected = allocate_cache_views(config, "npu")
    actual_backing, actual = allocate_cache_views(config, "npu")
    name = "model.layers.2.self_attn.indexer.k_cache"
    expected_key, expected_scale = expected[name]
    actual_key, actual_scale = actual[name]
    rows = expected_key.shape[1]
    slots = torch.tensor(
        [-1, rows + 1, 3 * rows + rows - 1],
        dtype=torch.int64,
        device="npu",
    )
    key = torch.randn(3, 128, dtype=torch.bfloat16, device="npu")
    if kind == "zero":
        key.zero_()
    elif kind == "tiny":
        key.mul_(1e-7)

    reference_scale = key.float().abs().amax(-1, keepdim=True).clamp_min_(1e-12) / 127.0
    reference_key = (key.float() / reference_scale).round_().clamp_(-127, 127).to(torch.int8)
    actual_quant, actual_quant_scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
    torch.testing.assert_close(actual_quant.cpu(), reference_key.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_quant_scale.float().cpu(),
        reference_scale.squeeze(-1).cpu(),
        rtol=1e-5,
        atol=1e-8,
    )

    _reference_scatter(expected_key, slots, reference_key)
    _reference_scatter(expected_scale, slots, reference_scale.to(torch.float16))
    indices = _slot_mapping_2d(slots, rows)
    scatter_cache_sk(actual_key, indices, actual_quant)
    scatter_cache_sk(
        actual_scale,
        indices,
        actual_quant_scale.unsqueeze(-1).to(torch.float16),
    )
    torch.npu.synchronize()

    for expected_raw, actual_raw in zip(expected_backing, actual_backing):
        torch.testing.assert_close(actual_raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,plane,width,dtype",
    [
        ("model.layers.3.self_attn.swa_cache", None, 512, torch.bfloat16),
        ("model.layers.2.self_attn.long_kv_cache", None, 512, torch.bfloat16),
        ("model.layers.20.self_attn.long_kv_cache", None, 512, torch.bfloat16),
        ("model.layers.2.self_attn.indexer.k_cache", 0, 128, torch.int8),
        ("model.layers.2.self_attn.indexer.k_cache", 1, 1, torch.float16),
    ],
)
def test_negative_coordinates_do_not_modify_packed_backing(name, plane, width, dtype):
    torch.manual_seed(59)
    config = make_cache_config(7)
    backing, caches = allocate_cache_views(config, "npu")
    cache = caches[name] if plane is None else caches[name][plane]
    before = [tensor.clone() for tensor in backing]
    indices = torch.full((3, 2), -1, dtype=torch.int32, device="npu")
    if dtype == torch.int8:
        values = torch.randint(-127, 128, (3, width), dtype=dtype, device="npu")
    else:
        values = torch.randn(3, width, dtype=dtype, device="npu")

    # The view is packed into a shared allocation: its page stride is larger
    # than the contiguous stride implied by the visible plane shape.
    squeezed = cache.squeeze(-2)
    assert squeezed.stride(0) > squeezed.shape[1] * squeezed.stride(1)
    scatter_cache_sk(cache, indices, values)
    torch.npu.synchronize()

    for expected_raw, actual_raw in zip(before, backing):
        torch.testing.assert_close(actual_raw.cpu(), expected_raw.cpu(), rtol=0, atol=0)
