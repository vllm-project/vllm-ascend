# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the MiniMax M3 Triton launch wrappers."""

import pytest
import torch

from vllm_ascend.ops.triton.msa import msa_m3_triton, msa_m3_triton_a5


@pytest.fixture(params=[msa_m3_triton, msa_m3_triton_a5], ids=["a2-a3", "a5"])
def operator_module(request):
    return request.param


@pytest.mark.parametrize("layout", ["plain", "head", "stacked", "tuple", "list"])
def test_index_cache_layout_preserves_values_and_strides(operator_module, layout):
    storage = torch.arange(3 * 128 * 16).reshape(3, 128, 16)
    expected = storage[..., ::2]
    cache = {
        "plain": expected,
        "head": expected.unsqueeze(2),
        "stacked": torch.stack((storage, -storage))[..., ::2].unsqueeze(3),
        "tuple": (expected, -expected),
        "list": [expected, -expected],
    }[layout]

    actual = operator_module._as_triton_index_kv_cache(cache)

    assert actual.shape == (3, 128, 8)
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 128), (3, 128, 2, 8), (3, 1, 1, 1, 8)])
def test_invalid_index_cache_layout_is_rejected(operator_module, shape):
    with pytest.raises(ValueError, match="Unexpected index cache"):
        operator_module._as_triton_index_kv_cache(torch.zeros(shape))


@pytest.mark.parametrize("layout", ["split", "list", "kv-first", "page-first"])
def test_main_cache_layout_preserves_key_value_order(operator_module, layout):
    key = torch.arange(3 * 128 * 2 * 8).reshape(3, 128, 2, 8)
    value = -key - 1
    cache = {
        "split": (key, value),
        "list": [key, value],
        "kv-first": torch.stack((key, value)),
        "page-first": torch.stack((key, value), dim=1),
    }[layout]

    if operator_module is msa_m3_triton:
        actual_key, actual_value = operator_module._split_triton_main_kv_cache(cache)
    else:
        fused = operator_module._as_triton_main_kv_cache(cache)
        assert fused.shape == (3, 2, 128, 2, 8)
        actual_key, actual_value = fused[:, 0], fused[:, 1]

    torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, value, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 128, 2, 8), (3, 3, 128, 2, 8)])
def test_invalid_main_cache_layout_is_rejected(operator_module, shape):
    normalize = (
        operator_module._split_triton_main_kv_cache
        if operator_module is msa_m3_triton
        else operator_module._as_triton_main_kv_cache
    )
    with pytest.raises(ValueError, match="Unexpected.*kv cache"):
        normalize(torch.zeros(shape))


@pytest.mark.parametrize("selected", [0, 2, 4])
@pytest.mark.parametrize("supplied_output", [False, True])
def test_topk_copy_pads_short_rows_and_preserves_output_guards(selected, supplied_output):
    heads, tokens, requested = 2, 3, 4
    raw = torch.arange(heads * tokens * selected).reshape(heads, tokens, selected)
    storage = torch.full((heads, (tokens + 1) * 2, (requested + 1) * 2), -77, dtype=torch.int32)
    output = storage[:, ::2, ::2] if supplied_output else None

    actual = msa_m3_triton._copy_topk_indices(raw, requested, output)

    assert actual.shape == (heads, tokens, requested)
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual[..., :selected], raw.to(torch.int32), rtol=0, atol=0)
    assert torch.all(actual[..., selected:] == -1)
    if supplied_output:
        assert actual.data_ptr() == output.data_ptr()
        changed = torch.zeros_like(storage, dtype=torch.bool)
        changed[:, : tokens * 2 : 2, : requested * 2 : 2] = True
        assert torch.all(storage[~changed] == -77)


@pytest.mark.parametrize("selected", [0, 2, 4])
def test_topk_score_copy_pads_with_negative_infinity(selected):
    raw = torch.arange(2 * 3 * selected, dtype=torch.float32).reshape(2, 3, selected)
    actual = msa_m3_triton._copy_topk_values(raw, 4)

    assert actual.shape == (2, 3, 4)
    assert actual.dtype == raw.dtype
    torch.testing.assert_close(actual[..., :selected], raw, rtol=0, atol=0)
    assert torch.all(torch.isneginf(actual[..., selected:]))
