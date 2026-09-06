# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention.dsa_attn_kv_plan import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
    get_dsa_attn_kv_plan,
    get_dsv4_attn_kv_dtype,
    is_a5_bf16_kv_enabled,
    resolve_dsv4_cache_dtype,
)
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType

_DSA_C_ASCEND_OPS = (
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "kv_compress_epilog",
    "npu_scatter_nd_update_sk",
)


@pytest.fixture(autouse=True)
def _stub_dsa_c_ascend_ops():
    # CPU images do not register these custom ops on torch.ops._C_ascend.
    with ExitStack() as stack:
        for name in _DSA_C_ASCEND_OPS:
            stack.enter_context(mock.patch.object(torch.ops._C_ascend, name, create=True, new=MagicMock()))
        yield


def _config(use_bf16: bool):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="bfloat16" if use_bf16 else "auto"))


def _cache_config(cache_dtype: str = "auto"):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


def _on(device_type):
    return mock.patch(
        "vllm_ascend.attention.dsa_attn_kv_plan.get_current_hardware_profile",
        return_value=get_hardware_profile(device_type),
    )


def test_get_dsa_attn_kv_plan_requires_vllm_config():
    with pytest.raises(TypeError):
        get_dsa_attn_kv_plan()


def test_a5_fp8_plan_uses_flat_shared_kv():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert not plan.requires_block_offset_slots
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}


def test_a5_bf16_plan_uses_sparse_flash_mla():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_sparse_attn_op() is sparse_flash_mla
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert not plan.requires_block_offset_slots
        torch.testing.assert_close(
            plan.format_dsa_slot_mapping(torch.tensor([5, -1], dtype=torch.int32), 128),
            torch.tensor([5, -1], dtype=torch.int32),
        )


def test_non_a5_plan_preserves_shared_kv_runtime_kwargs():
    with _on(AscendDeviceType.A3):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        assert plan.requires_block_offset_slots
        kwargs: dict[str, Any] = {}
        plan.add_dsa_sparse_attn_extra_kwargs(kwargs, cu_seqlens_ori_kv=torch.tensor([0, 1]))
        assert "cu_seqlens_ori_kv" in kwargs


def test_scatter_skips_none_updates():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        cache = torch.zeros(2, 1, 4)
        with mock.patch.object(torch.ops._C_ascend, "kv_compress_epilog") as epilog:
            plan.dsa_kv_compress_scatter(cache, None, torch.tensor([0], dtype=torch.int32))
            epilog.assert_not_called()


def _cpu_scatter_nd_update_(cache, indices, updates):
    """Apply npu_scatter_nd_update_ semantics on CPU for flat BF16 slots."""
    for row, update in zip(indices, updates, strict=True):
        if int(row[0]) < 0:
            continue
        cache[row[0]] = update


def test_bf16_scatter_is_aclgraph_static_and_keeps_pad_slots():
    """PAD_SLOT_ID rows stay in-graph and must keep index -1.

    Evaluating ``torch.any(valid)`` as a Python bool calls LocalScalarDense
    and aborts ACLGraph capture (EE1016 / error 107027). Boolean-indexing the
    valid rows is also illegal: it emits a data-dependent Nonzero. Clamping
    ``-1`` to ``0`` overwrites a live physical slot.
    """
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(True))
        num_blocks, block_size, num_kv_heads, head_dim = 3, 4, 1, 2
        cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        sentinel = torch.full((num_kv_heads, head_dim), 999.0)
        cache[num_blocks - 1, block_size - 1] = sentinel
        slot_zero = cache[0, 0].clone()

        slot_mapping = torch.tensor([1, -1, -1], dtype=torch.int32)
        updates = torch.tensor([[[1.0, 2.0]], [[-7.0, -7.0]], [[-8.0, -8.0]]])
        captured: dict[str, torch.Tensor] = {}

        def fake_scatter(cache_t, indices, values):
            captured["indices"] = indices.detach().clone()
            captured["updates"] = values.detach().clone()
            _cpu_scatter_nd_update_(cache_t, indices, values)

        with (
            mock.patch(
                "vllm_ascend.attention.dsa_attn_kv_plan.torch_npu.npu_scatter_nd_update_", side_effect=fake_scatter
            ),
            mock.patch("torch.any", wraps=torch.any) as any_spy,
        ):
            plan.dsa_kv_compress_scatter(cache, updates, slot_mapping)

        any_spy.assert_not_called()
        assert captured["indices"].shape == (slot_mapping.shape[0], 1)
        assert captured["updates"].shape[0] == slot_mapping.shape[0]
        torch.testing.assert_close(
            captured["indices"],
            torch.tensor([[1], [-1], [-1]], dtype=torch.int64),
        )
        torch.testing.assert_close(cache[0, 1], updates[0])
        torch.testing.assert_close(cache[0, 0], slot_zero)
        torch.testing.assert_close(cache[num_blocks - 1, block_size - 1], sentinel)


def test_bf16_scatter_does_not_early_return_on_all_padded_slots():
    """An all-pad decode capture batch must still emit a static scatter."""
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(True))
        cache = torch.zeros(2, 2, 1, 2)
        slot_mapping = torch.full((4,), -1, dtype=torch.int32)
        updates = torch.ones(4, 1, 2)

        with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.torch_npu.npu_scatter_nd_update_") as scatter:
            plan.dsa_kv_compress_scatter(cache, updates, slot_mapping)
            scatter.assert_called_once()
            indices = scatter.call_args.args[1]
            assert indices.shape == (4, 1)
            assert bool((indices == -1).all())


def test_is_a5_bf16_kv_enabled_requires_vllm_config():
    with _on(AscendDeviceType.A5), pytest.raises(TypeError):
        is_a5_bf16_kv_enabled()


def test_only_explicit_bfloat16_selects_bf16_kv_on_a5():
    with _on(AscendDeviceType.A5):
        assert is_a5_bf16_kv_enabled(_cache_config("bfloat16"))
        assert not is_a5_bf16_kv_enabled(_cache_config())
        assert not is_a5_bf16_kv_enabled(_cache_config("fp8"))
        # The A5 spec path rewrites cache_dtype once FP8 KV is chosen.
        assert not is_a5_bf16_kv_enabled(_cache_config("float8_e4m3fn"))


def test_a5_bf16_kv_is_disabled_on_non_a5():
    with _on(AscendDeviceType.A3):
        assert not is_a5_bf16_kv_enabled(_cache_config("bfloat16"))


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "expected_dtype"),
    [
        (AscendDeviceType.A3, "auto", torch.bfloat16),
        (AscendDeviceType.A5, "bfloat16", torch.bfloat16),
        (AscendDeviceType.A5, "auto", torch.float8_e4m3fn),
    ],
)
def test_dsv4_attn_kv_dtype_preserves_device_modes(device_type, cache_dtype, expected_dtype):
    with _on(device_type):
        assert get_dsv4_attn_kv_dtype(_cache_config(cache_dtype)) == expected_dtype


def test_non_a5_pins_cache_dtype_to_the_model_dtype():
    with _on(AscendDeviceType.A3):
        for launch in ("auto", "bfloat16", "fp8"):
            assert resolve_dsv4_cache_dtype(launch, "bfloat16") == "bfloat16"


def test_a5_collapses_non_bfloat16_requests_to_auto():
    # "auto" resolves to the model dtype everywhere downstream, so it carries
    # the FP8 mode without changing any value upstream would have computed.
    with _on(AscendDeviceType.A5):
        assert resolve_dsv4_cache_dtype("bfloat16", "bfloat16") == "bfloat16"
        assert resolve_dsv4_cache_dtype("auto", "bfloat16") == "auto"
        assert resolve_dsv4_cache_dtype("fp8", "bfloat16") == "auto"


def test_a5_mode_survives_the_spec_path_rewrite():
    with _on(AscendDeviceType.A5):
        for launch in ("auto", "fp8"):
            pinned = resolve_dsv4_cache_dtype(launch, "bfloat16")
            assert not is_a5_bf16_kv_enabled(_cache_config(pinned))
            # layer.get_kv_cache_spec pins FP8 once it has picked the mode.
            assert not is_a5_bf16_kv_enabled(_cache_config("float8_e4m3fn"))

        pinned = resolve_dsv4_cache_dtype("bfloat16", "bfloat16")
        assert is_a5_bf16_kv_enabled(_cache_config(pinned))
