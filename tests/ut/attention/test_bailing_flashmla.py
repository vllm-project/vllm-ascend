# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import bailing_flashmla
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize(
    ("architecture", "heads", "device_type", "has_ops", "enable_kv_nz", "expected"),
    [
        ("BailingMoeV3ForCausalLM", 4, AscendDeviceType.A3, True, False, True),
        ("BailingMoeV3VLForConditionalGeneration", 4, AscendDeviceType.A3, True, False, True),
        ("BailingMoeV3MTPModel", 4, AscendDeviceType.A3, True, False, False),
        ("BailingMoeV2_5ForCausalLM", 4, AscendDeviceType.A3, True, False, False),
        ("BailingMoeV3ForCausalLM", 8, AscendDeviceType.A3, True, False, True),
        ("BailingMoeV3ForCausalLM", 16, AscendDeviceType.A3, True, False, False),
        ("BailingMoeV3ForCausalLM", 4, AscendDeviceType.A2, True, False, False),
        ("BailingMoeV3ForCausalLM", 4, AscendDeviceType.A5, True, False, False),
        ("BailingMoeV3ForCausalLM", 4, AscendDeviceType.A3, False, False, False),
        ("BailingMoeV3ForCausalLM", 4, AscendDeviceType.A3, True, True, False),
    ],
)
def test_bailing_flashmla_selection(monkeypatch, architecture, heads, device_type, has_ops, enable_kv_nz, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
            get_num_attention_heads=lambda _: heads,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    monkeypatch.setattr(bailing_flashmla, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(bailing_flashmla, "has_bailing_flashmla_ops", lambda: has_ops)
    assert bailing_flashmla.supports_bailing_flashmla(config, enable_kv_nz=enable_kv_nz) is expected


@pytest.mark.parametrize(("query_ends", "query_width"), [([1, 2, 3], 1), ([4, 8], 4)])
def test_uniform_decode_needs_no_gather(query_ends, query_width):
    batch = bailing_flashmla.BailingFlashMLABatch.from_query_ends(query_ends, query_ends[-1], torch.device("cpu"))
    assert batch is not None
    assert batch.query_width == query_width
    assert batch.gather_queries is None
    assert batch.gather_outputs is None


@pytest.mark.parametrize("query_ends", [[], [0, 4], [1, 3], [1, 1, 4], [5, 4]])
def test_invalid_query_boundaries_do_not_use_flashmla(query_ends):
    assert bailing_flashmla.BailingFlashMLABatch.from_query_ends(query_ends, 4, torch.device("cpu")) is None


@pytest.mark.parametrize("heads", (4, 8))
def test_ragged_mtp_right_aligns_queries_and_reuses_plan(monkeypatch, heads):
    plan = object()
    prepare_decode = Mock(return_value=plan)

    def decode(query, cache, actual_plan, *, layout_out):
        assert actual_plan is plan
        assert layout_out == "NTD"
        assert cache[0] is latent_cache and cache[1] is rope_cache
        assert query[0][:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6]
        assert query[1][:, 0].tolist() == [10, 10, 10, 10, 11, 11, 11, 12, 13, 14, 15, 16]
        return torch.arange(12).reshape(1, 12, 1)

    flash_mla_decode = Mock(side_effect=decode)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.flash_mla_a3",
        SimpleNamespace(prepare_decode=prepare_decode, flash_mla_decode=flash_mla_decode),
    )
    batch = bailing_flashmla.BailingFlashMLABatch.from_query_ends([1, 3, 7], 7, torch.device("cpu"))
    assert batch is not None
    assert batch.query_width == 4
    batch.device_lengths = Mock(spec=torch.Tensor)
    batch.device_lengths.tolist.return_value = [128, 253, 300]

    query = tuple(torch.arange(offset, offset + 7)[:, None].expand(-1, heads) for offset in (0, 10))
    latent_cache = torch.empty(14, 128, 512)[::2]
    rope_cache = torch.empty(14, 128, 64)[::2]
    block_table = torch.zeros(3, 3, dtype=torch.int32)
    for _ in range(2):
        result = batch.attention(
            query, (latent_cache, rope_cache), block_table, [129, 254, 301], scale=0.125, causal=True, attn_mask=None
        )
        assert result.flatten().tolist() == [3, 6, 7, 8, 9, 10, 11]

    assert flash_mla_decode.call_count == 2
    batch.device_lengths.tolist.assert_called_once_with()
    prepare_decode.assert_called_once_with(
        block_table,
        [128, 253, 300],
        4,
        7,
        causal=True,
        softmax_scale=0.125,
        attn_mask=None,
        split_cache=True,
        validate_page_table=False,
        cache_page_strides=(latent_cache.stride(0), rope_cache.stride(0)),
        num_query_heads=heads,
    )
