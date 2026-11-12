# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3/A5 QSFA native heads, statistics, and graph replay (requires PR #54)."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import get_device_adaptor
from vllm_ascend.device.hardware import AscendDeviceType

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires A2/A3/A5 and the custom QSFA kernel from PR #54.",
)


def _make_case(heads, rope_dim, query_tokens, count):
    generator = torch.Generator().manual_seed(20261029)
    page_size, nope_dim, tile_size = 128, 512, 128
    length = 4096 if count == 2048 else 128
    query = torch.randn(query_tokens, heads, nope_dim + rope_dim, generator=generator).to(torch.bfloat16)
    kv_dtype = torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8
    key_nope = (torch.randn(length, 1, nope_dim, generator=generator) * 2).to(kv_dtype)
    key_rope = torch.randn(length, 1, rope_dim, generator=generator).to(torch.bfloat16)
    scales = torch.rand(length, 1, nope_dim // tile_size, generator=generator) * 0.1 + 0.05
    cache = (
        torch.cat((key_nope.view(torch.uint8), key_rope.view(torch.uint8), scales.view(torch.uint8)), -1)
        .reshape(length // page_size, page_size, 1, -1)
        .view(kv_dtype)
    )
    indices = torch.full((query_tokens, 1, 2048), -1, dtype=torch.int32)
    # All selected tokens are visible to every query in this chunk.
    selected = torch.randperm(length - query_tokens + 1, generator=generator)[:count]
    indices[:, 0, :count] = selected.to(torch.int32)
    scale = (nope_dim + rope_dim) ** -0.5
    kv_npu = cache.npu()
    inputs = dict(
        query=query.npu(),
        key=kv_npu,
        value=kv_npu,
        sparse_indices=indices.npu(),
        scale_value=scale,
        sparse_block_size=1,
        block_table=torch.arange(length // page_size, dtype=torch.int32).view(1, -1).npu(),
        actual_seq_lengths_query=torch.tensor([query_tokens], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([length], dtype=torch.int32).npu(),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=tile_size,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=rope_dim,
    )
    dequant = (key_nope.float() * scales.repeat_interleave(tile_size, -1)).to(torch.bfloat16)
    key = torch.cat((dequant, key_rope), -1)[selected, 0].float()
    return inputs, key, dequant[selected, 0].float()


def _check_reference(result, query, key, value, scale, return_lse, *, empty_sparse_length=None):
    output, maximum, total = (tensor.cpu().float() for tensor in result)
    assert torch.isfinite(output).all()
    if value.shape[0] == 0:
        assert torch.count_nonzero(output) == 0
        if return_lse:
            if get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3):
                # Arch22 preserves the masked logits' raw MAX/SUM, unlike A5.
                assert empty_sparse_length is not None
                torch.testing.assert_close(maximum, torch.full_like(maximum, -2e38), rtol=0, atol=0)
                torch.testing.assert_close(total, torch.full_like(total, empty_sparse_length), rtol=0, atol=0)
            else:
                assert torch.count_nonzero(total) == 0
                assert torch.isneginf(maximum + total.log()).all()
        return
    scores = query.float() @ key.T * scale
    expected = scores.softmax(-1) @ value
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
    if return_lse:
        assert maximum.shape == total.shape == (1, query.shape[0], query.shape[1])
        torch.testing.assert_close(maximum + total.log(), scores.logsumexp(-1).unsqueeze(0), atol=1e-5, rtol=1e-5)


def _run_adaptor(inputs, return_lse, *, packed=False):
    rope_dim = inputs["rope_head_dim"]
    query = inputs["query"]
    nope, rope = query.split((512, rope_dim), dim=-1)
    # Native heads and the three-result contract require Kimi's custom QSFA path.
    return get_device_adaptor()._execute_kv_quant_sparse_flash_attention_vllm(
        SimpleNamespace(scale=inputs["scale_value"], qk_rope_head_dim=rope_dim, is_kimi_k31=True),
        nope,
        rope,
        inputs["key"],
        inputs["block_table"],
        inputs["sparse_indices"],
        inputs["actual_seq_lengths_query"],
        inputs["actual_seq_lengths_kv"],
        sparse_mode=inputs["sparse_mode"],
        return_lse=return_lse,
        packed_query=query if packed else None,
    )


@pytest.mark.parametrize("heads", [12, 24, 65, 96, 97, 128])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("query_tokens,count,sparse_mode", [(1, 0, 0), (1, 32, 3), (4, 2048, 0)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_native_heads(heads, rope_dim, query_tokens, count, sparse_mode, return_lse):
    inputs, key, value = _make_case(heads, rope_dim, query_tokens, count)
    inputs["sparse_mode"] = sparse_mode
    result = _run_adaptor(inputs, return_lse)
    _check_reference(
        result,
        inputs["query"].cpu(),
        key,
        value,
        inputs["scale_value"],
        return_lse,
        empty_sparse_length=min(inputs["sparse_indices"].shape[-1], inputs["key"].shape[0] * inputs["key"].shape[1]),
    )


@pytest.mark.parametrize("heads", [12, 24, 65, 96, 97, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_mixed_empty_and_nonempty_rows(heads, return_lse):
    inputs, key, value = _make_case(heads, 64, 4, 2048)
    counts = (0, 32, 256, 2048)
    indices = inputs["sparse_indices"].cpu()
    for row, count in enumerate(counts):
        indices[row, :, count:] = -1
    inputs["sparse_indices"].copy_(indices)
    inputs["sparse_mode"] = 0
    output, maximum, total = _run_adaptor(inputs, return_lse, packed=True)
    for row, count in enumerate(counts):
        result = (
            output[row : row + 1],
            maximum[:, row : row + 1] if return_lse else maximum,
            total[:, row : row + 1] if return_lse else total,
        )
        _check_reference(
            result,
            inputs["query"][row : row + 1].cpu(),
            key[:count],
            value[:count],
            inputs["scale_value"],
            return_lse,
            empty_sparse_length=min(
                inputs["sparse_indices"].shape[-1], inputs["key"].shape[0] * inputs["key"].shape[1]
            ),
        )


@pytest.mark.parametrize("heads", [12, 24, 65, 96, 97, 128])
@pytest.mark.parametrize("rope_dim", [0, 64])
def test_packed_query_changed_input_graph_replay(heads, rope_dim):
    inputs, key, value = _make_case(heads, rope_dim, 4, 32)
    inputs["sparse_mode"] = 0
    storage = inputs["query"].transpose(0, 1).contiguous()
    inputs["query"] = storage.transpose(0, 1)
    for _ in range(3):
        _run_adaptor(inputs, True, packed=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = _run_adaptor(inputs, True, packed=True)
    changed_query = inputs["query"].cpu().neg()
    inputs["query"].copy_(changed_query)
    graph.replay()
    torch.npu.synchronize()
    _check_reference(result, changed_query, key, value, inputs["scale_value"], True)
