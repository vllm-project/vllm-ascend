# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.msa.msa_m3_index_postprocess import index_decode_topk_fused


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("shape", [(0, 0, 0), (0, 3, 32), (4, 0, 32), (4, 0, 0)])
def test_empty_shape(shape: tuple[int, int, int]) -> None:
    score = torch.empty(shape, dtype=torch.float32, device="npu")
    context = torch.zeros(shape[1], dtype=torch.int32, device="npu")
    indices, counts = index_decode_topk_fused(score, context, 1, 16, shape[-1])
    assert indices.shape == (*shape[:2], 16)
    assert counts.shape == shape[:2]
    assert indices.dtype == counts.dtype == torch.int32
    assert indices.device == counts.device == score.device


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("width", [0, 32])
def test_zero_logical_blocks(width: int) -> None:
    score = torch.empty((4, 8, width), dtype=torch.float32, device="npu")
    context = torch.zeros(8, dtype=torch.int32, device="npu")
    indices, counts = index_decode_topk_fused(score, context, 1, 16, 0)
    assert torch.all(indices == -1)
    assert torch.all(counts == 0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize(
    "change,message",
    [
        ({"score_shape": (2, 32)}, "float32 tensor"),
        ({"score_dtype": torch.float16}, "float32 tensor"),
        ({"context_shape": (1, 2)}, "context length"),
        ({"context_dtype": torch.float32}, "context length"),
        ({"context_device": "cpu"}, "same device"),
        ({"decode_query_len": 0}, "Decode tokens"),
        ({"decode_query_len": -1}, "Decode tokens"),
        ({"decode_query_len": 2}, "Decode tokens"),
        ({"context_shape": (0,)}, "Decode tokens"),
        ({"logical_block_count": -1}, "logical block count"),
        ({"logical_block_count": 33}, "logical block count"),
        ({"topk": 0}, "logical block count"),
        ({"topk": -1}, "logical block count"),
        ({"init_blocks": -1}, "non-negative"),
        ({"local_blocks": -1}, "non-negative"),
    ],
)
def test_invalid_inputs(change: dict, message: str) -> None:
    options = dict(change)
    score = torch.empty(
        options.pop("score_shape", (4, 2, 32)),
        dtype=options.pop("score_dtype", torch.float32),
        device="npu",
    )
    context = torch.zeros(
        options.pop("context_shape", (2,)),
        dtype=options.pop("context_dtype", torch.int32),
        device=options.pop("context_device", "npu"),
    )
    kwargs = dict(decode_query_len=1, topk=16, logical_block_count=32, init_blocks=0, local_blocks=1)
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        index_decode_topk_fused(score, context, **kwargs)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("context_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("topk,width", [(1, 1), (16, 7), (16, 33), (32, 16)])
def test_block_boundaries(context_dtype: torch.dtype, query_len: int, topk: int, width: int) -> None:
    context = torch.tensor([-query_len, 0, 127, 128, 255, 256, 4095], dtype=context_dtype)
    tokens = context.numel() * query_len
    reference = torch.arange(width, dtype=torch.float32).expand(4, tokens, width).contiguous()
    indices, counts = index_decode_topk_fused(reference.to("npu"), context.to("npu"), query_len, topk, width)
    indices, counts = indices.cpu(), counts.cpu()
    for query in range(tokens):
        visible = max(0, min((int(context[query // query_len]) + query % query_len) // 128 + 1, width))
        count = min(topk, visible)
        assert torch.all(counts[:, query] == count)
        expected = torch.arange(visible - 1, visible - count - 1, -1, dtype=torch.int32)
        for head in range(4):
            assert torch.equal(indices[head, query, :count], expected)
            assert torch.all(indices[head, query, count:] == -1)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 0), (1, 2)])
def test_decode_topk_varying_runtime_parameters(strided: bool, init_blocks: int, local_blocks: int) -> None:
    # Keep model configuration fixed while varying batch shape, widths and MTP.
    heads, topk, block_size = 4, 16, 128
    for requests, query_len, logical_blocks, width in [(1, 1, 3, 4), (3, 4, 17, 32), (2, 1, 8, 16)]:
        tokens = requests * query_len
        reference = torch.arange(heads * tokens * width, dtype=torch.float32).view(heads, tokens, width)
        context = torch.arange(requests, dtype=torch.int32) * block_size + block_size - 1
        score = reference.to("npu")
        context_npu = context.to("npu")
        if strided:
            score_storage = torch.empty((heads, tokens, width * 2), dtype=torch.float32, device="npu")
            score = score_storage[..., ::2]
            score.copy_(reference)
            context_storage = torch.empty(requests * 2, dtype=torch.int32, device="npu")
            context_npu = context_storage[::2]
            context_npu.copy_(context)

        indices, counts = index_decode_topk_fused(
            score, context_npu, query_len, topk, logical_blocks, init_blocks, local_blocks
        )
        indices, counts = indices.cpu(), counts.cpu()
        assert indices.dtype == torch.int32
        assert counts.dtype == torch.int32
        for query in range(tokens):
            visible = min((int(context[query // query_len]) + query % query_len) // block_size + 1, logical_blocks)
            expected_scores = reference[:, query, :visible].clone()
            expected_scores[:, :init_blocks] = 1.0e30
            if local_blocks:
                expected_scores[:, max(visible - local_blocks, 0) :] = 1.0e29
            valid = indices[:, query] >= 0
            assert torch.all(counts[:, query] == min(visible, topk))
            assert torch.equal(valid.sum(-1).to(torch.int32), counts[:, query])
            assert torch.all(indices[:, query][~valid] == -1)
            for head in range(heads):
                selected = indices[head, query][valid[head]].long()
                assert torch.all(selected < visible)
                assert selected.unique().numel() == selected.numel()
                # Compare scores to allow any ordering among forced-score ties.
                actual = expected_scores[head, selected].sort(descending=True).values
                expected = expected_scores[head].topk(min(visible, topk)).values
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
