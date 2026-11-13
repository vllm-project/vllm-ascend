# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from typing import Any

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.msa.msa_m3_index_postprocess import index_decode_topk_fused


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
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
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("width", [0, 32])
def test_zero_logical_blocks(width: int) -> None:
    score = torch.empty((4, 8, width), dtype=torch.float32, device="npu")
    context = torch.zeros(8, dtype=torch.int32, device="npu")
    indices, counts = index_decode_topk_fused(score, context, 1, 16, 0)
    assert torch.all(indices == -1)
    assert torch.all(counts == 0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
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
    kwargs: dict[str, Any] = dict(decode_query_len=1, topk=16, logical_block_count=32, init_blocks=0, local_blocks=1)
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        index_decode_topk_fused(score, context, **kwargs)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
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
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("topk,width", [(1, 0), (3, 7), (16, 513), (32, 2055)])
def test_sorted_decode_indices_match_normalized_selection(query_len: int, strided: bool, topk: int, width: int) -> None:
    context = torch.tensor([-query_len, 0, 127, 128, 255, 65535], dtype=torch.int64, device="npu")
    tokens = context.numel() * query_len
    score = torch.arange(width, dtype=torch.float32, device="npu").expand(1, tokens, width).contiguous()
    if strided:
        storage = torch.empty((1, tokens, width * 2), dtype=score.dtype, device=score.device)
        storage[..., ::2] = score
        score = storage[..., ::2]
        context_storage = torch.empty(context.numel() * 2, dtype=context.dtype, device=context.device)
        context_storage[::2] = context
        context = context_storage[::2]
    unsorted, expected_counts = index_decode_topk_fused(score, context, query_len, topk, width)
    keys = torch.where(unsorted >= 0, unsorted.to(torch.float32), float("inf"))
    ordered = keys.sort(dim=-1).values
    expected = torch.where(ordered == float("inf"), -1, ordered).to(torch.int32)
    actual, counts = index_decode_topk_fused(score, context, query_len, topk, width, sort_indices=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    assert actual.is_contiguous() and actual.dtype == torch.int32


@pytest.mark.parametrize(
    "requests,query_len,width,context_tokens,strided",
    [
        (5, 8, 1024, 131072, False),
        (10, 8, 512, 65536, True),
        (11, 4, 512, 65536, True),
        (1, 1, 2055, 263000, False),
        (2, 9, 33, 4095, False),
        (2, 16, 17, 2047, True),
    ],
)
@pytest.mark.parametrize("context_dtype", [torch.int32, torch.int64])
def test_sorted_decode_model_shapes_and_workspace_contract(
    requests: int, query_len: int, width: int, context_tokens: int, strided: bool, context_dtype: torch.dtype
) -> None:
    # Actual C40/C80 DP8 K7 and historical C88 DP8 K3 geometry, plus maximum-context and
    # supported-width boundary cases. The oracle never calls the fused path.
    heads, topk = 1, 16
    tokens = requests * query_len
    source = torch.arange(width, dtype=torch.float32).expand(heads, tokens, width).contiguous()
    context = torch.full((requests,), context_tokens, dtype=context_dtype)
    if requests > 1:
        context[-1] = -query_len  # FULL graph padding: no visible blocks.
    expected = torch.full((heads, tokens, topk), -1, dtype=torch.int32)
    expected_counts = torch.zeros((heads, tokens), dtype=torch.int32)
    prepared = source.clone()
    for row in range(tokens):
        visible_blocks = max(0, (int(context[row // query_len]) + row % query_len + 128) // 128)
        visible = min(width, visible_blocks)
        prepared[:, row, visible:] = -torch.inf
        if visible:
            prepared[:, row, :1] = 1.0e30
            prepared[:, row, max(0, visible_blocks - 2) : visible] = 1.0e29
        count = min(topk, visible)
        if count:
            selected = prepared[0, row, :visible].topk(count).indices.sort().values.to(torch.int32)
            expected[0, row, :count] = selected
        expected_counts[:, row] = count
    score = source.npu()
    if strided:
        backing = torch.full((heads, tokens, width * 2), -99.0, device="npu")
        score = backing[..., ::2]
        score.copy_(source)
    lengths = context.npu()
    score_metadata = (score.dtype, score.shape, score.stride(), score.data_ptr())
    lengths_metadata = (lengths.dtype, lengths.shape, lengths.stride(), lengths.data_ptr())
    actual, counts = index_decode_topk_fused(score, lengths, query_len, topk, width, 1, 2, sort_indices=True)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    assert (score.dtype, score.shape, score.stride(), score.data_ptr()) == score_metadata
    assert (lengths.dtype, lengths.shape, lengths.stride(), lengths.data_ptr()) == lengths_metadata
    torch.testing.assert_close(lengths.cpu(), context, rtol=0, atol=0)
    torch.testing.assert_close(score.cpu(), source if strided else prepared, rtol=0, atol=0)
    if strided:
        assert torch.all(backing[..., 1::2] == -99.0)


@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 0), (1, 2)])
def test_sorted_topk_unaligned_multihead_repeated(init_blocks: int, local_blocks: int, tmp_path) -> None:
    # MiniMax M3/M3.1 select 16 blocks. Exercise an unaligned score width with
    # repeated allocations and compare both paths with an independent CPU oracle.
    import_module("vllm_ascend.ops")
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _sort_valid_gbsa_prefix

    heads, requests, query_len, topk, width = 4, 2, 8, 16, 33
    generator = torch.Generator().manual_seed(18426)
    lengths = torch.full((requests,), (width - 1) * 128, dtype=torch.int32, device="npu")
    expected_counts = torch.full((heads, requests * query_len), topk, dtype=torch.int32)
    for trial in range(32):
        source = torch.randn((heads, requests * query_len, width), generator=generator)
        prepared = source.clone()
        prepared[..., :init_blocks] = 1.0e30
        if local_blocks:
            prepared[..., width - local_blocks :] = 1.0e29
        expected = prepared.topk(topk, dim=-1).indices.sort(dim=-1).values.to(torch.int32)
        legacy_score, fused_score = source.npu(), source.npu()
        raw, legacy_counts = index_decode_topk_fused(
            legacy_score, lengths, query_len, topk, width, init_blocks, local_blocks
        )
        legacy = _sort_valid_gbsa_prefix(raw, legacy_counts)
        fused, fused_counts = index_decode_topk_fused(
            fused_score, lengths, query_len, topk, width, init_blocks, local_blocks, sort_indices=True
        )
        actual = {"legacy": legacy.cpu(), "fused": fused.cpu()}
        try:
            for name, ids in actual.items():
                torch.testing.assert_close(ids, expected, rtol=0, atol=0, msg=f"{name}, trial={trial}")
            torch.testing.assert_close(legacy_counts.cpu(), expected_counts, rtol=0, atol=0)
            torch.testing.assert_close(fused_counts.cpu(), expected_counts, rtol=0, atol=0)
            torch.testing.assert_close(legacy_score.cpu(), prepared, rtol=0, atol=0)
            torch.testing.assert_close(fused_score.cpu(), prepared, rtol=0, atol=0)
        except AssertionError:
            torch.save(
                dict(
                    source=source,
                    prepared=prepared,
                    expected=expected,
                    raw=raw.cpu(),
                    legacy_counts=legacy_counts.cpu(),
                    fused_counts=fused_counts.cpu(),
                    **actual,
                ),
                tmp_path / f"unaligned-topk-trial-{trial}.pt",
            )
            raise
