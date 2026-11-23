# SPDX-License-Identifier: Apache-2.0
"""Paged NOVA-KV decode, masking and graph replay against dense reconstruction."""

import pytest
import torch

pytest.importorskip("tilelang")
pytest.importorskip("torch_npu")

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.tilelang.nova_kv_decode import NovaDecodeKernel

pytestmark = pytest.mark.skipif(not torch.npu.is_available() or not is_950(), reason="requires Ascend 950")


def _reference(query, mean, views, books, table, starts, ends):
    outputs, lses = [], []
    key, value, scales = (tensor.cpu() for tensor in views)
    books = books.cpu().float()
    for request in range(query.shape[0]):
        positions = torch.arange(int(starts[request]), int(ends[request]))
        if positions.numel() == 0:
            outputs.append(torch.zeros_like(query[request], device="cpu"))
            lses.append(torch.full(query.shape[1:3], -torch.inf))
            continue
        pages = table[request].cpu()[positions // key.shape[-1]].long()
        offsets = positions % key.shape[-1]
        heads_out, heads_lse = [], []
        for head in range(query.shape[1]):
            codes = key[pages, head, :, offsets].long()
            key_scale = scales[pages, head, 0, offsets].float()
            dense_key = books[head, torch.arange(64)[None], codes].flatten(1)
            dense_key = (dense_key * key_scale[:, None]).half().float()
            packed = value[pages, head, :, offsets].int()
            symbols = torch.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1).flatten(1)
            dense_value = symbols * scales[pages, head, 1, offsets].float()[:, None]
            dense_value += scales[pages, head, 2, offsets].float()[:, None]
            dense_value = dense_value.half().float()
            score = (query[request, head].cpu().half().float() @ dense_key.T + mean[request, head].cpu()[:, None]) / 16
            heads_out.append(score.softmax(-1) @ dense_value)
            heads_lse.append(score.logsumexp(-1))
        outputs.append(torch.stack(heads_out))
        lses.append(torch.stack(heads_lse))
    return torch.stack(outputs), torch.stack(lses)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_paged_attention_and_dynamic_graph_replay(batch, padded):
    torch.manual_seed(9413)
    device = torch.device("npu:0")
    pages, heads, page_tokens, groups = batch * 3, 2, 256, 6
    page_stride = heads * 64 * page_tokens + (128 if padded else 0)
    key = torch.empty_strided(
        (pages, heads, 64, page_tokens),
        (page_stride, 64 * page_tokens, page_tokens, 1),
        dtype=torch.uint8,
        device=device,
    )
    value = torch.empty_strided(key.shape, key.stride(), dtype=torch.uint8, device=device)
    scale_stride = heads * 3 * page_tokens + (64 if padded else 0)
    scales = torch.empty_strided(
        (pages, heads, 3, page_tokens),
        (scale_stride, 3 * page_tokens, page_tokens, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    key.copy_(torch.randint(0, 256, key.shape, dtype=torch.uint8, device=device))
    value.copy_(torch.randint(0, 256, value.shape, dtype=torch.uint8, device=device))
    scales.copy_(torch.rand(scales.shape, device=device).mul_(0.2).add_(0.05))
    scales[:, :, 2].sub_(0.2)
    views = key, value, scales
    books = torch.randn((heads, 64, 256, 4), device=device).half()
    table = torch.randperm(pages, device=device).reshape(batch, 3).int()
    # Poison the unused tail: masked zero probabilities must not multiply NaN V.
    for page in table[:, 2].cpu().tolist():
        scales[page, :, :, 188:] = torch.nan
    query = torch.randn((batch, heads, groups, 256), device=device).mul_(0.2)
    mean = torch.randn((batch, heads, groups), device=device).mul_(0.1)
    starts = torch.full((batch,), 64, dtype=torch.int32, device=device)
    ends = torch.full_like(starts, 700)
    work = NovaDecodeKernel(768, heads, groups, views, table, device, batch=batch, split_tokens=512)
    work(query, mean, views, books, table, starts, ends)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, lse = work(query, mean, views, books, table, starts, ends)
    for end in (700, 513, 512, 257, 65, 64):
        query.normal_(0, 0.3)
        ends.fill_(end)
        if batch == 4:
            ends[-1] = 64
        graph.replay()
        torch.npu.synchronize()
        expected, expected_lse = _reference(query, mean, views, books, table, starts.cpu(), ends.cpu())
        torch.testing.assert_close(output.cpu(), expected, atol=3e-4, rtol=5e-3)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5)
