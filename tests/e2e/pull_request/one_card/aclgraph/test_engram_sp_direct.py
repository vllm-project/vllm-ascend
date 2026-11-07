# SPDX-License-Identifier: Apache-2.0
"""Portable SP lookup/replay regression; global chunks use small HBM backing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding


def chunk_view(source, device):
    chunks, keepalive = [], []
    for start in range(0, len(source), 8):
        chunk = source[start : start + 8].to(device)
        chunks.append(chunk)
        # Force gaps between chunks so ignoring the pointer table cannot pass.
        keepalive.extend((chunk, torch.zeros(4096, dtype=torch.int32, device=device)))
    return SimpleNamespace(
        tensor=torch.empty(source.shape, dtype=source.dtype, device="meta"),
        ptrs=torch.tensor([chunk.data_ptr() for chunk in chunks], dtype=torch.int64, device=device),
        global_rows=True,
        chunk_rows=8,
        keepalive=keepalive,
    )


@pytest.mark.parametrize("heads", [4, 24])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("graph_replay", [False, True])
def test_token_owned_lookup_with_disjoint_chunks(heads, tp, graph_replay):
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    width, tokens, bucket = 256, 7, 10
    boundaries = [0]
    for size in [17, 19, 17, 20] * (heads // 4):
        boundaries.append(boundaries[-1] + size)
    rows = boundaries[-1]
    codes = torch.arange(rows * width, device="cpu").remainder(255).sub(127).to(torch.int8).reshape(rows, width)
    scales = torch.full((rows, 8), 0.125, device="cpu")
    table = SimpleNamespace(
        dp_size=1,
        tp_size=tp,
        n_hash_cols=heads,
        dim=width,
        num_embeddings=rows,
        part_n_hash_cols=heads // tp,
        _codes_uva=chunk_view(codes, device),
        _scales_uva=chunk_view(scales, device),
        _sp_head_bounds=torch.tensor(
            [(boundaries[r * heads // tp], boundaries[(r + 1) * heads // tp]) for r in range(tp)],
            dtype=torch.int64,
            device=device,
        ),
    )
    raw = torch.arange(tokens * heads, device="cpu").reshape(tokens, heads)
    for h in range(heads):
        raw[:, h] = raw[:, h].remainder(boundaries[h + 1] - boundaries[h]) + boundaries[h]
    raw[0, 0], raw[1, 1], raw[2, 2] = rows - 1, -1, rows
    ids = raw.to(device)
    slot = (bucket + tp - 1) // tp

    def reference(values, rank):
        pieces = []
        for owner in range(tp):
            lower, upper = boundaries[owner * heads // tp], boundaries[(owner + 1) * heads // tp]
            selected = values[:, owner * heads // tp : (owner + 1) * heads // tp]
            valid = (selected >= lower) & (selected < upper)
            result = (codes[selected.clamp(0, rows - 1)].float() * 0.125).bfloat16()
            pieces.append(result.masked_fill(~valid[..., None], 0))
        full = F.pad(torch.cat(pieces, dim=1).flatten(1), (0, 0, 0, slot * tp - tokens))
        return full[rank * slot : (rank + 1) * slot]

    for rank in range(tp):
        output = torch.full((slot + 1, heads * width), 99, dtype=torch.bfloat16, device=device)
        actual = AscendParallelEngramEmbedding.lookup_token_shard(table, ids, output, bucket, rank)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference(raw, rank), rtol=0, atol=0)
        assert torch.all(output[slot:].cpu() == 99)
        if not graph_replay:
            continue
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = AscendParallelEngramEmbedding.lookup_token_shard(table, ids, output, bucket, rank)
        changed = raw.roll(1, 0)
        ids.copy_(changed.to(device))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference(changed, rank), rtol=0, atol=0)
        assert torch.all(output[slot:].cpu() == 99)
        graph.reset()
        ids.copy_(raw.to(device))
