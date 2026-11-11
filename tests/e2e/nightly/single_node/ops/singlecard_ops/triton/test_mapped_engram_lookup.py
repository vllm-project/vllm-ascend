# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.engram_lookup import mapped_engram_gather_dequant


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens", [0, 1, 17, 2048])
@pytest.mark.parametrize("single_owner", [False, True])
def test_mapped_engram_lookup_dynamic_graph_ids_and_padding(dtype, tokens, single_owner):
    device = torch.device("npu:0")
    width, group, owner_rows, owners = 256, 32, 19, 3
    vocab_start, vocab_end = (owner_rows, owner_rows * 2) if single_owner else (3, owners * owner_rows - 1)
    weight_bytes = width * dtype.itemsize
    scale_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else torch.float32
    row_bytes = weight_bytes + (0 if dtype == torch.bfloat16 else width // group * scale_dtype.itemsize)
    storage = torch.zeros((owners, owner_rows + 2, row_bytes), dtype=torch.uint8, device=device)
    bases = torch.tensor([storage[i].data_ptr() + row_bytes for i in range(owners)], dtype=torch.int64, device=device)
    generator = torch.Generator().manual_seed(117)
    raw = torch.randint(-8, 8, (owners, owner_rows, width), generator=generator).to(dtype)
    storage[:, 1:-1, :weight_bytes].copy_(raw.view(torch.uint8).to(device))
    expected_rows = raw.float()
    if dtype != torch.bfloat16:
        if dtype == torch.float8_e4m3fn:
            scales = torch.randint(124, 131, (owners, owner_rows, width // group), generator=generator).to(torch.uint8)
            decoded = torch.pow(2.0, scales.float() - 127)
        else:
            scales = torch.rand((owners, owner_rows, width // group), generator=generator)
            decoded = scales
        storage[:, 1:-1, weight_bytes:].copy_(scales.view(torch.uint8).to(device))
        expected_rows *= decoded.repeat_interleave(group, dim=-1)
    expected_rows = expected_rows.reshape(-1, width).bfloat16()
    ids = torch.empty((tokens * 2, 6), dtype=torch.int64, device=device)[::2]
    out = torch.full((tokens, 4, width), 7, dtype=torch.bfloat16, device=device)

    def run():
        mapped_engram_gather_dequant(
            bases, ids, out, owner_rows, row_bytes, weight_bytes, 1, 3, vocab_start, vocab_end, dtype
        )

    def refill_and_check(graph=None):
        values = torch.randint(-2, owners * owner_rows + 2, ids.shape, generator=generator)
        if tokens:
            values[0, 1:4] = torch.tensor([owner_rows - 1, owner_rows, owners * owner_rows - 1])
        ids.copy_(values)
        run() if graph is None else graph.replay()
        torch.npu.synchronize()
        selected = values[:, 1:4]
        expected = expected_rows[selected.clamp(0, owners * owner_rows - 1)]
        expected[(selected < vocab_start) | (selected >= vocab_end)] = 0
        torch.testing.assert_close(out[:, :3].cpu(), expected, rtol=0, atol=0)
        assert torch.all(out[:, 3].cpu() == 7)

    refill_and_check()
    if tokens:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            run()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run()
        refill_and_check(graph)
        refill_and_check(graph)
