# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.selective_lim import (  # noqa: E402
    get_selective_lim_launcher,
)

BLOCK_SIZE = 128
HOST_BLOCKS = 64
K_BYTES, V_BYTES = 1024, 128
HOST_K, HOST_V, HOT_K, HOT_V = (1 << shift for shift in (36, 37, 38, 39))


def _reference(topk, table, lengths, requests, tails, keys, previous, hot):
    rows, width = topk.shape
    physical = torch.full((rows, width), -1, dtype=torch.int32)
    sources = torch.empty((2, rows, width), dtype=torch.int64)
    destinations = torch.empty_like(sources)
    sizes = torch.zeros((2, rows, width), dtype=torch.int32)
    updated = keys.clone()
    for row in range(rows):
        request = int(requests[row])
        resident = keys[row, :hot].tolist() if request and previous[row] == request else [-1] * hot
        mapping = {token: slot for slot, token in enumerate(resident) if token >= 0}
        source_rows = []
        used = set()
        for col, token in enumerate(topk[row].tolist()):
            source = -1
            if request and 0 <= token < lengths[row] and token // BLOCK_SIZE < table.shape[1]:
                page = int(table[row, token // BLOCK_SIZE])
                if 0 <= page < HOST_BLOCKS:
                    offset = token % BLOCK_SIZE
                    if (
                        tails[row, 0] >= 0
                        and token // BLOCK_SIZE == (lengths[row] - 1) // BLOCK_SIZE
                        and offset >= tails[row, 1]
                    ):
                        physical[row, col] = hot + offset
                    elif token != lengths[row] - 1:
                        source = page * BLOCK_SIZE + offset
                        if source in mapping:
                            physical[row, col] = mapping[source]
                            used.add(mapping[source])
            source_rows.append(source)
        free = iter(
            [i for i, token in enumerate(resident) if token < 0]
            + [i for i, token in enumerate(resident) if token >= 0 and i not in used]
        )
        for col, source in enumerate(source_rows):
            miss = source >= 0 and physical[row, col] < 0 and source not in mapping
            if source >= 0 and physical[row, col] < 0:
                if miss:
                    slot = next(free)
                    resident[slot] = source
                    mapping[source] = slot
                physical[row, col] = mapping[source]
            target = row * (hot + BLOCK_SIZE) + (int(physical[row, col]) if miss else 0)
            for component, (host, device, stride) in enumerate(((HOST_K, HOT_K, K_BYTES), (HOST_V, HOT_V, V_BYTES))):
                sources[component, row, col] = host + max(source, 0) * stride
                destinations[component, row, col] = device + target * stride
                sizes[component, row, col] = stride if miss else 0
        updated[row, :hot] = torch.tensor(resident)
        updated[row, hot] = -1
    return physical, sources, destinations, sizes, updated, requests.clone()


@pytest.mark.parametrize("rows,width,hot", [(1, 2048, 2048), (4, 2048, 2048), (4, 7, 16), (9, 31, 128)])
def test_precomputed_topk_parallel_rows_and_graph_replay(rows, width, hot):
    if not os.environ.get("VLLM_ASCEND_SELECTIVE_LIM_LIBRARY"):
        pytest.skip("Build and select the standalone precomputed TopK LIM library")
    torch.npu.set_device(0)
    launch = get_selective_lim_launcher()
    device = torch.device("npu:0")
    topk = torch.empty((rows, width), dtype=torch.int64, device=device)
    table = ((torch.arange(32).expand(rows, -1) * 3 + torch.arange(rows)[:, None]) % HOST_BLOCKS).int().to(device)
    lengths = (2305 + torch.arange(rows) * 17).int().to(device)
    requests = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    tails = torch.stack((torch.zeros(rows, dtype=torch.int32), ((lengths.cpu() - 1) % BLOCK_SIZE) // 2), dim=1).to(
        device
    )
    keys = torch.full((rows, hot + 1), -1, dtype=torch.int64, device=device)
    previous = torch.full((rows,), -1, dtype=torch.int64, device=device)
    physical = torch.empty((rows, width), dtype=torch.int32, device=device)
    sources = torch.empty((2, rows, width), dtype=torch.int64, device=device)
    destinations = torch.empty_like(sources)
    sizes = torch.empty((2, rows, width), dtype=torch.int32, device=device)
    arguments = (topk, table, lengths, requests, tails, keys, previous, physical, sources, destinations, sizes)

    def invoke():
        result = launch(
            rows,
            torch.npu.current_stream().npu_stream,
            *(tensor.data_ptr() for tensor in arguments),
            rows,
            width,
            table.shape[1],
            HOST_BLOCKS,
            BLOCK_SIZE,
            hot,
            hot + BLOCK_SIZE,
            K_BYTES,
            V_BYTES,
            HOST_K,
            HOST_V,
            HOT_K,
            HOT_V,
        )
        assert result == 0

    def update(step):
        ids = (torch.arange(width).expand(rows, -1) + step * 257) % (lengths.cpu()[:, None] - 1)
        ids[:, -1] = lengths.cpu() - 1
        if step:
            ids[:, 1] = ids[:, 0]
            ids[:, -2] = -1
        topk.copy_(ids)
        identities = torch.arange(1, rows + 1, dtype=torch.int64) + (step // 3) * 100
        if rows > 1:
            identities[-1] = 0 if step % 2 else identities[-1]
        requests.copy_(identities)

    def expected():
        return _reference(*(tensor.cpu() for tensor in (topk, table, lengths, requests, tails, keys, previous)), hot)

    def check(reference):
        for actual, target in zip((physical, sources, destinations, sizes, keys, previous), reference):
            assert torch.equal(actual.cpu(), target)

    update(0)
    reference = expected()
    invoke()
    torch.npu.synchronize()
    check(reference)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()
    torch.npu.synchronize()
    for step in range(8):
        update(step)
        reference = expected()
        graph.replay()
        torch.npu.synchronize()
        check(reference)
    reference = expected()
    for _ in range(50):
        graph.replay()
    torch.npu.synchronize()
    check(reference)
