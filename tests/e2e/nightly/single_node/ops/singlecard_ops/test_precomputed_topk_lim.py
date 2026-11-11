# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

BLOCK_SIZE = 128
HOST_BLOCKS = 64
K_BYTES, V_BYTES = 1024, 128
HOST_K, HOST_V, HOT_K, HOT_V = (1 << shift for shift in (36, 37, 38, 39))


def _reference(topk, table, lengths, requests, tails, keys, previous, hot, bases):
    rows, width = topk.shape
    host_k, host_v, hot_k, hot_v = bases
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
            for component, (host, device, stride) in enumerate(((host_k, hot_k, K_BYTES), (host_v, hot_v, V_BYTES))):
                sources[component, row, col] = host + max(source, 0) * stride
                destinations[component, row, col] = device + target * stride
                sizes[component, row, col] = stride if miss else 0
        updated[row, :hot] = torch.tensor(resident)
        updated[row, hot] = -1
    return physical, sources, destinations, sizes, updated, requests.clone()


@pytest.mark.parametrize(
    "layer_count,rows,width,hot",
    [(1, 1, 2048, 2048), (1, 4, 2048, 2048), (3, 1, 2048, 2048), (3, 4, 2048, 2048), (3, 4, 7, 16), (2, 9, 31, 128)],
)
def test_precomputed_topk_parallel_layers_and_rows_graph_replay(layer_count, rows, width, hot):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    torch.npu.set_device(0)
    pytest.importorskip("vllm_ascend.vllm_ascend_C", reason="Build the standard vLLM Ascend extension")
    if not hasattr(torch.ops._C_ascend, "npu_precomputed_topk_manage"):
        pytest.skip("Rebuild the standard extension with precomputed TopK support")
    manage = torch.ops._C_ascend.npu_precomputed_topk_manage
    device = torch.device("npu:0")
    topk = torch.empty((rows, width), dtype=torch.int64, device=device)
    layers = []
    descriptor_rows = []
    for layer in range(layer_count):
        lengths = (2305 + torch.arange(rows) * 17 + layer * 31).int()
        state = dict(
            table=torch.empty((rows, 32 + layer), dtype=torch.int32, device=device),
            lengths=lengths.to(device),
            requests=torch.empty(rows, dtype=torch.int64, device=device),
            tails=torch.empty((rows, 2), dtype=torch.int32, device=device),
            keys=torch.full((rows, hot + 1), -1, dtype=torch.int64, device=device),
            previous=torch.full((rows,), -1, dtype=torch.int64, device=device),
            physical=torch.empty((rows, width), dtype=torch.int32, device=device),
            sources=torch.empty((2, rows, width), dtype=torch.int64, device=device),
            destinations=torch.empty((2, rows, width), dtype=torch.int64, device=device),
            sizes=torch.empty((2, rows, width), dtype=torch.int32, device=device),
            bases=tuple(base + layer * (1 << 24) for base in (HOST_K, HOST_V, HOT_K, HOT_V)),
        )
        descriptor_rows.append(
            [
                state[name].data_ptr()
                for name in (
                    "table",
                    "lengths",
                    "requests",
                    "tails",
                    "keys",
                    "previous",
                    "physical",
                    "sources",
                    "destinations",
                    "sizes",
                )
            ]
            + list(state["bases"])
            + [state["table"].shape[1], 0]
        )
        layers.append(state)
    descriptors = torch.tensor(descriptor_rows, dtype=torch.int64, device=device)

    def invoke():
        result = manage(
            topk,
            descriptors,
            HOST_BLOCKS,
            BLOCK_SIZE,
            hot,
            hot + BLOCK_SIZE,
            K_BYTES,
            V_BYTES,
        )
        assert result is None

    def update(step):
        for layer, state in enumerate(layers):
            lengths = (2305 + torch.arange(rows) * 17 + layer * 31 + (step // 5) * BLOCK_SIZE).int()
            state["lengths"].copy_(lengths)
            identities = torch.arange(1, rows + 1, dtype=torch.int64) + layer * 100 + (step // 3) * 1000
            if rows > 1 and (step + layer) % 2:
                identities[-1] = 0
            state["requests"].copy_(identities)
            columns = state["table"].shape[1]
            table = (torch.arange(columns).expand(rows, -1) * 3 + torch.arange(rows)[:, None] + layer * 7) % HOST_BLOCKS
            if step >= 4:
                table = table.roll(1, -1)
            if layer == 2 and step == 6:
                table[:, 0] = -1
            state["table"].copy_(table)
            tails = torch.stack((torch.arange(rows), ((lengths - 1) % BLOCK_SIZE + step) % BLOCK_SIZE // 2), dim=1)
            if layer == 1 and step >= 4:
                tails[-1, 0] = -1
            state["tails"].copy_(tails)
            if layer == 1 and step == 2:
                state["keys"].fill_(-1)
        lengths = layers[0]["lengths"].cpu()
        ids = (torch.arange(width).expand(rows, -1) + (step // 2) * 257) % (lengths[:, None] - 1)
        for layer, state in enumerate(layers):
            ids[:, -1 - layer] = state["lengths"].cpu() - 1
        if step:
            ids[:, 1] = ids[:, 0]
            ids[:, 2] = -1
        topk.copy_(ids)

    def expected():
        return [
            _reference(
                topk.cpu(),
                state["table"].cpu(),
                state["lengths"].cpu(),
                state["requests"].cpu(),
                state["tails"].cpu(),
                state["keys"].cpu(),
                state["previous"].cpu(),
                hot,
                state["bases"],
            )
            for state in layers
        ]

    def check(references):
        for layer, (state, reference) in enumerate(zip(layers, references)):
            for name, target in zip(("physical", "sources", "destinations", "sizes", "keys", "previous"), reference):
                assert torch.equal(state[name].cpu(), target), (layer, name)

    update(0)
    reference = expected()
    invoke()
    torch.npu.synchronize()
    check(reference)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()
    torch.npu.synchronize()
    # Each layer keeps its own metadata and residency while the captured pointer table stays fixed.
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
