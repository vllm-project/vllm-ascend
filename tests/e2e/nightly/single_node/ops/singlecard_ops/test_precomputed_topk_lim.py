# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

BLOCK_SIZE = 128
HOST_BLOCKS = 64
K_BYTES, V_BYTES = 1024, 128
HOST_K, HOST_V, HOT_K, HOT_V = (1 << shift for shift in (36, 37, 38, 39))


def test_lru_reference_distinguishes_slot_order():
    # A/B/C/D occupy slots 0/1/2/3. Touch B then A in separate steps;
    # C becomes oldest although B is the first unselected slot by slot number.
    keys, order, previous = [-1] * 5, list(range(4)), -1
    for topk in ([0, 1, 2, 3], [1, -1, -1, -1], [0, -1, -1, -1]):
        _, _, keys, previous, order = _lru_reference_row(topk, [0, 1, 2], 300, 9, (-1, 0), keys, previous, order, 4)
    assert order == [2, 3, 1, 0]
    physical, misses, keys, previous, order = _lru_reference_row(
        [0, 4, -1, -1], [0, 1, 2], 300, 9, (-1, 0), keys, previous, order, 4
    )
    assert physical == [0, 2, -1, -1]
    assert misses == [False, True, False, False]
    assert keys == [0, 1, 4, 3, -1]
    assert order == [3, 1, 0, 2]


def test_lru_reference_protects_hits_and_handles_lifecycle():
    table = [0, 1, 2]
    physical, misses, keys, previous, order = _lru_reference_row(
        [4, 0, 1, 2], table, 300, 7, (-1, 0), [0, 1, 2, 3, -1], 7, [0, 1, 2, 3], 4
    )
    assert physical == [3, 0, 1, 2]  # The first miss must not evict a later hit.
    assert misses == [True, False, False, False]
    assert keys == [0, 1, 2, 4, -1]
    physical, misses, keys, previous, order = _lru_reference_row(
        [257, 257, -1, 300], table, 300, 7, (0, 1), keys, previous, order, 4
    )
    assert physical == [5, 5, -1, -1] and not any(misses)
    physical, misses, keys, previous, order = _lru_reference_row(
        [0, -1, -1, -1], table, 300, 0, (-1, 0), keys, previous, order, 4
    )
    assert physical == [-1] * 4 and not any(misses)
    assert keys == [-1] * 5 and previous == 0 and order == [0, 1, 2, 3]
    physical, misses, keys, previous, order = _lru_reference_row(
        [7, 7, -1, -1], table, 300, 8, (-1, 0), keys, previous, order, 4
    )
    assert physical == [0, 0, -1, -1] and misses == [True, False, False, False]
    assert keys == [7, -1, -1, -1, -1] and previous == 8


def _lru_reference_row(topk, table, length, request, tail, keys, previous, order, hot):
    """Protect this step's hits, evict old unselected slots, then touch in TopK order."""
    keep = request != 0 and previous == request
    resident = list(keys[:hot]) if keep else [-1] * hot
    recency = list(order) if keep else list(range(hot))
    assert sorted(recency) == list(range(hot))
    mapping = {token: slot for slot, token in enumerate(resident) if token >= 0}
    sources, physical, selected_slots = [], [], set()
    for token in topk:
        source, selected = -1, -1
        if request and 0 <= token < length and token // BLOCK_SIZE < len(table):
            page, offset = table[token // BLOCK_SIZE], token % BLOCK_SIZE
            if 0 <= page < HOST_BLOCKS:
                if tail[0] >= 0 and token // BLOCK_SIZE == (length - 1) // BLOCK_SIZE and offset >= tail[1]:
                    selected = hot + offset
                elif token != length - 1:
                    source = page * BLOCK_SIZE + offset
                    if source in mapping:
                        selected = mapping[source]
                        selected_slots.add(selected)
        sources.append(source)
        physical.append(selected)
    misses = [False] * len(topk)
    empty = [slot for slot, token in enumerate(resident) if token < 0]
    candidates = empty + [slot for slot in recency if resident[slot] >= 0 and slot not in selected_slots]
    for column, source in enumerate(sources):
        if source < 0 or physical[column] >= 0:
            continue
        if source not in mapping:
            assert candidates
            slot = candidates.pop(0)
            if resident[slot] >= 0:
                del mapping[resident[slot]]
            resident[slot] = source
            mapping[source] = slot
            misses[column] = True
        physical[column] = mapping[source]
    for source, slot in zip(sources, physical):
        if source >= 0 and 0 <= slot < hot:
            recency.remove(slot)
            recency.append(slot)
    return physical, misses, resident + [-1], request, recency


def _reference_lru(topk, table, lengths, requests, tails, keys, previous, lru, hot, bases):
    rows, width = topk.shape
    physical = torch.full((rows, width), -1, dtype=torch.int32)
    sources = torch.empty((2, rows, width), dtype=torch.int64)
    destinations = torch.empty_like(sources)
    sizes = torch.zeros((2, rows, width), dtype=torch.int32)
    updated, next_lru = keys.clone(), lru.clone()
    for row in range(rows):
        ids, pages = topk[row].tolist(), table[row].tolist()
        selected, misses, resident, _, order = _lru_reference_row(
            ids, pages, int(lengths[row]), int(requests[row]), tails[row].tolist(),
            keys[row].tolist(), int(previous[row]), lru[row].tolist(), hot,
        )
        physical[row] = torch.tensor(selected, dtype=torch.int32)
        updated[row, : hot + 1] = torch.tensor(resident, dtype=torch.int64)
        next_lru[row] = torch.tensor(order, dtype=torch.int32)
        for col, (token, miss) in enumerate(zip(ids, misses)):
            source = 0
            if int(requests[row]) and 0 <= token < int(lengths[row]) and token // BLOCK_SIZE < len(pages):
                page = pages[token // BLOCK_SIZE]
                if 0 <= page < HOST_BLOCKS:
                    source = page * BLOCK_SIZE + token % BLOCK_SIZE
            # Tail and current-token selections have no Host source in the native descriptor.
            if selected[col] >= hot or token == int(lengths[row]) - 1:
                source = 0
            target = row * (hot + BLOCK_SIZE) + (selected[col] if miss else 0)
            for component, (host, device, stride) in enumerate(zip(bases[:2], bases[2:], (K_BYTES, V_BYTES))):
                sources[component, row, col] = host + source * stride
                destinations[component, row, col] = device + target * stride
                sizes[component, row, col] = stride if miss else 0
    return physical, sources, destinations, sizes, updated, requests.clone(), next_lru


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


def test_precomputed_topk_lru_eviction_and_request_reuse_graph():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    torch.npu.set_device(0)
    pytest.importorskip("vllm_ascend.vllm_ascend_C", reason="Build the standard vLLM Ascend extension")
    manage = torch.ops._C_ascend.npu_precomputed_topk_manage
    device, rows, hot, width = torch.device("npu:0"), 2, 4, 4
    topk = torch.zeros((rows, width), dtype=torch.int64, device=device)
    layers, descriptors = [], []
    for layer in range(2):
        state = dict(
            table=torch.tensor([[10 * layer, 10 * layer + 1, 10 * layer + 2]] * rows,
                               dtype=torch.int32, device=device),
            lengths=torch.full((rows,), 300, dtype=torch.int32, device=device),
            requests=torch.ones(rows, dtype=torch.int64, device=device),
            tails=torch.tensor([[-1, 0], [0, 1]], dtype=torch.int32, device=device),
            keys=torch.full((rows, hot + 1), -1, dtype=torch.int64, device=device),
            previous=torch.full((rows,), -1, dtype=torch.int64, device=device),
            lru=torch.arange(hot, dtype=torch.int32, device=device).expand(rows, -1).clone(),
            physical=torch.empty((rows, width), dtype=torch.int32, device=device),
            sources=torch.empty((2, rows, width), dtype=torch.int64, device=device),
            destinations=torch.empty((2, rows, width), dtype=torch.int64, device=device),
            sizes=torch.empty((2, rows, width), dtype=torch.int32, device=device),
            bases=tuple(base + layer * (1 << 24) for base in (HOST_K, HOST_V, HOT_K, HOT_V)),
        )
        descriptors.append([state[name].data_ptr() for name in (
            "table", "lengths", "requests", "tails", "keys", "previous",
            "physical", "sources", "destinations", "sizes",
        )] + list(state["bases"]) + [state["table"].shape[1], state["lru"].data_ptr()])
        layers.append(state)
    descriptors = torch.tensor(descriptors, dtype=torch.int64, device=device)

    def invoke():
        manage(topk, descriptors, HOST_BLOCKS, BLOCK_SIZE, hot, hot + BLOCK_SIZE, K_BYTES, V_BYTES)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()
    torch.npu.synchronize()
    for state in layers:
        state["keys"].fill_(-1)
        state["previous"].fill_(-1)
        state["lru"].copy_(torch.arange(hot, dtype=torch.int32).expand(rows, -1))

    steps = [
        ([[0, 1, 2, 3], [257, 257, -1, 300]], [9, 20]),
        ([[1, -1, -1, -1], [0, -1, -1, -1]], [9, 0]),
        ([[0, -1, -1, -1], [7, 7, -1, -1]], [9, 21]),
        ([[0, 4, -1, -1], [7, 129, -1, 300]], [9, 21]),
        ([[5, 5, -1, -1], [7, -1, -1, -1]], [9, 21]),
        ([[6, 1, 4, 5], [7, -1, -1, -1]], [9, 21]),
    ]
    for step, (indices, requests) in enumerate(steps):
        topk.copy_(torch.tensor(indices, dtype=torch.int64))
        for state in layers:
            state["requests"].copy_(torch.tensor(requests, dtype=torch.int64))
            if step == 3:
                state["table"][1, 1] = -1
        reference = [
            _reference_lru(topk.cpu(), state["table"].cpu(), state["lengths"].cpu(),
                           state["requests"].cpu(), state["tails"].cpu(), state["keys"].cpu(),
                           state["previous"].cpu(), state["lru"].cpu(), hot, state["bases"])
            for state in layers
        ]
        if step == 3:
            old = _reference(topk.cpu(), layers[0]["table"].cpu(), layers[0]["lengths"].cpu(),
                             layers[0]["requests"].cpu(), layers[0]["tails"].cpu(), layers[0]["keys"].cpu(),
                             layers[0]["previous"].cpu(), hot, layers[0]["bases"])
            assert int(old[0][0, 1]) == 1 and int(reference[0][0][0, 1]) == 2
        graph.replay()
        torch.npu.synchronize()
        for layer, (state, expected) in enumerate(zip(layers, reference)):
            names = ("physical", "sources", "destinations", "sizes", "keys", "previous", "lru")
            for name, target in zip(names, expected):
                assert torch.equal(state[name].cpu(), target), (step, layer, name)
            actual_slots = state["physical"].cpu().tolist()
            actual_keys = state["keys"].cpu().tolist()
            pages = state["table"].cpu().tolist()
            for row, ids in enumerate(indices):
                for token, slot in zip(ids, actual_slots[row]):
                    if 0 <= slot < hot:
                        expected_source = pages[row][token // BLOCK_SIZE] * BLOCK_SIZE + token % BLOCK_SIZE
                        assert actual_keys[row][slot] == expected_source
                    elif slot >= hot:
                        assert slot == hot + token % BLOCK_SIZE
            if step == 1:
                assert state["lru"][0].cpu().tolist() == [0, 2, 3, 1], "all-hit access must refresh LRU"
            if step == 3:
                assert state["physical"][0, :2].cpu().tolist() == [0, 2], "LRU must evict slot 2, not slot 1"
            if step == 4:
                assert state["physical"][0, 0] == state["physical"][0, 1]
                assert state["sizes"][0, 0, :2].cpu().tolist() == [K_BYTES, 0]
            if step == 5:
                assert state["physical"][0, :2].cpu().tolist() == [0, 1], "protect the later hit before evicting"


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
            lru=torch.arange(hot, dtype=torch.int32, device=device).expand(rows, -1).clone(),
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
            + [state["table"].shape[1], state["lru"].data_ptr()]
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

    def poison_outputs():
        for state in layers:
            state["physical"].fill_(-12345)
            state["sizes"].fill_(-1)

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
            if step == 7:
                # Row zero keeps its nonzero owner from step six; normalize its dirty sentinel.
                state["keys"][0, hot].fill_(17)
        lengths = layers[0]["lengths"].cpu()
        ids = (torch.arange(width).expand(rows, -1) + (step // 2) * 257) % (lengths[:, None] - 1)
        for layer, state in enumerate(layers):
            ids[:, -1 - layer] = state["lengths"].cpu() - 1
        if step:
            ids[:, 1] = ids[:, 0]
            ids[:, 2] = -1
        if step == 1:
            # Preserve the warmed source subset while changing physical-output order.
            ids = ids.flip(-1)
        topk.copy_(ids)
        poison_outputs()

    def expected():
        return [
            _reference_lru(
                topk.cpu(),
                state["table"].cpu(),
                state["lengths"].cpu(),
                state["requests"].cpu(),
                state["tails"].cpu(),
                state["keys"].cpu(),
                state["previous"].cpu(),
                state["lru"].cpu(),
                hot,
                state["bases"],
            )
            for state in layers
        ]

    def check(references):
        for layer, (state, reference) in enumerate(zip(layers, references)):
            names = ("physical", "sources", "destinations", "sizes", "keys", "previous", "lru")
            for name, target in zip(names, reference):
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
        poison_outputs()
        graph.replay()
    torch.npu.synchronize()
    check(reference)
