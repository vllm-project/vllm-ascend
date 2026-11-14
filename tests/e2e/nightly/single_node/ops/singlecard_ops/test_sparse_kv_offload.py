from zlib import adler32

import pytest

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.request_identity import (  # noqa: E402
    RequestIdentityRegistry,
)
from vllm_ascend.utils import enable_custom_op  # noqa: E402

ROWS = 4
TOPK = 8
CAPACITY = 16
MAX_TOKEN = 64
THREADS = 1
MEMBERSHIP_CONTROL_OFFSET = 16384
MEMBERSHIP_STORAGE_SIZE = 16400
EXTERNAL_PLAN_READY_MARKER = 0x5A45
DIRECT_SELECTION_LAYOUT_MARKER = 0x5A44
PAIRED_SELECTION_COPY_MARKER = 0x5A56
COLLIDING_REQUEST_IDS = (
    "234054828099429543",
    "471754017250656876",
    "304399536143223879",
)


def _load_sparse_kv_ops():
    if not enable_custom_op():
        pytest.skip("custom ops are unavailable")

    required = {
        "sparse_kv_warmup_lru_resident_threads",
        "sparse_kv_lru_resident_compact_with_plan_stable_rows",
        "sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
        "sparse_kv_restore_bfloat16_tensor",
        "sparse_kv_restore_int16_tensor",
    }
    missing = [name for name in required if not hasattr(torch.ops._C_ascend, name)]
    if missing:
        pytest.skip(f"sparse KV ops are unavailable: {missing}")
    return torch.ops._C_ascend


@pytest.fixture(scope="module")
def sparse_kv_ops():
    return _load_sparse_kv_ops()


class PlannerState:
    def __init__(self):
        self.req_ids = torch.empty(ROWS, dtype=torch.int64)
        self.last_req_ids = torch.full((ROWS,), -1, dtype=torch.int64)
        self.topk = torch.empty((ROWS, TOPK), dtype=torch.int32)
        self.stable_prefix_lens = torch.empty(ROWS, dtype=torch.int32)
        self.visible_seq_lens = torch.empty(ROWS, dtype=torch.int32)
        self.slot_to_token = torch.full((ROWS, CAPACITY), -1, dtype=torch.int32)
        self.lru_slots = torch.arange(CAPACITY, dtype=torch.int32).repeat(ROWS, 1)
        self.current_slots = torch.empty((ROWS, TOPK), dtype=torch.int32)
        self.miss_count = torch.empty(ROWS, dtype=torch.int32)
        self.miss_tokens = torch.empty((ROWS, TOPK), dtype=torch.int32)
        self.miss_slots = torch.empty((ROWS, TOPK), dtype=torch.int32)
        self.token_mark = torch.zeros((THREADS, MAX_TOKEN), dtype=torch.int32)
        self.token_pos = torch.full((THREADS, MAX_TOKEN), -1, dtype=torch.int32)
        self.slot_workspace = torch.empty((THREADS, CAPACITY * 3), dtype=torch.int32)
        self.miss_position = torch.empty((THREADS, TOPK), dtype=torch.int32)
        self.epochs = torch.zeros(THREADS, dtype=torch.int32)
        self.physical_rows = torch.empty(ROWS * 3, dtype=torch.int32)
        self.membership = torch.full((ROWS, MEMBERSHIP_STORAGE_SIZE), -1, dtype=torch.int16)
        control = self.membership[
            :,
            MEMBERSHIP_CONTROL_OFFSET : MEMBERSHIP_CONTROL_OFFSET + 8,
        ]
        control[:, 1] = EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = TOPK
        control[:, 3] = MEMBERSHIP_CONTROL_OFFSET - TOPK
        control[:, 7] = PAIRED_SELECTION_COPY_MARKER

    @staticmethod
    def ptr(tensor):
        return tensor.data_ptr()

    def set_inputs(self, req_ids, topk, stable_prefix_lens, visible_seq_lens=None):
        num_rows = len(req_ids)
        if visible_seq_lens is None:
            visible_seq_lens = [MAX_TOKEN] * num_rows
        self.req_ids[:num_rows].copy_(torch.tensor(req_ids, dtype=torch.int64))
        self.topk[:num_rows].copy_(topk)
        self.stable_prefix_lens[:num_rows].copy_(torch.tensor(stable_prefix_lens, dtype=torch.int32))
        self.visible_seq_lens[:num_rows].copy_(torch.tensor(visible_seq_lens, dtype=torch.int32))

    def call(self, ops, req_ids, topk, stable_prefix_lens, visible_seq_lens=None, enqueue=False):
        num_rows = len(req_ids)
        self.set_inputs(req_ids, topk, stable_prefix_lens, visible_seq_lens)
        planner = (
            ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows
            if enqueue
            else ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
        )
        plan = self.membership[:, MEMBERSHIP_CONTROL_OFFSET - TOPK :]
        planner(
            self.ptr(self.req_ids),
            self.ptr(self.last_req_ids),
            self.ptr(self.topk),
            self.ptr(self.stable_prefix_lens),
            self.ptr(self.slot_to_token),
            self.ptr(self.lru_slots),
            self.ptr(self.current_slots),
            self.ptr(self.miss_count),
            self.ptr(self.miss_tokens),
            self.ptr(self.miss_slots),
            self.ptr(self.token_mark),
            self.ptr(self.token_pos),
            self.ptr(self.slot_workspace),
            self.ptr(self.miss_position),
            self.ptr(self.epochs),
            self.ptr(self.physical_rows),
            ROWS,
            plan.data_ptr(),
            self.membership.stride(0),
            num_rows,
            TOPK,
            CAPACITY,
            MAX_TOKEN,
            THREADS,
            THREADS,
            self.ptr(self.visible_seq_lens),
        )

    @property
    def plan(self):
        return self.membership[
            :,
            MEMBERSHIP_CONTROL_OFFSET - TOPK : MEMBERSHIP_CONTROL_OFFSET + 8,
        ]


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_sparse_kv_warmup_threads(sparse_kv_ops):
    assert sparse_kv_ops.sparse_kv_warmup_lru_resident_threads(1) == 1


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_sparse_kv_restore_tensor_views(sparse_kv_ops):
    bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    int16 = torch.tensor([3, 4], dtype=torch.int16)

    bf16_view = sparse_kv_ops.sparse_kv_restore_bfloat16_tensor(bf16.data_ptr(), list(bf16.shape))
    int16_view = sparse_kv_ops.sparse_kv_restore_int16_tensor(int16.data_ptr(), list(int16.shape))

    assert bf16_view.data_ptr() == bf16.data_ptr()
    assert int16_view.data_ptr() == int16.data_ptr()
    assert bf16_view.dtype == torch.bfloat16
    assert int16_view.dtype == torch.int16


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_stable_rows_suffix_invalidation_and_plan_encoding(sparse_kv_ops):
    state = PlannerState()
    first = torch.stack([torch.arange(0, TOPK), torch.arange(TOPK, TOPK * 2)]).to(torch.int32)

    state.call(sparse_kv_ops, [101, 202], first, [MAX_TOKEN, MAX_TOKEN])
    assert state.miss_count[:2].tolist() == [TOPK, TOPK]
    assert bool((state.plan[:2, :TOPK] < 0).all())
    assert state.plan[:2, TOPK + 4].tolist() == [0, 1]
    assert state.plan[:2, TOPK + 5].tolist() == [
        DIRECT_SELECTION_LAYOUT_MARKER,
        DIRECT_SELECTION_LAYOUT_MARKER,
    ]
    assert state.plan[:2, TOPK + 6].tolist() == [CAPACITY, CAPACITY]
    assert state.plan[:2, TOPK + 7].tolist() == [
        PAIRED_SELECTION_COPY_MARKER,
        PAIRED_SELECTION_COPY_MARKER,
    ]

    swapped = first.flip(0)
    state.call(sparse_kv_ops, [202, 101], swapped, [MAX_TOKEN, MAX_TOKEN])
    assert state.miss_count[:2].tolist() == [0, 0]
    assert bool((state.plan[:2, :TOPK] > 0).all())
    assert state.plan[:2, TOPK + 4].tolist() == [1, 0]
    assert state.plan[:2, TOPK + 7].tolist() == [
        PAIRED_SELECTION_COPY_MARKER,
        PAIRED_SELECTION_COPY_MARKER,
    ]

    state.call(sparse_kv_ops, [202, 101], swapped, [12, MAX_TOKEN])
    assert state.miss_count[:2].tolist() == [4, 0]
    assert bool((state.plan[0, :4] > 0).all())
    assert bool((state.plan[0, 4:TOPK] < 0).all())
    assert bool((state.plan[1, :TOPK] > 0).all())
    assert state.plan[:2, TOPK + 4].tolist() == [1, 0]


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
@pytest.mark.parametrize("force_row_reuse", [False, True])
@pytest.mark.parametrize("wrap_owner_ids", [False, True])
@pytest.mark.parametrize("next_request_id", COLLIDING_REQUEST_IDS)
def test_request_incarnations_do_not_inherit_resident_hits(
    sparse_kv_ops, force_row_reuse, wrap_owner_ids, next_request_id
):
    assert {adler32(req_id.encode()) for req_id in COLLIDING_REQUEST_IDS} == {585106354}
    state = PlannerState()
    registry = RequestIdentityRegistry(on_wrap=lambda: state.last_req_ids.fill_(-1))
    first_request_id = COLLIDING_REQUEST_IDS[0]
    first_owner = registry.get_or_create(first_request_id)
    owners = [first_owner]
    if force_row_reuse:
        # No empty physical rows remain, so the new owner must reset row zero.
        owners.extend(registry.get_or_create(f"occupant-{row}") for row in range(1, ROWS))

    topk = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
    state.call(sparse_kv_ops, owners, topk.repeat(len(owners), 1), [MAX_TOKEN] * len(owners))
    assert state.physical_rows[0].item() == 0
    assert state.miss_count[0].item() == TOPK

    # Repeated decode/MTP metadata for the same live request retains its hits.
    assert registry.get_or_create(first_request_id) == first_owner
    state.call(sparse_kv_ops, [first_owner], topk, [MAX_TOKEN])
    assert state.physical_rows[0].item() == 0
    assert state.miss_count[0].item() == 0
    assert bool((state.plan[0, :TOPK] > 0).all())

    registry.remove(first_request_id)
    if wrap_owner_ids:
        registry._next_owner_id = (1 << 63) - 1
        registry.get_or_create("last-before-wrap")
    next_owner = registry.get_or_create(next_request_id)
    if wrap_owner_ids:
        # Recycle the exact old owner: clearing resident identities must make
        # its historical KV cold even though the integer ID matches again.
        assert next_owner == first_owner
    else:
        assert next_owner != first_owner
    state.call(sparse_kv_ops, [next_owner], topk, [MAX_TOKEN])

    expected_row = 0 if force_row_reuse or wrap_owner_ids else 1
    assert state.physical_rows[0].item() == expected_row
    assert state.last_req_ids[expected_row].item() == next_owner
    assert state.miss_count[0].item() == TOPK
    assert bool((state.plan[0, :TOPK] < 0).all())
    assert state.plan[0, TOPK + 4].item() == expected_row

    # Both row-selection paths must produce the same reloads as a cold cache.
    reference = PlannerState()
    reference.call(sparse_kv_ops, [next_owner], topk, [MAX_TOKEN])
    torch.testing.assert_close(state.current_slots[0], reference.current_slots[0])
    torch.testing.assert_close(state.miss_tokens[0], reference.miss_tokens[0])
    torch.testing.assert_close(state.miss_slots[0], reference.miss_slots[0])
    torch.testing.assert_close(state.slot_to_token[expected_row], reference.slot_to_token[0])
    torch.testing.assert_close(state.plan[0, :TOPK], reference.plan[0, :TOPK])

    state.call(sparse_kv_ops, [next_owner], topk, [MAX_TOKEN])
    assert state.physical_rows[0].item() == expected_row
    assert state.miss_count[0].item() == 0
    assert bool((state.plan[0, :TOPK] > 0).all())


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_mtp_future_tokens_are_not_encoded_before_visible(sparse_kv_ops):
    state = PlannerState()
    topk = torch.arange(8, 16, dtype=torch.int32).repeat(ROWS, 1)
    visible_seq_lens = [9, 10, 11, 12]

    state.call(
        sparse_kv_ops,
        [101] * ROWS,
        topk,
        [8] * ROWS,
        visible_seq_lens,
    )

    assert state.miss_count[:ROWS].tolist() == [1, 2, 3, 4]
    for row, visible_seq_len in enumerate(visible_seq_lens):
        visible_count = visible_seq_len - 8
        assert bool((state.plan[row, : visible_count - 1] < 0).all())
        assert state.plan[row, visible_count - 1] > 0
        assert bool((state.plan[row, visible_count:TOPK] == 0).all())
        assert state.physical_rows[ROWS * 2 + row] == (row * CAPACITY + state.plan[row, visible_count - 1] - 1)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_missing_current_token_uses_reserved_row_sentinel(sparse_kv_ops):
    state = PlannerState()
    topk = torch.arange(TOPK, dtype=torch.int32).repeat(ROWS, 1)

    state.call(
        sparse_kv_ops,
        [101] * ROWS,
        topk,
        [8] * ROWS,
        [12] * ROWS,
    )

    assert state.physical_rows[ROWS * 2 : ROWS * 3].tolist() == [row * CAPACITY + CAPACITY - 1 for row in range(ROWS)]


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_enqueued_planner_callback_owns_payload_until_stream_completion(sparse_kv_ops):
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    state = PlannerState()
    topk = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
    state.call(
        sparse_kv_ops,
        [101],
        topk,
        [MAX_TOKEN],
        enqueue=True,
    )
    torch_npu.npu.synchronize()
    assert state.miss_count[0].item() == TOPK
    assert bool((state.plan[0, :TOPK] < 0).all())


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A3,
    reason="Sparse KV offload native bindings are built only for Ascend A3",
)
def test_enqueued_planner_callback_survives_graph_replays(sparse_kv_ops):
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")

    states = [PlannerState(), PlannerState()]
    first = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
    for state in states:
        state.set_inputs([101], first, [MAX_TOKEN])
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            for state in states:
                state.call(
                    sparse_kv_ops,
                    [101],
                    first,
                    [MAX_TOKEN],
                    enqueue=True,
                )
        torch_npu.npu.synchronize()

        graph.replay()
        torch_npu.npu.synchronize()
        for state in states:
            assert state.miss_count[0].item() == TOPK
            assert bool((state.plan[0, :TOPK] < 0).all())

        for state in states:
            state.set_inputs([101], first.flip(1), [MAX_TOKEN])
        graph.replay()
        torch_npu.npu.synchronize()
        for state in states:
            assert state.miss_count[0].item() == 0
            assert bool((state.plan[0, :TOPK] > 0).all())

        for state in states:
            state.set_inputs([202], first, [MAX_TOKEN])
        graph.replay()
        torch_npu.npu.synchronize()
        for state in states:
            assert state.miss_count[0].item() == TOPK
            assert bool((state.plan[0, :TOPK] < 0).all())
    finally:
        graph.reset()
        torch_npu.npu.synchronize()
