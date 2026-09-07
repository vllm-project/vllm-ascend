import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

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


def _load_sparse_kv_ops():
    if not enable_custom_op():
        pytest.skip("custom ops are unavailable")

    required = {
        "sparse_kv_warmup_lru_resident_threads",
        "sparse_kv_lru_resident_compact",
        "sparse_kv_lru_resident_compact_with_plan_stable_rows",
        "sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
        "sparse_kv_compute_lru_resident_addrs",
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


def test_sparse_kv_warmup_threads(sparse_kv_ops):
    assert sparse_kv_ops.sparse_kv_warmup_lru_resident_threads(1) == 1


def test_sparse_kv_restore_tensor_views(sparse_kv_ops):
    bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    int16 = torch.tensor([3, 4], dtype=torch.int16)

    bf16_view = sparse_kv_ops.sparse_kv_restore_bfloat16_tensor(bf16.data_ptr(), list(bf16.shape))
    int16_view = sparse_kv_ops.sparse_kv_restore_int16_tensor(int16.data_ptr(), list(int16.shape))

    assert bf16_view.data_ptr() == bf16.data_ptr()
    assert int16_view.data_ptr() == int16.data_ptr()
    assert bf16_view.dtype == torch.bfloat16
    assert int16_view.dtype == torch.int16


def test_sparse_kv_compute_lru_resident_addrs(sparse_kv_ops):
    miss_count = torch.tensor([2, 1], dtype=torch.int32)
    miss_tokens = torch.tensor([[1, 6, -1], [4, -1, -1]], dtype=torch.int32)
    miss_slots = torch.tensor([[0, 2, -1], [1, -1, -1]], dtype=torch.int32)
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    gvas_buffer = torch.full((12,), -1, dtype=torch.int64)
    addr_buffer = torch.full((12,), -1, dtype=torch.int64)
    size_buffer = torch.full((12,), -1, dtype=torch.int32)
    num_tokens_buffer = torch.zeros(1, dtype=torch.int32)

    num_tokens_to_load = sparse_kv_ops.sparse_kv_compute_lru_resident_addrs(
        miss_count,
        miss_tokens,
        miss_slots,
        block_table,
        4,
        8,
        12,
        1000,
        2000,
        3000,
        4000,
        4,
        2,
        gvas_buffer,
        addr_buffer,
        size_buffer,
        num_tokens_buffer,
    )

    assert num_tokens_to_load == 3
    assert num_tokens_buffer.item() == 6
    assert gvas_buffer[:6].tolist() == [1328, 1368, 1672, 2492, 2552, 3008]
    assert addr_buffer[:6].tolist() == [3000, 3016, 3040, 4000, 4024, 4060]
    assert size_buffer[:6].tolist() == [8, 8, 8, 12, 12, 12]


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
