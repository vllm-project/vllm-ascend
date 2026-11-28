import math
from collections.abc import Callable

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.gbsa import qsa_gbsa_attention
from vllm_ascend.ops.triton.qwen4_exp.gbsa_input_preparation import prepare_qsa_gbsa_inputs
from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
requires_a3 = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")


@requires_a3
@torch.inference_mode()
def test_qsa_pack_supports_non_power_of_two_output_width() -> None:
    groups, ratio = 512, 4
    block_indices = torch.arange(groups, dtype=torch.int32).repeat(2, 1).npu()
    query_positions = torch.tensor([2048, 4094], dtype=torch.int64).npu()
    sequence_lengths = torch.tensor([4096, 4096], dtype=torch.int32).npu()
    token_to_req = torch.tensor([0, 1], dtype=torch.int32).npu()

    actual = pack_qsa_group_indices_npu(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        token_topk=2048,
        compress_ratio=ratio,
    )

    expected = torch.full((2, groups + ratio - 1), -1, dtype=torch.int32)
    expected[0, :512] = torch.arange(512, dtype=torch.int32)
    expected[0, 512] = 2048
    expected[1, :512] = torch.arange(512, dtype=torch.int32)
    expected[1, 512:515] = torch.arange(4092, 4095, dtype=torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_a3
@torch.inference_mode()
def test_qsa_gbsa_preparation_ragged_noncontiguous_table() -> None:
    compact = torch.tensor(
        [[0, 2, -1, 4, -1, -1, -1, -1, 16, 17, 18], [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1]],
        dtype=torch.int32,
        device="npu",
    )
    positions = torch.tensor([18, 31], dtype=torch.int64, device="npu")
    lengths = torch.tensor([19, 32], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    table_storage = torch.arange(24, dtype=torch.int32, device="npu").reshape(2, 12)
    block_table = table_storage[:, ::2]

    sparse, counts, visible, cu_query, row_table = prepare_qsa_gbsa_inputs(
        compact, positions, lengths, token_to_req, block_table, num_kv_heads=2
    )

    sparse_cpu = sparse.cpu()
    for head in range(2):
        torch.testing.assert_close(sparse_cpu[head, 0, :2], torch.tensor([0, 2], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(sparse_cpu[head, 1, :8], torch.arange(8, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), torch.tensor([[2, 8], [2, 8]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(visible.cpu(), torch.tensor([19, 32], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(cu_query.cpu(), torch.tensor([0, 1, 2], dtype=torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(row_table.cpu(), block_table.cpu(), rtol=0, atol=0)


GBSA_ROWS = 2
GBSA_NUM_Q_HEADS = 4
GBSA_NUM_KV_HEADS = 2
GBSA_HEAD_DIM = 128
GBSA_PAGE_SIZE = 128
GBSA_BF16_ATOL = 2**-6
GBSA_BF16_RTOL = 2**-6
GBSA_BF16_MAX_ABS_ERROR = 1.0
GBSA_REQUIRED_PASS_RATE = 0.99
GbsaCase = tuple[
    Callable[[], torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _compact_gbsa_indices(seq_lens_values: tuple[int, int]) -> torch.Tensor:
    groups = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu")
    residual_slots = torch.full((GBSA_ROWS, 3), -1, dtype=torch.int32, device="npu")
    for row, length in enumerate(seq_lens_values):
        tail_start = length // 4 * 4
        tail_count = length - tail_start
        if tail_count:
            residual_slots[row, :tail_count] = torch.arange(tail_start, length, dtype=torch.int32, device="npu")
    return torch.cat((groups, residual_slots), dim=1)


def _gbsa_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for row in range(query.shape[0]):
        request = int(token_to_req[row])
        length = int(seq_lens[request])
        tokens = torch.arange(length, device=query.device)
        pages = block_table[request, tokens // GBSA_PAGE_SIZE].long()
        offsets = tokens % GBSA_PAGE_SIZE
        keys = key_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        values = value_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,shd->hs", query[row].float(), keys)
        probabilities = (scores / math.sqrt(GBSA_HEAD_DIM)).softmax(-1)
        rows.append(torch.einsum("hs,shd->hd", probabilities, values))
    return torch.stack(rows)


def _make_gbsa_case(seq_lens_values: tuple[int, int] = (9, 11)) -> GbsaCase:
    torch.manual_seed(20261009)
    query = torch.randn((GBSA_ROWS, GBSA_NUM_Q_HEADS, GBSA_HEAD_DIM), dtype=torch.bfloat16, device="npu")
    key_cache = torch.randn((4, GBSA_PAGE_SIZE, GBSA_NUM_KV_HEADS, GBSA_HEAD_DIM), dtype=torch.bfloat16, device="npu")
    value_cache = torch.randn_like(key_cache)
    compact_indices = _compact_gbsa_indices(seq_lens_values)
    block_table = torch.tensor([[2], [3]], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device="npu")
    logical_positions = seq_lens - 1
    output = torch.empty_like(query)

    def call() -> torch.Tensor:
        return qsa_gbsa_attention(
            query,
            key_cache,
            value_cache,
            compact_indices,
            block_table,
            token_to_req,
            logical_positions,
            seq_lens,
            output,
        )

    return call, query, key_cache, value_cache, block_table, token_to_req, seq_lens


def _expected_gbsa(case: GbsaCase) -> torch.Tensor:
    _, query, key_cache, value_cache, block_table, token_to_req, seq_lens = case
    return _gbsa_reference(query, key_cache, value_cache, block_table, token_to_req, seq_lens)


def _assert_gbsa_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.cpu().double() - expected.cpu().double()).abs()
    assert torch.isfinite(error).all()
    passing = error <= GBSA_BF16_ATOL + GBSA_BF16_RTOL * expected.cpu().double().abs()
    pass_rate = passing.double().mean().item()
    max_abs_error = error.max().item()
    assert pass_rate >= GBSA_REQUIRED_PASS_RATE, (
        f"GBSA BF16 mixed-tolerance pass rate {pass_rate:.2%} is below {GBSA_REQUIRED_PASS_RATE:.2%}; "
        f"max absolute error: {max_abs_error:.6f}"
    )
    assert max_abs_error <= GBSA_BF16_MAX_ABS_ERROR, (
        f"GBSA BF16 max absolute error {max_abs_error:.6f} exceeds {GBSA_BF16_MAX_ABS_ERROR}"
    )


@pytest.mark.parametrize("execution", ("eager", "compile"))
@pytest.mark.parametrize(
    "seq_lens_values",
    (
        pytest.param((1, 1), id="minimum-length"),
        pytest.param((9, 11), id="within-page"),
    ),
)
@torch.no_grad()
def test_qwen4_exp_gbsa_matches_dense_reference(execution: str, seq_lens_values: tuple[int, int]) -> None:
    case = _make_gbsa_case(seq_lens_values)
    call = case[0]
    expected = _expected_gbsa(case)
    if execution == "compile":
        call = torch.compile(call, backend="npugraph_ex", fullgraph=True, dynamic=False)

    actual = call().clone()
    torch.npu.synchronize()

    _assert_gbsa_close(actual, expected)


@torch.no_grad()
def test_qwen4_exp_gbsa_npugraph_replays_updated_input() -> None:
    case = _make_gbsa_case()
    call, query, *_ = case
    call()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured_output = call()

    graph.replay()
    torch.npu.synchronize()
    _assert_gbsa_close(captured_output, _expected_gbsa(case))

    query.mul_(0.5)
    expected_replay = _expected_gbsa(case)
    graph.replay()
    torch.npu.synchronize()
    _assert_gbsa_close(captured_output, expected_replay)


@pytest.mark.parametrize("current_visible", (2050, 2051, 2052))
@torch.inference_mode()
def test_qsa_gbsa_preparation_reuses_saved_mtp_tail(current_visible: int) -> None:
    groups = torch.arange(512, dtype=torch.int32, device="npu").unsqueeze(0)
    saved_tail = torch.tensor([[2048, 2049, -1]], dtype=torch.int32, device="npu")
    compact = torch.cat((groups, saved_tail), dim=1)
    positions = torch.tensor([current_visible - 1], dtype=torch.int64, device="npu")
    lengths = torch.tensor([current_visible], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0], dtype=torch.int32, device="npu")
    block_table = torch.arange(17, dtype=torch.int32, device="npu").unsqueeze(0)

    sparse, counts, visible, _, _ = prepare_qsa_gbsa_inputs(
        compact, positions, lengths, token_to_req, block_table, num_kv_heads=1
    )

    torch.testing.assert_close(sparse[0, 0, :512].cpu(), torch.arange(512, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), torch.tensor([[512]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(visible.cpu(), torch.tensor([2050], dtype=torch.int32), rtol=0, atol=0)
