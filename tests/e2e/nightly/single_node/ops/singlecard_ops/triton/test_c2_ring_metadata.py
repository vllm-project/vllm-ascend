# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


def reference(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    source_cos: torch.Tensor,
    source_sin: torch.Tensor,
    num_reqs: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    output_dtype: torch.dtype,
    skip_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager C2 chain from ``AscendDSAV41StateMetadataBuilder``."""
    num_tokens = positions.shape[0]
    rope_dim = source_cos.shape[-1]
    starts = query_start_loc[:num_reqs].to(torch.int32)
    ends = query_start_loc[1 : num_reqs + 1].to(torch.int32)
    query_lens = ends - starts
    live = torch.arange(num_reqs) < num_actual_reqs
    used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
    used = torch.where(live, used, torch.zeros_like(used))
    if skip_update:
        used = torch.zeros_like(used)
    ring = torch.stack(
        (
            (seq_lens - query_lens).clamp_min(0),
            used,
            starts,
            starts,
            torch.where(used > 0, block_table[:num_reqs, 0], torch.zeros_like(used)),
        )
    )
    valid_end = min(int(query_start_loc[num_actual_reqs]), num_actual_tokens)
    complete = (positions.remainder(2) == 1) & (torch.arange(num_tokens) < valid_end)
    if skip_update:
        complete = torch.zeros_like(complete)
    source_positions = torch.where(complete, positions - 1, torch.zeros_like(positions))
    gather_idx = source_positions.reshape(-1, 1, 1, 1).expand(num_tokens, 1, 1, rope_dim)
    cos = torch.gather(source_cos, 0, gather_idx).reshape(num_tokens, rope_dim).to(output_dtype)
    sin = torch.gather(source_sin, 0, gather_idx).reshape(num_tokens, rope_dim).to(output_dtype)
    return ring, complete, source_positions, cos, sin


def validate_cmp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Ring rows and source positions are integer controls, the mask is boolean
    # and the RoPE rows are gathered copies: exact comparison applies.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def run_kernel(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    source_cos: torch.Tensor,
    source_sin: torch.Tensor,
    num_reqs: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    output_dtype: torch.dtype = torch.float32,
    skip_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = positions.shape[0]
    rope_dim = source_cos.shape[-1]
    ring = torch.full((5, num_reqs), 777, dtype=torch.int32, device="npu")
    complete = ~torch.ones((num_tokens,), dtype=torch.bool, device="npu")
    source_positions = torch.full((num_tokens,), 777, dtype=torch.int64, device="npu")
    cos_output = torch.zeros((num_tokens, 1, 1, rope_dim), dtype=output_dtype, device="npu")
    sin_output = torch.zeros_like(cos_output)
    result = build_c2_ring_metadata(
        query_start_loc,
        seq_lens,
        positions,
        block_table,
        source_cos,
        source_sin,
        num_reqs,
        num_tokens,
        num_actual_reqs,
        num_actual_tokens,
        skip_update=skip_update,
        ring_metadata_output=ring,
        complete_mask_output=complete,
        source_positions_output=source_positions,
        cos_output=cos_output,
        sin_output=sin_output,
    )
    assert result is None
    return (
        ring.cpu(),
        complete.cpu(),
        source_positions.cpu(),
        cos_output.cpu().reshape(num_tokens, rope_dim),
        sin_output.cpu().reshape(num_tokens, rope_dim),
    )


def make_case(
    num_reqs: int,
    num_tokens: int,
    rope_dim: int,
    num_actual_reqs: int,
    source_dtype: torch.dtype = torch.bfloat16,
    seed: int = 41,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Request boundaries split ``num_tokens`` evenly, so the trailing requests of
    # a padded carrier extend past ``num_actual_tokens``; both the clamp and the
    # ``live`` mask are exercised.
    cuts = [0]
    for index in range(num_reqs):
        remaining = num_tokens - cuts[-1]
        cuts.append(cuts[-1] + (remaining + num_reqs - index - 1) // (num_reqs - index))
    query_start_loc = torch.tensor(cuts, dtype=torch.int32, device="npu")
    seq_lens = torch.randint(1, 4096, (num_reqs,), dtype=torch.int32, device="npu")
    positions = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    block_table = torch.randint(0, 4096, (num_reqs, 4), dtype=torch.int32, device="npu")
    source_rows = max(num_tokens, 1)
    # Production shape: the full-source RoPE table is [rows, 1, 1, rope_dim] and
    # the gathered output buffer is a separate [tokens, 1, 1, rope_dim] tensor.
    source_cos = torch.randn(source_rows, 1, 1, rope_dim, dtype=source_dtype, device="npu")
    source_sin = torch.randn(source_rows, 1, 1, rope_dim, dtype=source_dtype, device="npu")
    return query_start_loc, seq_lens, positions, block_table, source_cos, source_sin


# Real inference shapes: DeepSeek V4.1 on Ascend, ``qk_rope_head_dim`` 64 and the
# full-source RoPE table in the model dtype (bf16). The prefill case uses 2048
# tokens; the padded case is a 64-token FULL-graph carrier with 37 live tokens.
# Generic cases cover a non power-of-two rope width and an fp32 source table,
# which is the eager-preparation dtype.
TEST_CASES = [
    pytest.param(6, 6, 6, 6, 64, torch.bfloat16, False, id="decode_c6_rope64_bf16"),
    pytest.param(6, 6, 6, 6, 64, torch.bfloat16, True, id="decode_c6_rope64_bf16_skip"),
    pytest.param(6, 2048, 6, 2048, 64, torch.bfloat16, False, id="prefill_rope64_bf16"),
    pytest.param(8, 64, 3, 37, 64, torch.bfloat16, False, id="full_graph_padded_rope64_bf16"),
    pytest.param(3, 33, 3, 33, 96, torch.float32, False, id="non_power_of_two_rope_fp32"),
    pytest.param(4, 17, 4, 17, 64, torch.float32, False, id="fp32_source_rope64"),
]


@pytest.mark.parametrize(
    ("num_reqs", "num_tokens", "num_actual_reqs", "num_actual_tokens", "rope_dim", "source_dtype", "skip_update"),
    TEST_CASES,
)
@torch.inference_mode()
def test_matches_eager_chain(
    num_reqs: int,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    rope_dim: int,
    source_dtype: torch.dtype,
    skip_update: bool,
) -> None:
    case = make_case(num_reqs, num_tokens, rope_dim, num_actual_reqs, source_dtype)
    expected = reference(
        case[0].cpu(),
        case[1].cpu(),
        case[2].cpu(),
        case[3].cpu(),
        case[4].cpu(),
        case[5].cpu(),
        num_reqs,
        num_actual_reqs,
        num_actual_tokens,
        torch.float32,
        skip_update,
    )
    actual = run_kernel(*case, num_reqs, num_actual_reqs, num_actual_tokens, torch.float32, skip_update)
    for got, want in zip(actual, expected):
        validate_cmp(got, want)

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("num_reqs", [1, 70])
@torch.inference_mode()
def test_multi_program_split(num_reqs: int) -> None:
    # REQ_BLOCK is 64 and TOKEN_BLOCK is 16, so both grid arms need coverage.
    num_tokens, rope_dim = 260, 64
    num_actual_reqs, num_actual_tokens = min(num_reqs, 4), 200
    case = make_case(num_reqs, num_tokens, rope_dim, num_actual_reqs)
    expected = reference(
        case[0].cpu(),
        case[1].cpu(),
        case[2].cpu(),
        case[3].cpu(),
        case[4].cpu(),
        case[5].cpu(),
        num_reqs,
        num_actual_reqs,
        num_actual_tokens,
        torch.float32,
    )
    actual = run_kernel(*case, num_reqs, num_actual_reqs, num_actual_tokens)
    for got, want in zip(actual, expected):
        validate_cmp(got, want)


@torch.inference_mode()
def test_padded_graph_tail_and_dead_requests() -> None:
    num_reqs, num_tokens, rope_dim = 8, 64, 64
    num_actual_reqs, num_actual_tokens = 3, 37
    case = make_case(num_reqs, num_tokens, rope_dim, num_actual_reqs)
    ring, complete, source_positions, cos, sin = run_kernel(*case, num_reqs, num_actual_reqs, num_actual_tokens)
    # Requests past ``num_actual_reqs`` publish no ring work: used == 0 and the
    # first block is cleared, even though the block table still holds entries.
    assert (ring[1, num_actual_reqs:] == 0).all()
    assert (ring[4, num_actual_reqs:] == 0).all()
    # Token rows past ``num_actual_tokens`` never mark a completed pair.
    assert not complete[num_actual_tokens:].any()
    assert (source_positions[num_actual_tokens:] == 0).all()
    # Incomplete rows gather the row-0 RoPE entry rather than zero-filling.
    row0_cos = case[4].cpu()[0].reshape(1, rope_dim).to(torch.float32).expand(64 - 37, rope_dim)
    row0_sin = case[5].cpu()[0].reshape(1, rope_dim).to(torch.float32).expand(64 - 37, rope_dim)
    validate_cmp(cos[num_actual_tokens:], row0_cos)
    validate_cmp(sin[num_actual_tokens:], row0_sin)


@torch.inference_mode()
def test_skip_update_publishes_no_work() -> None:
    num_reqs, num_tokens, rope_dim = 5, 37, 64
    case = make_case(num_reqs, num_tokens, rope_dim, 3)
    ring, complete, source_positions, _, _ = run_kernel(*case, num_reqs, 3, 29, torch.float32, True)
    assert (ring[1] == 0).all()
    assert (ring[4] == 0).all()
    assert not complete.any()
    assert (source_positions == 0).all()
    # The sequence-length-derived row is independent of the update switch.
    expected_ring0 = (
        case[1].cpu() - (case[0].cpu()[1 : num_reqs + 1].to(torch.int32) - case[0].cpu()[:num_reqs].to(torch.int32))
    ).clamp_min(0)
    validate_cmp(ring[0], expected_ring0)


@torch.inference_mode()
def test_empty_request_and_token_carriers_launch_nothing() -> None:
    rope_dim = 64
    ring = torch.full((5, 0), 777, dtype=torch.int32, device="npu")
    complete = torch.empty((0,), dtype=torch.bool, device="npu")
    source_positions = torch.empty((0,), dtype=torch.int64, device="npu")
    cos_output = torch.zeros((0, 1, 1, rope_dim), dtype=torch.float32, device="npu")
    sin_output = torch.zeros_like(cos_output)
    build_c2_ring_metadata(
        torch.zeros((1,), dtype=torch.int32, device="npu"),
        torch.empty((0,), dtype=torch.int32, device="npu"),
        torch.empty((0,), dtype=torch.int64, device="npu"),
        torch.empty((0, 4), dtype=torch.int32, device="npu"),
        torch.randn(1, 1, 1, rope_dim, dtype=torch.bfloat16, device="npu"),
        torch.randn(1, 1, 1, rope_dim, dtype=torch.bfloat16, device="npu"),
        0,
        0,
        0,
        0,
        skip_update=False,
        ring_metadata_output=ring,
        complete_mask_output=complete,
        source_positions_output=source_positions,
        cos_output=cos_output,
        sin_output=sin_output,
    )
    assert ring.shape == (5, 0)


@torch.inference_mode()
def test_rejects_inconsistent_output_shapes() -> None:
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([4], dtype=torch.int32, device="npu")
    positions = torch.arange(4, dtype=torch.int64, device="npu")
    block_table = torch.zeros((1, 4), dtype=torch.int32, device="npu")
    source_cos = torch.randn(4, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    source_sin = torch.randn(4, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    common = dict(
        skip_update=False,
        complete_mask_output=torch.zeros((4,), dtype=torch.bool, device="npu"),
        source_positions_output=torch.zeros((4,), dtype=torch.int64, device="npu"),
        cos_output=torch.zeros((4, 1, 1, 64), dtype=torch.float32, device="npu"),
        sin_output=torch.zeros((4, 1, 1, 64), dtype=torch.float32, device="npu"),
    )
    with pytest.raises(ValueError, match=r"ring_metadata_output must have shape \[5, num_reqs\]"):
        build_c2_ring_metadata(
            query_start_loc,
            seq_lens,
            positions,
            block_table,
            source_cos,
            source_sin,
            1,
            4,
            1,
            4,
            ring_metadata_output=torch.zeros((4, 1), dtype=torch.int32, device="npu"),
            **common,
        )
    with pytest.raises(ValueError, match="cos_output has an invalid shape"):
        build_c2_ring_metadata(
            query_start_loc,
            seq_lens,
            positions,
            block_table,
            source_cos,
            source_sin,
            1,
            4,
            1,
            4,
            ring_metadata_output=torch.zeros((5, 1), dtype=torch.int32, device="npu"),
            **{**common, "cos_output": torch.zeros((4, 1, 1, 32), dtype=torch.float32, device="npu")},
        )
    with pytest.raises(ValueError, match="source RoPE tables must have the same width"):
        build_c2_ring_metadata(
            query_start_loc,
            seq_lens,
            positions,
            block_table,
            source_cos,
            torch.randn(4, 1, 1, 32, dtype=torch.bfloat16, device="npu"),
            1,
            4,
            1,
            4,
            ring_metadata_output=torch.zeros((5, 1), dtype=torch.int32, device="npu"),
            **common,
        )


@pytest.mark.parametrize("rope_dim", [64, 128])
@torch.inference_mode()
def test_graph_replay_keeps_buffers_and_semantics(rope_dim: int) -> None:
    num_reqs, num_tokens, num_actual_reqs, num_actual_tokens = 5, 64, 3, 37
    query_start_loc, seq_lens, positions, block_table, source_cos, source_sin = make_case(
        num_reqs, num_tokens, rope_dim, num_actual_reqs
    )
    ring = torch.zeros((5, num_reqs), dtype=torch.int32, device="npu")
    complete = torch.zeros((num_tokens,), dtype=torch.bool, device="npu")
    source_positions = torch.zeros((num_tokens,), dtype=torch.int64, device="npu")
    cos_output = torch.zeros((num_tokens, 1, 1, rope_dim), dtype=torch.float32, device="npu")
    sin_output = torch.zeros_like(cos_output)

    def launch():
        build_c2_ring_metadata(
            query_start_loc,
            seq_lens,
            positions,
            block_table,
            source_cos,
            source_sin,
            num_reqs,
            num_tokens,
            num_actual_reqs,
            num_actual_tokens,
            skip_update=False,
            ring_metadata_output=ring,
            complete_mask_output=complete,
            source_positions_output=source_positions,
            cos_output=cos_output,
            sin_output=sin_output,
        )
        return ring.data_ptr(), complete.data_ptr(), source_positions.data_ptr()

    pointer = launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        launch()
    for seed in (0, 1, 2):
        fresh = make_case(num_reqs, num_tokens, rope_dim, num_actual_reqs, seed=seed)
        seq_lens.copy_(fresh[1])
        positions.copy_(fresh[2])
        block_table.copy_(fresh[3])
        source_cos.copy_(fresh[4])
        source_sin.copy_(fresh[5])
        graph.replay()
        torch.npu.synchronize()
        assert (ring.data_ptr(), complete.data_ptr(), source_positions.data_ptr()) == pointer
        expected = reference(
            query_start_loc.cpu(),
            seq_lens.cpu(),
            positions.cpu(),
            block_table.cpu(),
            source_cos.cpu(),
            source_sin.cpu(),
            num_reqs,
            num_actual_reqs,
            num_actual_tokens,
            torch.float32,
        )
        actual = (
            ring.cpu(),
            complete.cpu(),
            source_positions.cpu(),
            cos_output.cpu().reshape(num_tokens, rope_dim),
            sin_output.cpu().reshape(num_tokens, rope_dim),
        )
        for got, want in zip(actual, expected):
            validate_cmp(got, want)
