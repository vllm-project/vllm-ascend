# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.packed_cache_slot_mapping import build_packed_cache_slot_mapping

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


def reference(
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    skip_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the original eager slot-mapping chain."""
    slots = slots.to(torch.int64)
    if compress_ratio == 2:
        complete = (slots >= 0) & ((slots + 1).remainder(compress_ratio) == 0)
        slots = torch.where(complete, slots // compress_ratio, -1)
    valid = slots >= 0
    if compress_ratio == 2:
        if skip_update:
            valid.zero_()
        else:
            valid_end = min(int(query_start_loc[num_actual_reqs]), num_actual_tokens)
            valid &= torch.arange(slots.shape[0]) < valid_end
            valid &= positions.remainder(2) == 1
    clamped = slots.clamp_min(0)
    pages = torch.where(valid, clamped // page_size, torch.full_like(clamped, -1))
    rows = torch.where(valid, clamped % page_size, torch.full_like(clamped, -1))
    flat = torch.where(valid, slots, torch.full_like(slots, -1))
    return torch.stack((pages, rows), dim=1), flat


def validate_cmp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Pages, rows and flat slots are integer indices: exact comparison is the
    # applicable criterion for this operator type.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def run_kernel(
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    skip_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = slots.shape[0]
    coordinates = torch.full((num_tokens, 2), 31415, dtype=torch.int64, device="npu")
    flat = torch.full((num_tokens,), 31415, dtype=torch.int64, device="npu")
    actual_coordinates, actual_flat = build_packed_cache_slot_mapping(
        slots,
        positions,
        query_start_loc,
        num_tokens,
        num_actual_reqs,
        num_actual_tokens,
        page_size,
        compress_ratio,
        skip_update=skip_update,
        coordinates_output=coordinates,
        flat_output=flat,
    )
    # The launch must reuse the supplied graph-stable buffers.
    assert actual_coordinates.data_ptr() == coordinates.data_ptr()
    assert actual_flat.data_ptr() == flat.data_ptr()
    return actual_coordinates.cpu(), actual_flat.cpu()


def make_case(
    tokens: int, requests: int, max_slot: int, seed: int = 41
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    slots = torch.randint(-1, max_slot, (tokens,), dtype=torch.int32, device="npu")
    positions = torch.randint(-1, 2 * tokens + 2, (tokens,), dtype=torch.int64, device="npu")
    # Uneven request boundaries: the only entry the kernel reads is the one at
    # ``num_actual_reqs``, so the split itself is intentionally non-uniform.
    cuts = [0]
    for index in range(requests):
        remaining = tokens - cuts[-1]
        cuts.append(cuts[-1] + (remaining + requests - index - 1) // (requests - index))
    return slots, positions, torch.tensor(cuts, dtype=torch.int32, device="npu")


# Real inference shapes: DeepSeek V4.1 on Ascend, storage block size 128 and
# ``qk_rope_head_dim`` 64. ``page_size`` is the KV cache storage block size
# reported by ``get_storage_block_size``.
#   decode              : 6 concurrent requests -> 6 tokens
#   decode upper bound  : 256 tokens (the occupancy sweep of the PR)
#   prefill             : max_num_batched_tokens -> 8192 tokens
# plus the boundaries the kernel branches on: single token, grid saturation at
# 320 tokens (BLOCK_SIZE 8 saturates 40 vector cores), the ratio-1 SWA plane
# (positions are never dereferenced), a non power-of-two page, and a page
# boundary case.
TEST_CASES = [
    pytest.param(6, 2, 128, id="decode_c6_ratio2_page128"),
    pytest.param(256, 2, 128, id="decode_upper_ratio2_page128"),
    pytest.param(8192, 2, 128, id="prefill_ratio2_page128"),
    pytest.param(1, 2, 128, id="single_token_ratio2"),
    pytest.param(320, 2, 128, id="grid_saturation_ratio2"),
    pytest.param(37, 1, 128, id="swa_ratio1_page128"),
    pytest.param(37, 2, 3, id="non_power_of_two_page"),
    pytest.param(8, 2, 16, id="page_boundary"),
]


@pytest.mark.parametrize(("num_tokens", "compress_ratio", "page_size"), TEST_CASES)
@torch.inference_mode()
def test_matches_eager_chain(num_tokens: int, compress_ratio: int, page_size: int) -> None:
    requests = min(num_tokens, 6)
    slots, positions, query_start_loc = make_case(num_tokens, requests, 1 << 20)
    expected_coordinates, expected_flat = reference(
        slots.cpu(), positions.cpu(), query_start_loc.cpu(), requests, num_tokens, page_size, compress_ratio
    )
    actual_coordinates, actual_flat = run_kernel(
        slots, positions, query_start_loc, requests, num_tokens, page_size, compress_ratio
    )
    validate_cmp(actual_coordinates, expected_coordinates)
    validate_cmp(actual_flat, expected_flat)

    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_ratio_one_without_positions() -> None:
    slots, _, query_start_loc = make_case(37, 3, 1 << 16)
    expected = reference(slots.cpu(), slots.cpu(), query_start_loc.cpu(), 3, 37, 128, 1)
    actual = run_kernel(slots, slots, query_start_loc, 3, 37, 128, 1)
    for got, want in zip(actual, expected):
        validate_cmp(got, want)


@torch.inference_mode()
def test_ratio_one_masks_stale_positive_tail() -> None:
    slots = torch.arange(1, 17, dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor([0, 16], dtype=torch.int32, device="npu")
    actual = run_kernel(slots, slots, query_start_loc, 1, 8, 128, 1)
    # The old ratio-1 eager path relies on upstream PAD slots; fused also clips valid_end.
    padded_slots = slots.cpu()
    padded_slots[8:] = -1
    expected = reference(padded_slots, padded_slots, query_start_loc.cpu(), 1, 8, 128, 1)
    for got, want in zip(actual, expected):
        validate_cmp(got, want)


@pytest.mark.parametrize("page_size", [16, 128])
@torch.inference_mode()
def test_padded_graph_tail_stays_invalid(page_size: int) -> None:
    # A captured carrier keeps slot ids from an earlier replay in its padding;
    # those rows must never address a packed cache plane.
    tokens, actual_tokens, requests = 64, 37, 4
    slots, positions, query_start_loc = make_case(tokens, requests, 1 << 20)
    slots[actual_tokens:] = torch.randint(0, 1 << 20, (tokens - actual_tokens,), dtype=torch.int32, device="npu")
    positions[actual_tokens:] = torch.arange(
        2 * actual_tokens + 2, 2 * actual_tokens + 2 + tokens - actual_tokens, dtype=torch.int64, device="npu"
    )
    expected_coordinates, expected_flat = reference(
        slots.cpu(), positions.cpu(), query_start_loc.cpu(), requests, actual_tokens, page_size, 2
    )
    actual_coordinates, actual_flat = run_kernel(
        slots, positions, query_start_loc, requests, actual_tokens, page_size, 2
    )
    validate_cmp(actual_coordinates, expected_coordinates)
    validate_cmp(actual_flat, expected_flat)
    assert (actual_coordinates[actual_tokens:] == -1).all()
    assert (actual_flat[actual_tokens:] == -1).all()


@torch.inference_mode()
def test_ratio_two_writes_completed_pairs_only() -> None:
    page_size = 128
    slots = torch.tensor([0, 1, 2, 3, -1, 5, 6, 7, 9, 11], dtype=torch.int32, device="npu")
    positions = torch.tensor([0, 1, 2, 3, 4, 6, 7, 9, 8, 11], dtype=torch.int64, device="npu")
    query_start_loc = torch.tensor([0, 4, 10], dtype=torch.int32, device="npu")
    expected_coordinates, expected_flat = reference(
        slots.cpu(), positions.cpu(), query_start_loc.cpu(), 2, 10, page_size, 2
    )
    actual_coordinates, actual_flat = run_kernel(slots, positions, query_start_loc, 2, 10, page_size, 2)
    validate_cmp(actual_coordinates, expected_coordinates)
    validate_cmp(actual_flat, expected_flat)
    # Only completed pairs (odd slot and odd position) publish a cache row.
    assert (actual_flat[[1, 3, 7, 9]] != -1).all()
    assert (actual_flat[[0, 2, 4, 5, 6, 8]] == -1).all()


@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_skip_update_only_affects_ratio_two(compress_ratio: int) -> None:
    page_size = 16
    slots, positions, query_start_loc = make_case(37, 3, 1 << 16)
    expected_coordinates, expected_flat = reference(
        slots.cpu(), positions.cpu(), query_start_loc.cpu(), 3, 37, page_size, compress_ratio, skip_update=True
    )
    actual_coordinates, actual_flat = run_kernel(
        slots, positions, query_start_loc, 3, 37, page_size, compress_ratio, skip_update=True
    )
    validate_cmp(actual_coordinates, expected_coordinates)
    validate_cmp(actual_flat, expected_flat)
    if compress_ratio == 2:
        assert (actual_flat == -1).all()
        assert (actual_coordinates == -1).all()


@pytest.mark.parametrize("page_size", [16, 128, 3])
@torch.inference_mode()
def test_wide_slot_ids_keep_64_bit_pages(page_size: int) -> None:
    # Slots above 2^40 must not wrap the page/row split.
    slots = torch.tensor([1 << 40, (1 << 41) + 7, (1 << 44) + 12345, -1], dtype=torch.int64, device="npu")
    positions = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device="npu")
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    for compress_ratio in (1, 2):
        expected_coordinates, expected_flat = reference(
            slots.cpu(), positions.cpu(), query_start_loc.cpu(), 1, 4, page_size, compress_ratio
        )
        actual_coordinates, actual_flat = run_kernel(slots, positions, query_start_loc, 1, 4, page_size, compress_ratio)
        validate_cmp(actual_coordinates, expected_coordinates)
        validate_cmp(actual_flat, expected_flat)


@torch.inference_mode()
def test_noncontiguous_inputs_use_element_strides() -> None:
    page_size = 64
    tokens, requests = 19, 4
    stride_slots, stride_positions, stride_query_start_loc = make_case(tokens, requests, 1 << 18)
    padded_slots = torch.full((2 * tokens,), -1, dtype=torch.int32, device="npu")
    padded_positions = torch.zeros((2 * tokens,), dtype=torch.int64, device="npu")
    padded_query_start_loc = torch.zeros((2 * (requests + 1),), dtype=torch.int32, device="npu")
    padded_slots[::2] = stride_slots
    padded_positions[::2] = stride_positions
    padded_query_start_loc[::2] = stride_query_start_loc
    slots = padded_slots[::2]
    positions = padded_positions[::2]
    query_start_loc = padded_query_start_loc[::2]
    assert not slots.is_contiguous() and not positions.is_contiguous()

    for compress_ratio in (1, 2):
        expected_coordinates, expected_flat = reference(
            slots.cpu(), positions.cpu(), query_start_loc.cpu(), requests, tokens, page_size, compress_ratio
        )
        actual_coordinates, actual_flat = run_kernel(
            slots, positions, query_start_loc, requests, tokens, page_size, compress_ratio
        )
        validate_cmp(actual_coordinates, expected_coordinates)
        validate_cmp(actual_flat, expected_flat)


@torch.inference_mode()
def test_empty_token_carrier_launches_nothing() -> None:
    page_size = 128
    slots = torch.empty((0,), dtype=torch.int32, device="npu")
    positions = torch.empty((0,), dtype=torch.int64, device="npu")
    query_start_loc = torch.zeros((5,), dtype=torch.int32, device="npu")
    for compress_ratio in (1, 2):
        actual_coordinates, actual_flat = run_kernel(slots, positions, query_start_loc, 4, 0, page_size, compress_ratio)
        assert actual_coordinates.shape == (0, 2)
        assert actual_flat.shape == (0,)
    # A launched kernel would overwrite these sentinels.
    coordinates = torch.full((8, 2), 31415, dtype=torch.int64, device="npu")
    flat = torch.full((8,), 31415, dtype=torch.int64, device="npu")
    build_packed_cache_slot_mapping(
        torch.empty((0,), dtype=torch.int32, device="npu"),
        torch.empty((0,), dtype=torch.int64, device="npu"),
        query_start_loc,
        0,
        4,
        0,
        page_size,
        2,
        coordinates_output=coordinates,
        flat_output=flat,
    )
    torch.npu.synchronize()
    assert (coordinates == 31415).all()
    assert (flat == 31415).all()


def test_rejects_unsupported_compress_ratio() -> None:
    slots = torch.zeros((4,), dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="compress_ratio must be 1 or 2"):
        build_packed_cache_slot_mapping(
            slots,
            slots,
            query_start_loc,
            4,
            1,
            4,
            128,
            4,
            coordinates_output=torch.zeros((4, 2), dtype=torch.int64, device="npu"),
            flat_output=torch.zeros((4,), dtype=torch.int64, device="npu"),
        )


@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_graph_replay_keeps_buffers_and_semantics(compress_ratio: int) -> None:
    page_size = 128
    tokens, actual_tokens, requests = 64, 37, 4
    slots, positions, query_start_loc = make_case(tokens, requests, 1 << 20)
    coordinates = torch.zeros((tokens, 2), dtype=torch.int64, device="npu")
    flat = torch.zeros((tokens,), dtype=torch.int64, device="npu")

    def launch():
        return build_packed_cache_slot_mapping(
            slots,
            positions,
            query_start_loc,
            tokens,
            requests,
            actual_tokens,
            page_size,
            compress_ratio,
            coordinates_output=coordinates,
            flat_output=flat,
        )

    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured_coordinates, captured_flat = launch()
    pointer = (captured_coordinates.data_ptr(), captured_flat.data_ptr())
    for seed in (0, 1, 2):
        fresh_slots, fresh_positions, _ = make_case(tokens, requests, 1 << 20, seed=seed)
        if compress_ratio == 1:
            fresh_slots[actual_tokens:] = -1
        slots.copy_(fresh_slots)
        positions.copy_(fresh_positions)
        graph.replay()
        torch.npu.synchronize()
        assert (captured_coordinates.data_ptr(), captured_flat.data_ptr()) == pointer
        assert captured_coordinates.data_ptr() == coordinates.data_ptr()
        expected_coordinates, expected_flat = reference(
            slots.cpu(), positions.cpu(), query_start_loc.cpu(), requests, actual_tokens, page_size, compress_ratio
        )
        validate_cmp(captured_coordinates.cpu(), expected_coordinates)
        validate_cmp(captured_flat.cpu(), expected_flat)
