# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.candidate_pages import store_candidate_pages


def _reference(cache, initial, correction, key, decay, destinations, lengths, active):
    for request in range(len(active)):
        if not active[request]:
            continue
        state = initial[request]
        for step in range(min(int(lengths[request]), correction.size(2))):
            u = correction[request, :, step]
            k = key[request, :, step].float()
            state = state * decay[request, :, step, None, :] + u[:, :, None] * k[:, None, :]
            page = int(destinations[request, step])
            if 0 < page < cache.size(0):
                cache[page].copy_(state)


@pytest.mark.parametrize("window", [1, 8], ids=["decode", "dspark7_verify"])
def test_candidate_pages_preserve_padded_storage_and_replay(window):
    batch, heads, dim, pages = 4, 12, 128, 14
    shape = (pages, heads, dim, dim)
    page_stride = heads * dim * dim + 32
    strides = (page_stride, dim * dim, dim, 1)
    offset = 16
    pristine = torch.full((offset + pages * page_stride,), -321.0)
    backing = pristine.to("npu")
    cache = backing.as_strided(shape, strides, offset)
    rng = torch.Generator().manual_seed(15)
    state_shape = (batch, heads, dim, dim)
    factor_shape = (batch, heads, window, dim)
    # These dyadic inputs have exactly representable FP32 updates through W=8.
    initial = torch.randint(-6, 7, state_shape, generator=rng).float() / 8
    correction = torch.randint(-2, 3, factor_shape, generator=rng).float() / 4
    key = torch.randint(-2, 3, factor_shape, generator=rng).half() / 4
    decay = torch.randint(1, 3, factor_shape, generator=rng).float() / 2
    factors = (initial, correction, key, decay)
    device_factors = [value.to("npu") for value in factors]

    slots = torch.tensor(
        [[1, 0, 2, -1, 3, pages, 4, 5], [8, 7, 6, 12, 12, 12, 12, 12], [9] * 8, [10] * 8],
        dtype=torch.int32,
    )[:, :window]
    lens = torch.tensor([window, min(3, window), window, 0], dtype=torch.int32)
    flags = torch.tensor([True, True, False, True])
    # Keep pointers stable across capture. Both metadata axes can be strided.
    destinations = torch.empty((batch, 2 * window + 1), dtype=torch.int32, device="npu")[:, 1::2]
    lengths = torch.empty(2 * batch, dtype=torch.int32, device="npu")[1::2]
    active = torch.empty(2 * batch, dtype=torch.bool, device="npu")[1::2]

    def install(slots_cpu, lengths_cpu, active_cpu):
        destinations.copy_(slots_cpu)
        lengths.copy_(lengths_cpu)
        active.copy_(active_cpu)
        backing.copy_(pristine)

    def check(slots_cpu, lengths_cpu, active_cpu):
        expected = pristine.clone()
        expected_cache = expected.as_strided(shape, strides, offset)
        _reference(expected_cache, *factors, slots_cpu, lengths_cpu, active_cpu)
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
        for actual, source in zip(device_factors, factors):
            torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)

    install(slots, lens, flags)
    store_candidate_pages(cache, *device_factors, destinations, lengths, active)
    check(slots, lens, flags)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        store_candidate_pages(cache, *device_factors, destinations, lengths, active)

    # Move the metadata to different requests and route writes to different pages.
    new_slots = torch.where((slots > 0) & (slots < pages), pages - slots, slots).roll(1, dims=0)
    cases = [(slots, lens, flags), (new_slots, lens.roll(1), flags.roll(1))]
    for replay_slots, replay_lengths, replay_active in cases:
        install(replay_slots, replay_lengths, replay_active)
        graph.replay()
        check(replay_slots, replay_lengths, replay_active)
