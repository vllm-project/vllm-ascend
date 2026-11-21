# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU gate for FlashKDA prefill, page guards and first MegaKDA verify.

Requires the reference FlashKDA and current MegaKDA packages. Synthetic inputs
exercise real operators; real-checkpoint V2 serving remains a separate gate.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.e2e.nightly.single_node.ops.test_mega_kda import (
    BENCHMARK_MEASURE_ROUNDS,
    BENCHMARK_WARMUP_ROUNDS,
    CONV_WIDTH,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    HIDDEN_SIZE,
    NORM_EPS,
    NUM_HEADS,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    STATE_ATOL,
    STATE_RTOL,
    _baseline_step,
    _make_attention,
)
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.ops.kda_state_copy_plan import initialize_kda_state_copy
from vllm_ascend.ops.kimi_mega_kda import load_mega_kda_operator, run_mega_recurrent_kda
from vllm_ascend.ops.kimi_prefill_fused_kda import (
    PrefillFusedKDAOperators,
    create_prefill_kda_metadata_workspace,
    load_prefill_fused_kda_operator,
    run_prefill_fused_kda,
)

BATCH = 4
SPEC_LEN = 8
GUARD_BYTES = 64


@torch.inference_mode()
@pytest.mark.parametrize("num_requests,tokens,num_heads", [(1, 219, 12), (2, 194, NUM_HEADS)])
def test_flash_kda_metadata_reuse_across_layers_and_forwards(record_property, num_requests, tokens, num_heads):
    """Real NPU gate: different layer payloads share only the schedule."""
    operator = load_prefill_fused_kda_operator()
    assert operator is not None, "Install the reference ops.flash_kda package"
    metadata_shapes = []

    def build_metadata(*args, **kwargs):
        metadata = operator.metadata(*args, **kwargs)
        metadata_shapes.append(metadata.shape)
        return metadata

    counted = PrefillFusedKDAOperators(operator.prefill, MagicMock(side_effect=build_metadata))
    num_layers = 3
    vc = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"kimi": SimpleNamespace(_prefill_fused_kda_operator=counted)}
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=num_requests, max_num_batched_tokens=tokens),
    )
    workspace = create_prefill_kda_metadata_workspace(vc, ["kimi"], torch.device("npu"))
    address = workspace.buffer.data_ptr()
    starts = torch.tensor([0, tokens] if num_requests == 1 else [0, 65, tokens], dtype=torch.int32, device="npu")
    a_log = torch.zeros(num_heads, dtype=torch.float32, device="npu")
    dt_bias = torch.zeros(num_heads, HEAD_DIM, dtype=torch.float32, device="npu")
    inputs = []
    for _ in range(num_layers):
        q, k, v, gate = (
            torch.randn(tokens, num_heads, HEAD_DIM, device="npu", dtype=torch.bfloat16) * 0.1 for _ in range(4)
        )
        beta = torch.randn(tokens, num_heads, device="npu", dtype=torch.bfloat16)
        state = torch.randn(num_requests, num_heads, HEAD_DIM, HEAD_DIM, device="npu") * 0.1
        inputs.append((q, k, v, gate, beta, state))

    def run_layer(op, args, context=None):
        return run_prefill_fused_kda(
            op,
            *args,
            starts,
            a_log,
            dt_bias,
            scale=HEAD_DIM**-0.5,
            lower_bound=GATE_LOWER_BOUND,
            metadata_context=context,
        )

    for middle in (None, None) if num_requests == 1 else (65, 129):
        # Keep the boundaries' address, but change their contents each forward.
        if middle is not None:
            starts[1] = middle
        context = workspace.new_context(starts)
        for args in inputs:
            expected = run_layer(operator, args)
            actual = run_layer(counted, args, context)
            assert context.metadata.shape == metadata_shapes[-1]
            torch.npu.synchronize()
            torch.testing.assert_close(actual[0], expected[0], rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
            torch.testing.assert_close(actual[1], expected[1], rtol=STATE_RTOL, atol=STATE_ATOL)
        assert workspace.buffer.data_ptr() == address
    assert counted.metadata.call_count == 2

    for name, reuse in (("per_layer_metadata", False), ("shared_metadata", True)):

        def run_forward(reuse=reuse):
            context = workspace.new_context(starts) if reuse else None
            for args in inputs:
                run_layer(operator, args, context)

        for _ in range(BENCHMARK_WARMUP_ROUNDS):
            run_forward()
        torch.npu.synchronize()
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(BENCHMARK_MEASURE_ROUNDS):
            run_forward()
        end.record()
        end.synchronize()
        record_property(f"{name}_{num_layers}_layers_us", start.elapsed_time(end) * 1000 / BENCHMARK_MEASURE_ROUNDS)


def _bind_cache(layer, layout):
    slots = BATCH * SPEC_LEN + 1
    conv_shape = (slots, CONV_WIDTH + SPEC_LEN - 2, 3 * NUM_HEADS * HEAD_DIM)
    state_shape = (slots, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    conv_bytes = conv_shape[1] * conv_shape[2] * 2
    state_bytes = NUM_HEADS * HEAD_DIM * HEAD_DIM * 4
    page_bytes = conv_bytes + state_bytes + GUARD_BYTES
    backing = torch.full((slots * page_bytes + 2 * GUARD_BYTES,), 37, dtype=torch.int8, device="npu")
    raw = backing[GUARD_BYTES:-GUARD_BYTES]
    conv = raw[: slots * conv_bytes].view(torch.bfloat16).view(conv_shape)
    state = raw[slots * conv_bytes : slots * (conv_bytes + state_bytes)].view(torch.float32).view(state_shape)
    if layout == "page_strided":
        conv = torch.as_strided(raw.view(torch.bfloat16), conv_shape, (page_bytes // 2, conv_shape[2], 1))
        state = torch.as_strided(
            raw[conv_bytes:].view(torch.float32),
            state_shape,
            (page_bytes // 4, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        )
    layer.kv_cache = (conv, state)
    initialize_kda_state_copy({"prefill_gate": layer}, max_selected=BATCH)
    return backing, conv_bytes, state_bytes, page_bytes


def _prefill_step(layer, qkv, gate, beta, starts, indices, flags, metadata, *, fused):
    convolved = layer._run_causal_conv1d(
        qkv,
        layer.ascend_conv1d_weight,
        layer.kv_cache[0],
        starts,
        indices,
        flags.to(torch.int32),
        run_mode=0,
    )
    q, k, v = (part.reshape(1, -1, NUM_HEADS, HEAD_DIM) for part in convolved.chunk(3, -1))
    args = (q, k, v, gate, beta, layer.kv_cache[1], indices, flags, metadata)
    if fused:
        return layer._run_prefill_fused_kda(*args, query_start_loc=starts)
    return layer._run_prefill(q, k, v, gate, beta.float().sigmoid(), *args[5:])


@pytest.mark.parametrize("layout", ["contiguous", "page_strided"])
@torch.inference_mode()
def test_flash_kda_prefill_cache_and_mega_verify(layout, record_property):
    torch.manual_seed(42)
    reference = _make_attention()
    candidate = deepcopy(reference)
    candidate._prefill_fused_kda_operator = load_prefill_fused_kda_operator()
    assert candidate._prefill_fused_kda_operator is not None, "Install the reference ops.flash_kda package"
    mega = load_mega_kda_operator()
    backing, conv_bytes, state_bytes, page_bytes = _bind_cache(candidate, layout)
    _bind_cache(reference, "contiguous")
    for actual, expected in zip(candidate.kv_cache, reference.kv_cache):
        expected.normal_(std=0.1)
        actual.copy_(expected)
    original_cache = candidate.kv_cache
    original_layouts = [(state.data_ptr(), state.stride(), state.storage_offset()) for state in original_cache]

    # Tail chunks, an empty request, stale fresh-request pages, continuation
    # and request reorder all pass through the production gather/scatter path.
    rounds = (
        ((1, 65, 129, 0), (1, 9, 17, 25), (False, True, False, False)),
        ((3, 64, 17, 2), (17, 9, 25, 1), (True, True, False, True)),
    )
    for lengths, slots, reusable in rounds:
        starts_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        starts = starts_cpu.to("npu")
        indices = torch.tensor(slots, device="npu", dtype=torch.int32)
        flags = torch.tensor(reusable, device="npu", dtype=torch.bool)
        metadata = _build_non_spec_chunked_prefill_metadata(None, starts_cpu, starts.device, False)
        tokens = sum(lengths)
        qkv = torch.randn(tokens, 3 * NUM_HEADS * HEAD_DIM, device="npu", dtype=torch.bfloat16) * 0.1
        gate = torch.randn(1, tokens, NUM_HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
        beta = torch.randn(1, tokens, NUM_HEADS, device="npu", dtype=torch.bfloat16)
        before = backing.clone()
        expected = _prefill_step(reference, qkv, gate, beta, starts, indices, flags, metadata, fused=False)
        actual = _prefill_step(candidate, qkv, gate, beta, starts, indices, flags, metadata, fused=True)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        for actual_state, expected_state in zip(candidate.kv_cache, reference.kv_cache):
            torch.testing.assert_close(actual_state, expected_state, rtol=STATE_RTOL, atol=STATE_ATOL)
        untouched = torch.ones_like(backing, dtype=torch.bool)
        for length, slot in zip(lengths, slots):
            conv_offset = GUARD_BYTES + slot * (page_bytes if layout == "page_strided" else conv_bytes)
            untouched[conv_offset : conv_offset + conv_bytes] = False
            if length:
                state_offset = GUARD_BYTES + (
                    slot * page_bytes + conv_bytes
                    if layout == "page_strided"
                    else candidate.kv_cache[0].shape[0] * conv_bytes + slot * state_bytes
                )
                untouched[state_offset : state_offset + state_bytes] = False
        torch.testing.assert_close(backing[untouched], before[untouched], rtol=0, atol=0)

    # First verify must consume FlashKDA's state through the original V2 cache.
    indices = torch.arange(1, BATCH * SPEC_LEN + 1, device="npu", dtype=torch.int32).reshape(BATCH, SPEC_LEN)
    hidden = torch.randn(BATCH, SPEC_LEN, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16)
    accepted = torch.ones(BATCH, device="npu", dtype=torch.int32)
    expected = _baseline_step(reference, hidden.flatten(0, 1), reference.kv_cache, indices, accepted)
    actual = run_mega_recurrent_kda(
        hidden,
        candidate._build_mega_weights(),
        *candidate.kv_cache,
        indices[:, 0],
        indices.flatten(),
        torch.full_like(accepted, SPEC_LEN),
        accepted,
        candidate.A_log,
        candidate.dt_bias,
        HEAD_DIM**-0.5,
        GATE_LOWER_BOUND,
        NORM_EPS,
        operator=mega,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    for actual_state, expected_state in zip(candidate.kv_cache, reference.kv_cache):
        torch.testing.assert_close(actual_state, expected_state, rtol=STATE_RTOL, atol=STATE_ATOL)
    assert candidate.kv_cache is original_cache
    assert [(state.data_ptr(), state.stride(), state.storage_offset()) for state in original_cache] == original_layouts

    # Record end-to-end Conv + gather + FlashKDA + scatter latency. Serving
    # throughput and hardware-specific performance gates need deployment A/B.
    flags.zero_()
    for name, layer, fused in (("ascendc", reference, False), ("flash_kda", candidate, True)):
        for _ in range(BENCHMARK_WARMUP_ROUNDS):
            _prefill_step(layer, qkv, gate, beta, starts, indices[:, 0], flags, metadata, fused=fused)
        torch.npu.synchronize()
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(BENCHMARK_MEASURE_ROUNDS):
            _prefill_step(layer, qkv, gate, beta, starts, indices[:, 0], flags, metadata, fused=fused)
        end.record()
        end.synchronize()
        record_property(f"{name}_{layout}_prefill_us", start.elapsed_time(end) * 1000 / BENCHMARK_MEASURE_ROUNDS)
