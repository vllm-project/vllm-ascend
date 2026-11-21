# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU release gate: PR #6 MegaKDA versus the current AscendC KDA path.

Uses real kernels and independent caches, including prefill, first Verify,
partial acceptance, request reordering and an intervening unfused round.
This synthetic-weight gate does not replace real-checkpoint serving tests.
Missing/incompatible MegaKDA packages fail the test instead of skipping it.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.ops.kda import run_chunk_kda
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention
from vllm_ascend.ops.kimi_mega_kda import (
    load_mega_kda_operator,
    run_mega_recurrent_kda,
)
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated

# Production per-rank shape of the PR #6 TP=8 deployment.
HIDDEN_SIZE = 7168
NUM_HEADS = 6
HEAD_DIM = 128
CONV_WIDTH = 4
PREFILL_LENGTH = 5
NORM_EPS = 1e-6
GATE_LOWER_BOUND = -5.0
# Fixed BF16 gates; never widen these to make an incompatible binary pass.
OUTPUT_RTOL = 0.02
OUTPUT_ATOL = 0.02
STATE_RTOL = 0.01
STATE_ATOL = 0.001
BENCHMARK_WARMUP_ROUNDS = 3
BENCHMARK_MEASURE_ROUNDS = 20


def _make_attention():
    layer = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(layer)
    layer.hidden_size = HIDDEN_SIZE
    layer.local_num_heads = NUM_HEADS
    layer.head_dim = HEAD_DIM
    layer.local_projection_size = NUM_HEADS * HEAD_DIM
    layer.conv_size = CONV_WIDTH
    layer.gate_lower_bound = GATE_LOWER_BOUND
    layer.uses_mixed_projection = False
    projection = layer.local_projection_size
    layer.in_proj_qkvgfab = nn.Linear(HIDDEN_SIZE, 4 * projection + HEAD_DIM + NUM_HEADS, bias=False)
    layer.f_b_proj = nn.Linear(HEAD_DIM, projection, bias=False)
    layer.o_proj = nn.Linear(projection, HIDDEN_SIZE, bias=False)
    layer.register_parameter("ascend_conv1d_weight", nn.Parameter(torch.randn(CONV_WIDTH, 3 * projection) * 0.1))
    layer.register_parameter("A_log", nn.Parameter(torch.zeros(NUM_HEADS)))
    layer.register_parameter("dt_bias", nn.Parameter(torch.zeros(NUM_HEADS, HEAD_DIM)))
    layer.to(device="npu", dtype=torch.bfloat16)
    # Match the real parameter dtypes; the packed Conv activation weight is BF16.
    layer.A_log.data = layer.A_log.float()
    layer.dt_bias.data = layer.dt_bias.float()
    layer.o_norm = SimpleNamespace(weight=torch.ones(HEAD_DIM, device="npu"), eps=NORM_EPS)
    return layer


def _project(layer, hidden):
    projection = layer.local_projection_size
    qkv, gate, f_a, beta = F.linear(hidden, layer.in_proj_qkvgfab.weight).split(
        (3 * projection, projection, HEAD_DIM, NUM_HEADS), dim=-1
    )
    raw_gate = F.linear(f_a, layer.f_b_proj.weight).reshape(1, -1, NUM_HEADS, HEAD_DIM)
    return qkv, raw_gate, beta.float().sigmoid().unsqueeze(0), gate.reshape(-1, NUM_HEADS, HEAD_DIM)


def _baseline_step(layer, hidden, cache, indices, accepted):
    batch, spec_len = indices.shape
    qkv, raw_gate, beta, gate = _project(layer, hidden)
    starts = torch.arange(batch + 1, device="npu", dtype=torch.int32) * spec_len
    qkv = layer._run_causal_conv1d(
        qkv,
        layer.ascend_conv1d_weight,
        cache[0],
        starts,
        indices[:, 0],
        None,
        run_mode=1,
        max_query_len=spec_len,
        num_accepted_tokens=accepted,
    )
    q, k, v = (part.reshape(1, -1, NUM_HEADS, HEAD_DIM) for part in qkv.chunk(3, -1))
    output = layer._run_recurrent(q, k, v, raw_gate, beta, cache[1], starts, indices, num_accepted_tokens=accepted)
    normalized = rms_norm_gated(
        output.reshape(-1, NUM_HEADS, HEAD_DIM), gate, layer.o_norm.weight, None, "sigmoid", eps=NORM_EPS
    )
    return F.linear(normalized.reshape(-1, layer.local_projection_size), layer.o_proj.weight)


def _prefill(layer, batch, cache, base_indices):
    hidden = torch.randn(batch * PREFILL_LENGTH, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16)
    qkv, raw_gate, beta, _ = _project(layer, hidden)
    starts_cpu = tuple(index * PREFILL_LENGTH for index in range(batch + 1))
    starts = torch.tensor(starts_cpu, device="npu", dtype=torch.int32)
    qkv = layer._run_causal_conv1d(
        qkv,
        layer.ascend_conv1d_weight,
        cache[0],
        starts,
        base_indices,
        torch.zeros(batch, device="npu", dtype=torch.int32),
        run_mode=0,
    )
    q, k, v = (part.reshape(1, -1, NUM_HEADS, HEAD_DIM) for part in qkv.chunk(3, -1))
    initial = torch.zeros(batch, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="npu", dtype=torch.float32)
    chunks = tuple(value for index in range(batch) for value in (index, 0))
    _, final = run_chunk_kda(
        q, k, v, raw_gate, beta, initial, starts_cpu, chunks, layer.A_log, layer.dt_bias, lower_bound=GATE_LOWER_BOUND
    )
    cache[1].index_copy_(0, base_indices.long(), final)


@pytest.mark.parametrize("batch,spec_len,cache_spec_len", [(8, 1, 1), (8, 1, 8), (8, 8, 8), (16, 8, 8)])
@torch.inference_mode()
def test_mega_kda_output_and_cache_lifecycle(batch, spec_len, cache_spec_len, record_property):
    torch.manual_seed(42)
    operator = load_mega_kda_operator()
    layer = _make_attention()
    weights = layer._build_mega_weights()
    num_slots = batch * cache_spec_len + 1
    reference_cache = (
        torch.zeros(
            num_slots, CONV_WIDTH + cache_spec_len - 2, 3 * NUM_HEADS * HEAD_DIM, device="npu", dtype=torch.bfloat16
        ),
        torch.zeros(num_slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="npu", dtype=torch.float32),
    )
    indices = torch.arange(1, num_slots, device="npu", dtype=torch.int32).reshape(batch, cache_spec_len)
    indices = indices[:, :spec_len].contiguous()
    _prefill(layer, batch, reference_cache, indices[:, 0])
    mega_cache = tuple(state.clone() for state in reference_cache)
    # Verify -> partial acceptance -> unfused fallback -> Mega again.
    for round_index in range(4):
        indices = indices.flip(0).contiguous()
        hidden = torch.randn(batch * spec_len, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16)
        accepted = torch.ones(batch, device="npu", dtype=torch.int32)
        if round_index > 0:
            accepted = 1 + torch.arange(batch, device="npu", dtype=torch.int32) % spec_len
        expected = _baseline_step(layer, hidden, reference_cache, indices, accepted)
        if round_index == 2:
            actual = _baseline_step(layer, hidden, mega_cache, indices, accepted)
        else:
            conv_accepted = torch.full_like(accepted, spec_len) if round_index == 0 else accepted
            actual = run_mega_recurrent_kda(
                hidden.reshape(batch, spec_len, HIDDEN_SIZE),
                weights,
                *mega_cache,
                indices[:, 0],
                indices.flatten(),
                conv_accepted,
                accepted,
                layer.A_log,
                layer.dt_bias,
                HEAD_DIM**-0.5,
                GATE_LOWER_BOUND,
                NORM_EPS,
                operator=operator,
            )
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        # Check the entire caches, including untouched slots and reserved block 0.
        torch.testing.assert_close(mega_cache[0], reference_cache[0], rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        torch.testing.assert_close(mega_cache[1], reference_cache[1], rtol=STATE_RTOL, atol=STATE_ATOL)

    # Timing is recorded for comparison, not claimed as a hardware-independent
    # speedup guarantee. Server throughput/TP/SP need separate real-model A/B.
    def unfused():
        return _baseline_step(layer, hidden, reference_cache, indices, accepted)

    def fused():
        return run_mega_recurrent_kda(
            hidden.reshape(batch, spec_len, HIDDEN_SIZE),
            weights,
            *mega_cache,
            indices[:, 0],
            indices.flatten(),
            accepted,
            accepted,
            layer.A_log,
            layer.dt_bias,
            HEAD_DIM**-0.5,
            GATE_LOWER_BOUND,
            NORM_EPS,
            operator=operator,
        )

    for name, operation in (("ascendc", unfused), ("mega", fused)):
        for _ in range(BENCHMARK_WARMUP_ROUNDS):
            operation()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(BENCHMARK_MEASURE_ROUNDS):
            operation()
        end.record()
        end.synchronize()
        latency_us = start.elapsed_time(end) * 1000 / BENCHMARK_MEASURE_ROUNDS
        record_property(f"{name}_latency_us", latency_us)
        print(f"{name}: B={batch}, S={spec_len}, latency={latency_us:.2f}us")


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "page_strided"])
@torch.inference_mode()
def test_mega_kda_v2_original_cache_graph_lifecycle(capture, layout, record_property):
    """Real Mega ABI receives the original cache, with no adapter or copies.

    Graph replay changes indices, acceptance, input and padding in place.
    CPU contracts separately exercise V2's metadata builder and planner;
    real-checkpoint V2 serving remains a required integration gate.
    """
    torch.manual_seed(42)
    batch, spec_len = 8, 8
    operator = load_mega_kda_operator()
    layer = _make_attention()
    weights = layer._build_mega_weights()
    num_slots = batch * spec_len + 1
    conv_shape = (CONV_WIDTH + spec_len - 2, 3 * NUM_HEADS * HEAD_DIM)
    recurrent_shape = (NUM_HEADS, HEAD_DIM, HEAD_DIM)
    reference_cache = (
        torch.zeros(num_slots, *conv_shape, device="npu", dtype=torch.bfloat16),
        torch.zeros(num_slots, *recurrent_shape, device="npu", dtype=torch.float32),
    )
    conv_bytes = reference_cache[0][0].numel() * 2
    recurrent_bytes = reference_cache[1][0].numel() * 4
    guard_bytes = 64
    page_bytes = conv_bytes + recurrent_bytes + guard_bytes
    backing = torch.full((num_slots * page_bytes + 2 * guard_bytes,), 37, dtype=torch.int8, device="npu")
    raw = backing[guard_bytes:-guard_bytes]
    conv = torch.as_strided(
        raw.view(torch.bfloat16), reference_cache[0].shape, (page_bytes // 2, *reference_cache[0].stride()[1:])
    )
    recurrent = torch.as_strided(
        raw[conv_bytes:].view(torch.float32),
        reference_cache[1].shape,
        (page_bytes // 4, *reference_cache[1].stride()[1:]),
    )
    if layout == "contiguous":
        # Disjoint dense arrays in the same disposable allocation. This
        # controls for an installed binary that only accepts dense state.
        conv = raw[: num_slots * conv_bytes].view(torch.bfloat16).view_as(reference_cache[0])
        recurrent = raw[num_slots * conv_bytes : num_slots * (conv_bytes + recurrent_bytes)]
        recurrent = recurrent.view(torch.float32).view_as(reference_cache[1])
    layer.kv_cache = (conv, recurrent)
    original_cache = layer.kv_cache
    original_layouts = [(t.data_ptr(), t.stride(), t.storage_offset()) for t in original_cache]
    request_indices = torch.arange(1, num_slots, device="npu", dtype=torch.int32).reshape(batch, spec_len)
    _prefill(layer, batch, reference_cache, request_indices[:, 0])
    for target, source in zip(layer.kv_cache, reference_cache):
        target.copy_(source)
    initial_backing = backing.clone()
    # All physical destinations are NULL during capture/warmup. The external
    # operator must preserve that sentinel and the shared page guards.
    indices = torch.zeros_like(request_indices)
    conv_indices = torch.zeros(batch, device="npu", dtype=torch.int32)
    hidden = torch.zeros(batch, spec_len, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16)
    accepted = torch.ones(batch, device="npu", dtype=torch.int32)
    conv_accepted = torch.full_like(accepted, spec_len)

    def fused():
        output = run_mega_recurrent_kda(
            hidden,
            weights,
            conv,
            recurrent,
            conv_indices,
            indices.flatten(),
            conv_accepted,
            accepted,
            layer.A_log,
            layer.dt_bias,
            HEAD_DIM**-0.5,
            GATE_LOWER_BOUND,
            NORM_EPS,
            operator=operator,
        )
        return output

    fused()
    torch.npu.synchronize()
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured_output = fused()
    torch.testing.assert_close(backing, initial_backing, rtol=0, atol=0)
    for round_index in range(4):
        order = torch.arange(batch - 1, -1, -1, device="npu") if round_index % 2 else torch.arange(batch, device="npu")
        current = request_indices.index_select(0, order)
        live = batch - 1 if round_index == 1 else batch
        indices.copy_(current)
        indices[live:].zero_()
        conv_indices.copy_(indices[:, 0])
        hidden.normal_()
        accepted.copy_(1 + torch.arange(batch, device="npu", dtype=torch.int32) % spec_len)
        if round_index == 0:
            accepted.fill_(1)
        conv_accepted.copy_(torch.full_like(accepted, spec_len) if round_index == 0 else accepted)
        before = backing.clone()
        expected = _baseline_step(
            layer, hidden[:live].reshape(-1, HIDDEN_SIZE), reference_cache, current[:live], accepted[:live]
        )
        if round_index == 2:
            # A native round between replays must update the very same cache.
            actual = _baseline_step(layer, hidden.reshape(-1, HIDDEN_SIZE), layer.kv_cache, current, accepted)
        elif capture:
            graph.replay()
            actual = captured_output
        else:
            actual = fused()
        torch.npu.synchronize()
        torch.testing.assert_close(actual[: live * spec_len], expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        torch.testing.assert_close(conv, reference_cache[0], rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        torch.testing.assert_close(recurrent, reference_cache[1], rtol=STATE_RTOL, atol=STATE_ATOL)
        # Compare all untouched bytes exactly, including payloads of requests
        # omitted this round, page guards, null slot and both backing guards.
        untouched = torch.ones_like(backing, dtype=torch.bool)
        for slot in current[:live, 0].cpu().tolist():
            start = guard_bytes + slot * (page_bytes if layout == "page_strided" else conv_bytes)
            untouched[start : start + conv_bytes] = False
        for slot in current[:live].flatten().cpu().tolist():
            start = guard_bytes + (
                slot * page_bytes + conv_bytes
                if layout == "page_strided"
                else num_slots * conv_bytes + slot * recurrent_bytes
            )
            untouched[start : start + recurrent_bytes] = False
        torch.testing.assert_close(backing[untouched], before[untouched], rtol=0, atol=0)
        assert layer.kv_cache is original_cache
        assert [(t.data_ptr(), t.stride(), t.storage_offset()) for t in original_cache] == original_layouts
    record_property("v2_cache_layout", layout)
    record_property("graph_replay", capture)

    def native():
        return _baseline_step(layer, hidden.reshape(-1, HIDDEN_SIZE), layer.kv_cache, indices, accepted)

    if capture:
        native()
        torch.npu.synchronize()
        native_graph = torch.npu.NPUGraph()
        with torch.npu.graph(native_graph):
            native()
    for name, operation in (
        ("v2_native", native_graph.replay if capture else native),
        ("v2_mega_direct_cache", graph.replay if capture else fused),
    ):
        for _ in range(BENCHMARK_WARMUP_ROUNDS):
            operation()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(BENCHMARK_MEASURE_ROUNDS):
            operation()
        end.record()
        end.synchronize()
        latency_us = start.elapsed_time(end) * 1000 / BENCHMARK_MEASURE_ROUNDS
        record_property(f"{name}_latency_us", latency_us)
        print(f"{name}: B={batch}, S={spec_len}, graph={capture}, latency={latency_us:.2f}us")
