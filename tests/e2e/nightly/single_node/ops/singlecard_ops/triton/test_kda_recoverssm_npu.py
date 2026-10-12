# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and commit must agree with independent, multi-step KDA recurrence."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

import vllm_ascend.ops.triton.kda.recoverssm as recoverssm_module  # noqa: E402
from vllm_ascend.device.device_config import is_950  # noqa: E402
from vllm_ascend.ops.triton.kda.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    _compact_conv_state_kernel,
    _conv_compaction_block_d,
    kda_recoverssm_verify,
)


def assert_accuracy(actual, expected, record_property, name):
    """Check both dtype-specific limits; retain per-output metrics in JUnit XML."""
    rtol, atol, fixed_limit = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }[actual.dtype]
    assert actual.shape == expected.shape
    observed = actual.detach().to(device="cpu", dtype=torch.float64)
    golden = expected.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(observed).all() and torch.isfinite(golden).all())
    error = (observed - golden).abs()
    elements = error.numel()
    matched = int((error <= atol + rtol * golden.abs()).sum())
    ratio = matched / elements if elements else 1.0
    maximum = float(error.max()) if elements and finite else 0.0 if finite else None
    limit = max(fixed_limit, 32 * torch.finfo(actual.dtype).eps)
    metrics = {
        "dtype": str(actual.dtype),
        "rtol": rtol,
        "atol": atol,
        "elements": elements,
        "matched": matched,
        "pass_ratio": ratio,
        "max_abs_error": maximum,
        "max_abs_error_limit": limit,
        "finite": finite,
    }
    record_property("accuracy/" + name, json.dumps(metrics, sort_keys=True))
    assert finite, f"{name}: non-finite result or reference: {metrics}"
    assert ratio >= 0.99, f"{name}: fewer than 99% of elements satisfy mixed tolerance: {metrics}"
    assert maximum <= limit, f"{name}: maximum absolute error exceeds its limit: {metrics}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_criteria_enforce_both_limits(dtype, record_property):
    atol = {torch.float16: 2**-9, torch.bfloat16: 2**-6, torch.float32: 2**-16}[dtype]
    golden = torch.zeros(100, dtype=torch.float64)
    actual = torch.zeros(100, dtype=dtype)
    actual[0] = 2 * atol
    assert_accuracy(actual, golden, record_property, "exactly_99_percent")
    actual[1] = 2 * atol
    with pytest.raises(AssertionError, match="fewer than 99%"):
        assert_accuracy(actual, golden, record_property, "below_99_percent")
    actual[1] = 0
    actual[0] = {torch.float16: 0.2, torch.bfloat16: 2.0, torch.float32: 0.02}[dtype]
    with pytest.raises(AssertionError, match="maximum absolute error"):
        assert_accuracy(actual, golden, record_property, "exceeds_maximum")


def assert_verify_records(correction, kg, records, record_property):
    for block, (expected_correction, expected_kg) in records.items():
        length = expected_correction.shape[1]
        assert_accuracy(correction[block, :, :length], expected_correction, record_property, f"correction_block{block}")
        torch.testing.assert_close(kg[block, :, :length].cpu().double(), expected_kg, atol=0, rtol=0)


def kda_recoverssm_commit(state, correction, kg, a_log, bias, bound, starts, indices, counts):
    # Adapt the old mathematical cases to the full multi-layer commit API.
    length = correction.shape[2]
    conv = torch.zeros(
        (state.shape[0], 3 * state.shape[1] * state.shape[2], length + 2), dtype=kg.dtype, device=state.device
    )
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2).contiguous()
    layer = SimpleNamespace(
        kv_cache=(conv, state, correction, kg),
        A_log=a_log,
        dt_bias=bias,
        gate_lower_bound=bound,
        local_num_heads=state.shape[1],
        head_dim=state.shape[3],
    )
    context = KDARecoverSSMCommitContext.create([layer], spec_query_len=length, max_num_reqs=indices.numel())
    context.commit(counts, indices, starts)


def page_view(shape, dtype, padding, head_padding=0):
    elements = 1
    inner_strides: list[int] = []
    for size in reversed(shape[1:]):
        inner_strides.insert(0, elements)
        elements *= size
    inner_strides[0] += head_padding
    elements += (shape[1] - 1) * head_padding
    backing = torch.full((shape[0], elements + padding), 11.0, dtype=dtype, device="npu")
    return backing, backing.as_strided(shape, (elements + padding, *inner_strides))


@pytest.mark.parametrize("head_padding", [0, 128])
@pytest.mark.parametrize("heads", [2, 32, 64])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 128])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_multilayer_paged_commit_maps_sampler_rows_and_compacts_conv(
    state_dtype, padding, heads, head_padding, record_property
):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(73)
    dim, length, slots = 128, 6, 5
    blocks, starts, accepted = [3, 1, 0], [0, 6, 8, 8], [4, 2, 0]
    indices = torch.tensor(blocks, dtype=torch.int32, device="npu")
    if head_padding:
        # Vary the runtime metadata stride together with the cache head strides.
        indices = torch.stack((indices, torch.zeros_like(indices)), dim=1).flatten()[::2]
    offsets = torch.tensor(starts, dtype=torch.int32, device="npu")
    # Sampler contains non-spec requests and uses a different row order.
    counts = torch.tensor([2, 0, 4, 0, 0], dtype=torch.int32, device="npu")
    request_indices = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    layers, expectations, backings, head_gaps = [], [], [], []
    for _ in range(2):
        state_backing, state = page_view((slots, heads, dim, dim), state_dtype, padding, head_padding)
        correction_backing, correction = page_view((slots, heads, length, dim), torch.float32, padding, head_padding)
        kg_backing, kg = page_view((slots, heads, length, 2 * dim), torch.bfloat16, padding, head_padding)
        conv_shape = (slots, 3 * heads * dim, length + 2)
        if not is_conv_state_dim_first():
            conv_shape = (slots, length + 2, 3 * heads * dim)
        conv_backing, conv = page_view(conv_shape, torch.bfloat16, padding)
        state.copy_(torch.randn_like(state) * 0.05)
        conv.copy_(torch.randn_like(conv))
        initial, original_conv = state.clone(), conv.clone()
        q, k, v, gate = (torch.randn((1, 8, heads, dim), dtype=torch.bfloat16, device="npu") for _ in range(4))
        beta = torch.randn((1, 8, heads), dtype=torch.bfloat16, device="npu")
        a_log = torch.randn(heads, device="npu") * 0.2
        bias = torch.randn(heads * dim, device="npu") * 0.1
        expected_output, expected_state, records = reference(
            q, k, v, gate, beta, a_log, bias, -5.0, initial, starts, blocks, accepted
        )
        output = kda_recoverssm_verify(
            q, k, v, gate, beta, a_log, bias, -5.0, state, correction, kg, offsets, indices, length
        )
        assert_accuracy(output, expected_output, record_property, "verify")
        assert_verify_records(correction, kg, records, record_property)
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        for block, count in zip(blocks[:2], accepted[:2]):
            correction[block, :, count:] = float("nan")
            kg[block, :, count:] = float("nan")
        expected_conv = original_conv.clone()
        source = original_conv if is_conv_state_dim_first() else original_conv.transpose(-1, -2)
        destination = expected_conv if is_conv_state_dim_first() else expected_conv.transpose(-1, -2)
        for block, count in zip(blocks[:2], accepted[:2]):
            destination[block, :, :3] = source[block, :, count - 1 : count + 2]
        layer = SimpleNamespace(
            kv_cache=(conv, state, correction, kg),
            A_log=a_log,
            dt_bias=bias,
            gate_lower_bound=-5.0,
            local_num_heads=heads,
            head_dim=dim,
        )
        layers.append(layer)
        expectations.append((expected_state, expected_conv))
        backings.extend([state_backing, correction_backing, kg_backing, conv_backing])
        if head_padding:
            for backing, view in ((state_backing, state), (correction_backing, correction), (kg_backing, kg)):
                for head in range(heads - 1):
                    start = (head + 1) * view.stride(1) - head_padding
                    head_gaps.append(backing[:, start : start + head_padding])
    context = KDARecoverSSMCommitContext.create(layers, spec_query_len=length, max_num_reqs=3)
    context.commit(counts, indices, offsets, request_indices=request_indices)
    for index, (layer, (state, conv)) in enumerate(zip(layers, expectations)):
        assert_accuracy(layer.kv_cache[1], state, record_property, f"commit_layer{index}")
        torch.testing.assert_close(layer.kv_cache[0], conv, atol=0, rtol=0)
    for gap in head_gaps:
        torch.testing.assert_close(gap, torch.full_like(gap, 11.0), atol=0, rtol=0)
    if padding:
        for backing in backings:
            torch.testing.assert_close(
                backing[:, -padding:], torch.full_like(backing[:, -padding:], 11.0), atol=0, rtol=0
            )


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_conv_compaction_recorded_b11_l11_shape():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(73)
    # Most common captured layout: B11/L11 (200 of384 commit records).
    channels, state_len, layers = 12288, 8, 11
    page_stride, history, block_d, plan_capacity = 1245184, 3, 256, 16
    blocks = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54]
    accepted = [4, 2, 6, 2, 2, 5, 3, 3, 1, 3, 5]
    # Keep captured strides/indices without allocating hundreds of unused pages.
    slots = max(blocks) + 2
    checked_blocks = [0, *blocks, slots - 1]
    backings, views, expectations = [], [], []
    for _ in range(layers):
        backing = torch.full((slots, page_stride), 11.0, dtype=torch.bfloat16, device="npu")
        conv = backing.as_strided((slots, channels, state_len), (page_stride, 1, channels))
        conv.copy_(torch.randn_like(conv))
        # Snapshot the complete touched pages, their padding, and two untouched pages.
        expected_backing = backing[checked_blocks].clone()
        expected = expected_backing.as_strided((len(checked_blocks), channels, state_len), (page_stride, 1, channels))
        for row, (block, count) in enumerate(zip(blocks, accepted), start=1):
            expected[row, :, :history] = conv[block, :, count - 1 : count - 1 + history]
        backings.append(backing)
        views.append(conv)
        expectations.append(expected_backing)
    metadata = [
        torch.tensor(values, dtype=torch.int64, device="npu")
        for values in (
            [view.data_ptr() for view in views],
            [page_stride] * layers,
            [1] * layers,
            [channels] * layers,
        )
    ]
    indices = torch.tensor(blocks, dtype=torch.int32, device="npu")
    counts = torch.tensor(accepted + [0] * (plan_capacity - len(blocks)), dtype=torch.int32, device="npu")
    finals = torch.tensor(blocks + [0] * (plan_capacity - len(blocks)), dtype=torch.int32, device="npu")
    boundary = torch.zeros(plan_capacity, dtype=torch.int32, device="npu")
    _compact_conv_state_kernel[(triton.cdiv(channels, block_d), len(blocks), layers)](
        views[0],
        *metadata,
        indices,
        counts,
        finals,
        boundary,
        boundary,
        0,
        channels,
        history,
        1,
        BLOCK_D=block_d,
        BLOCK_HISTORY=4,
        ALIGN_MODE=False,
        DIM_STRIDE=1,
        num_warps=4,
    )
    for backing, expected_backing in zip(backings, expectations):
        torch.testing.assert_close(backing[checked_blocks], expected_backing, atol=0, rtol=0)


@pytest.mark.parametrize("channels", [127, 12288])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_conv_compaction_device_tile_preserves_other_slots(channels, dim_first, align):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(91)
    slots, state_len, history = 8, 8, 3
    sources, destinations, accepted = [6, 3, 1, 0], [7, 4, 2, 0], [6, 1, 3, 6]
    boundaries, boundary_lengths = [5, 0, 0, 0], [2, 0, 0, 0]
    views, backings, expected_backings = [], [], []
    for _ in range(2):
        shape = (slots, channels, state_len) if dim_first else (slots, state_len, channels)
        backing, conv = page_view(shape, torch.bfloat16, 128)
        conv.copy_(torch.randn_like(conv))
        expected_backing = backing.clone()
        expected = expected_backing.as_strided(conv.shape, conv.stride())
        source_view = conv if dim_first else conv.transpose(-1, -2)
        expected_view = expected if dim_first else expected.transpose(-1, -2)
        for source, destination, count in zip(sources[:3], destinations[:3], accepted[:3]):
            expected_view[destination, :, :history] = source_view[source, :, count - 1 : count + 2]
        if align:
            expected_view[5, :, :history] = source_view[6, :, 1:4]
        views.append(source_view)
        backings.append(backing)
        expected_backings.append(expected_backing)
    metadata = [
        torch.tensor(values, dtype=torch.int64, device="npu")
        for values in (
            [view.data_ptr() for view in views],
            [view.stride(0) for view in views],
            [view.stride(1) for view in views],
            [view.stride(2) for view in views],
        )
    ]
    indices, counts, finals, boundary, lengths = (
        torch.tensor(values, dtype=torch.int32, device="npu")
        for values in (sources, accepted, destinations, boundaries, boundary_lengths)
    )
    block_d = _conv_compaction_block_d(views[0].stride(1))
    _compact_conv_state_kernel[(triton.cdiv(channels, block_d), len(sources), len(views))](
        views[0],
        *metadata,
        indices,
        counts,
        finals,
        boundary,
        lengths,
        0,
        channels,
        history,
        1,
        BLOCK_D=block_d,
        BLOCK_HISTORY=4,
        ALIGN_MODE=align,
        DIM_STRIDE=views[0].stride(1),
        USE_1D_CHANNEL_GATHER=not is_950(),
        num_warps=4,
    )
    for actual, expected in zip(backings, expected_backings):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def reference(q, k, v, raw_g, raw_beta, a_log, bias, lower_bound, state, starts, blocks, accepted):
    q, k, v, raw_g, raw_beta, a_log, bias, state = (
        tensor.detach().to(device="cpu", dtype=torch.float64)
        for tensor in (q, k, v, raw_g, raw_beta, a_log, bias, state)
    )
    output = torch.zeros_like(v)
    final = state.clone()
    records = {}
    heads, value_dim, key_dim = state.shape[1:]
    for row, block in enumerate(blocks):
        if block == 0:
            continue
        current = state[block].clone()
        corrections = []
        for position, token in enumerate(range(starts[row], starts[row + 1])):
            key = k[0, token]
            key_norm = key.square().sum(-1, keepdim=True)
            key = key * torch.rsqrt(torch.where(key_norm > 0, key_norm, 1.0))
            query = q[0, token]
            query_norm = query.square().sum(-1, keepdim=True)
            query = query * torch.rsqrt(torch.where(query_norm > 0, query_norm, 1.0)) * key_dim**-0.5
            gate_input = raw_g[0, token] + bias.view(heads, key_dim)
            a = a_log.view(heads, 1).exp()
            if lower_bound is None:
                gate = -a * torch.nn.functional.softplus(gate_input)
            else:
                gate = lower_bound * torch.sigmoid(a * gate_input)
            current = current * gate.exp()[:, None, :]
            correction = raw_beta[0, token].sigmoid()[:, None] * (v[0, token] - (current * key[:, None, :]).sum(-1))
            corrections.append(correction)
            current = current + correction[:, :, None] * key[:, None, :]
            output[0, token] = (current * query[:, None, :]).sum(-1)
            if position + 1 == accepted[row]:
                final[block] = current
        if corrections:
            span = slice(starts[row], starts[row + 1])
            records[block] = (
                torch.stack(corrections, dim=1),
                torch.cat((k[0, span], raw_g[0, span]), dim=-1).transpose(0, 1),
            )
    return output, final, records


@pytest.mark.parametrize("num_spec", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_preserves_accepted_state_across_steps(
    num_spec, state_dtype, activation_dtype, lower_bound, record_property
):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(42)
    device = "npu"
    heads, dim, capacity = 4, 128, num_spec + 1
    # Permuted slots, rejected-all row, short rows, padding and an empty row.
    blocks = [3, 1, 5, 2, 0, 4]
    lengths = [capacity, 1, capacity - 1, capacity, 1, 0]
    accepted = [0, 1, min(3, capacity - 1), capacity, 1, 0]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    indices = torch.tensor(blocks, dtype=torch.int32, device=device)
    query_starts = torch.tensor(starts, dtype=torch.int64, device=device)
    counts = torch.tensor(accepted, dtype=torch.int32, device=device)
    state = (torch.randn(8, heads, dim, dim, device=device) * 0.05).to(state_dtype)
    correction = torch.full((8, heads, capacity, dim), float("nan"), device=device)
    kg = torch.full((8, heads, capacity, 2 * dim), float("nan"), dtype=activation_dtype, device=device)
    a_log = (0.2 * torch.randn(heads, device=device)).view(1, 1, heads, 1)
    bias = 0.1 * torch.randn(heads * dim, device=device)

    for _ in range(2):
        # Token-strided inputs exercise projected/split tensor layouts.
        projections = torch.randn(1, starts[-1], heads * dim * 4, dtype=activation_dtype, device=device)
        q, k, v, raw_g = (x.view(1, starts[-1], heads, dim) for x in projections.split(heads * dim, dim=-1))
        beta = torch.randn(1, starts[-1], heads, dtype=activation_dtype, device=device)
        initial = state.clone()
        expected_out, expected_state, records = reference(
            q, k, v, raw_g, beta, a_log, bias, lower_bound, initial, starts, blocks, accepted
        )
        output_buffer = torch.empty_like(v)
        output = kda_recoverssm_verify(
            q,
            k,
            v,
            raw_g,
            beta,
            a_log.reshape(-1),
            bias,
            lower_bound,
            state,
            correction,
            kg,
            query_starts,
            indices,
            capacity,
            out=output_buffer,
        )
        assert output is output_buffer
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        assert_accuracy(output, expected_out, record_property, "verify")
        assert_verify_records(correction, kg, records, record_property)
        # Rejected suffix records must never influence the committed state.
        for row, block in enumerate(blocks):
            if block > 0:
                correction[block, :, accepted[row] :] = float("nan")
                kg[block, :, accepted[row] :] = float("nan")
        kda_recoverssm_commit(state, correction, kg, a_log, bias, lower_bound, query_starts, indices, counts)
        assert_accuracy(state, expected_state, record_property, "commit")
        # Includes block zero and unused slots as well as zero-acceptance rows.
        torch.testing.assert_close(state[[0, 3, 4, 6, 7]], initial[[0, 3, 4, 6, 7]], atol=0, rtol=0)


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_empty_batch():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    device, heads, dim, capacity = "npu", 2, 128, 5
    q = torch.empty(1, 0, heads, dim, dtype=torch.bfloat16, device=device)
    beta = torch.empty(1, 0, heads, dtype=q.dtype, device=device)
    state = torch.zeros(1, heads, dim, dim, device=device)
    correction = torch.empty(1, heads, capacity, dim, device=device)
    kg = torch.empty(1, heads, capacity, 2 * dim, dtype=q.dtype, device=device)
    a_log, bias = torch.zeros(heads, device=device), torch.zeros(heads * dim, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    output = kda_recoverssm_verify(
        q, q, q, q, beta, a_log.reshape(-1), bias, -5.0, state, correction, kg, starts, indices, capacity
    )
    assert output.shape == q.shape
    kda_recoverssm_commit(state, correction, kg, a_log, bias, -5.0, starts, indices, indices)
    assert torch.count_nonzero(state) == 0


@pytest.mark.parametrize("amplitude", [1.0, 1e-2, 1e-3, 1e-6, 1e-8, 0.0])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_matches_ascendc_small_norms(amplitude, record_property):
    from vllm_ascend.ops.kda import run_recurrent_kda

    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(61566)
    device, heads, dim, capacity = "npu", 8, 128, 6
    q, k = (torch.randn(1, capacity, heads, dim, dtype=torch.bfloat16, device=device) * amplitude for _ in range(2))
    v, gate = (torch.randn_like(q) for _ in range(2))
    beta = torch.randn(1, capacity, heads, dtype=q.dtype, device=device)
    a_log, bias = torch.zeros(heads, device=device), torch.zeros(heads * dim, device=device)
    initial = torch.randn(heads, dim, dim, device=device) * 0.05
    native = torch.zeros(capacity + 1, heads, dim, dim, device=device)
    native[1] = initial
    checkpoint = torch.zeros(2, heads, dim, dim, device=device)
    checkpoint[1] = initial
    starts = torch.tensor([0, capacity], dtype=torch.int32, device=device)
    indices = torch.ones(1, dtype=torch.int32, device=device)
    native_indices = torch.arange(1, capacity + 1, dtype=torch.int32, device=device).view(1, capacity)
    expected = run_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        native,
        starts,
        native_indices,
        a_log.reshape(-1),
        bias,
        lower_bound=-5.0,
        beta_is_preprocessed=False,
        num_accepted_tokens=indices,
    )
    correction = torch.empty(2, heads, capacity, dim, device=device)
    kg = torch.empty(2, heads, capacity, 2 * dim, dtype=q.dtype, device=device)
    output = kda_recoverssm_verify(
        q,
        k,
        v,
        gate,
        beta,
        a_log.reshape(-1),
        bias,
        -5.0,
        checkpoint,
        correction,
        kg,
        starts,
        indices,
        capacity,
    )
    assert_accuracy(output, expected, record_property, "verify_ascendc")
    # Retain the stronger small-norm regression checks alongside the common criteria.
    torch.testing.assert_close(output.float(), expected.float(), atol=1e-4, rtol=2e-4)
    for count in (1, 3, capacity):
        restored = checkpoint.clone()
        accepted = torch.full_like(indices, count)
        kda_recoverssm_commit(restored, correction, kg, a_log, bias, -5.0, starts, indices, accepted)
        assert_accuracy(restored[1], native[count], record_property, f"commit_ascendc_accepted{count}")
        torch.testing.assert_close(restored[1], native[count], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("key_dim,value_dim", [(7, 5), (7, 8), (8, 5)])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_recoverssm_ragged_tiles(key_dim, value_dim, lower_bound, record_property):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(83)
    heads, capacity, slots = 2, 3, 4
    starts, blocks, accepted = [0, 3, 5, 6], [2, 1, 0], [2, 1, 0]
    q, k, gate = (torch.randn(1, 6, heads, key_dim, dtype=torch.bfloat16, device="npu") for _ in range(3))
    v = torch.randn(1, 6, heads, value_dim, dtype=torch.bfloat16, device="npu")
    beta = torch.randn(1, 6, heads, dtype=torch.bfloat16, device="npu")
    state = torch.randn(slots, heads, value_dim, key_dim, device="npu") * 0.05
    initial = state.clone()
    correction = torch.empty(slots, heads, capacity, value_dim, device="npu")
    kg = torch.empty(slots, heads, capacity, 2 * key_dim, dtype=q.dtype, device="npu")
    a_log = torch.randn(heads, device="npu") * 0.1
    bias = torch.randn(heads * key_dim, device="npu") * 0.1
    offsets, indices, counts = (
        torch.tensor(values, dtype=torch.int32, device="npu") for values in (starts, blocks, accepted)
    )
    expected_output, expected_state, records = reference(
        q, k, v, gate, beta, a_log, bias, lower_bound, initial, starts, blocks, accepted
    )
    output = kda_recoverssm_verify(
        q, k, v, gate, beta, a_log, bias, lower_bound, state, correction, kg, offsets, indices, capacity
    )
    assert_accuracy(output, expected_output, record_property, "verify_ragged")
    assert_verify_records(correction, kg, records, record_property)
    torch.testing.assert_close(state, initial, atol=0, rtol=0)
    kda_recoverssm_commit(state, correction, kg, a_log, bias, lower_bound, offsets, indices, counts)
    assert_accuracy(state, expected_state, record_property, "commit_ragged")


@pytest.mark.parametrize("mode", ["cross", "exact", "no_cross"])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_align_context(mode, graph_mode, state_dtype, record_property):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(19516)
    slots, heads, dim, capacity = 12, 2, 128, 6
    blocks, starts, accepted = [1, 2, 0], [0, 6, 9, 15], [4, 3, 0]
    offsets = torch.tensor(starts, dtype=torch.int64, device="npu")

    def strided(values):
        backing = torch.zeros(len(values) * 2, dtype=torch.int32, device="npu")
        backing[::2] = torch.tensor(values, dtype=torch.int32, device="npu")
        return backing[::2]

    indices = strided(blocks)
    mapping, counts = strided([2, 0, 1]), strided([9, -2, 4])
    computed_values = {"cross": [7, 0, 5], "exact": [5, 0, 4], "no_cross": [1, 0, 1]}[mode]
    computed = strided(computed_values)
    table_cpu = torch.tensor([[7, 8, 9], [0, 0, 0], [5, 6, 10]], dtype=torch.int32)
    table = table_cpu.npu()
    layers, expected, originals = [], [], []
    block_size = 8
    for layer_no in range(2):
        state = (torch.randn(slots, heads, dim, dim, device="npu") * 0.03).to(state_dtype)
        initial = state.clone()
        correction = torch.full((slots, heads, capacity, dim), float("nan"), device="npu")
        kg = torch.full((slots, heads, capacity, 2 * dim), float("nan"), dtype=torch.bfloat16, device="npu")
        q, k, v, gate = (torch.randn(1, 15, heads, dim, dtype=torch.bfloat16, device="npu") for _ in range(4))
        beta = torch.randn(1, 15, heads, dtype=torch.bfloat16, device="npu")
        a_log = torch.randn(heads, device="npu") * 0.1
        bias = torch.randn(heads * dim, device="npu") * 0.1
        golden_output, final, _ = reference(q, k, v, gate, beta, a_log, bias, -5.0, initial, starts, blocks, accepted)
        output = kda_recoverssm_verify(
            q, k, v, gate, beta, a_log, bias, -5.0, state, correction, kg, offsets, indices, capacity
        )
        assert_accuracy(output, golden_output, record_property, f"layer{layer_no}_verify")
        conv = torch.randn(slots, 3 * heads * dim, capacity + 2, dtype=torch.bfloat16, device="npu")
        # Different physical layouts exercise the per-layer stride-table path.
        if layer_no:
            conv = conv.transpose(-1, -2).contiguous().transpose(-1, -2)
        original_conv, golden_conv = conv.clone(), conv.clone()
        golden_state = initial.cpu().double()
        touched = set()
        for row, src in enumerate(blocks[:2]):
            req = [2, 0, 1][row]
            prior, count = computed_values[req], accepted[row]
            dst = int(table_cpu[req, min((prior + count) // block_size, 2)])
            boundary = (prior // block_size + 1) * block_size
            golden_state[dst] = final[src]
            golden_conv[dst, :, :3] = original_conv[src, :, count - 1 : count + 2]
            touched.add(dst)
            if prior + count >= boundary:
                boundary_dst = int(table_cpu[req, boundary // block_size - 1])
                length = boundary - prior
                _, intermediate, _ = reference(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    a_log,
                    bias,
                    -5.0,
                    initial,
                    starts,
                    blocks,
                    [length if index == row else 0 for index in range(3)],
                )
                golden_state[boundary_dst] = intermediate[src]
                golden_conv[boundary_dst, :, :3] = original_conv[src, :, length - 1 : length + 2]
                touched.add(boundary_dst)
        stored_conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
        layers.append(
            SimpleNamespace(
                kv_cache=(stored_conv, state, correction, kg),
                A_log=a_log,
                dt_bias=bias,
                gate_lower_bound=-5.0,
                local_num_heads=heads,
                head_dim=dim,
            )
        )
        expected.append((golden_state, golden_conv, touched))
        originals.append((initial, original_conv))
    context = KDARecoverSSMCommitContext.create(layers, spec_query_len=capacity, max_num_reqs=3)
    assert context.conv_dim_stride is None

    def launch():
        context.commit(counts, indices, offsets, mapping, table, computed, block_size)

    if graph_mode:
        launch()
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            launch()
        for layer, (initial, conv) in zip(layers, originals):
            layer.kv_cache[1].copy_(initial)
            view = layer.kv_cache[0] if is_conv_state_dim_first() else layer.kv_cache[0].transpose(-1, -2)
            view.copy_(conv)
        captured.replay()
    else:
        launch()
    for index, (layer, (state, conv, touched), (initial, _)) in enumerate(zip(layers, expected, originals)):
        assert_accuracy(layer.kv_cache[1], state, record_property, f"layer{index}_align_state")
        view = layer.kv_cache[0] if is_conv_state_dim_first() else layer.kv_cache[0].transpose(-1, -2)
        torch.testing.assert_close(view, conv, atol=0, rtol=0)
        unused = sorted(set(range(slots)) - touched)
        torch.testing.assert_close(layer.kv_cache[1][unused], initial[unused], atol=0, rtol=0)


def test_context_input_constraints():
    with pytest.raises(ValueError, match="at least one layer"):
        KDARecoverSSMCommitContext.create([], spec_query_len=6, max_num_reqs=4)


@pytest.mark.parametrize("num_layers", [11, 12, 34])
@pytest.mark.parametrize("batch", [11, 12])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_model_commit_context(batch, graph_mode, num_layers, record_property):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    slots, heads, dim, span = batch + 2, 64 if is_950() else 32, 128, 6
    # Real contexts have 11/11/12 layers; 34 layers is an extra capacity stress case.
    real_pool_layout = num_layers != 34
    state_page_stride = {32: 625152, 64: 1250304}[heads]

    def allocate(shape, dtype, strides):
        if real_pool_layout:
            return torch.empty_strided(shape, strides, dtype=dtype, device="npu")
        return torch.empty(shape, dtype=dtype, device="npu")

    indices = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    starts = torch.arange(batch + 1, dtype=torch.int32, device="npu") * span
    counts_cpu = torch.tensor(([1, 3, 6] * ((batch + 2) // 3))[:batch], dtype=torch.int32)
    counts = counts_cpu.npu()
    entries, seeds = [], []
    # k=0 and g=A_log=dt_bias=0 give S_after = S_before * exp(-2.5 * accepted).
    for layer_no in range(num_layers):
        seed = torch.arange(slots, dtype=torch.float32) * 0.002 + (layer_no + 1) * 0.001
        state = allocate((slots, heads, dim, dim), torch.float32, (state_page_stride, dim * dim, dim, 1))
        state.copy_(seed.npu()[:, None, None, None])
        correction = allocate((slots, heads, span, dim), torch.float32, (state_page_stride, span * dim, dim, 1)).fill_(
            1
        )
        kg = allocate(
            (slots, heads, span, 2 * dim),
            torch.bfloat16,
            (2 * state_page_stride, span * 2 * dim, 2 * dim, 1),
        ).zero_()
        conv = allocate((slots, 3 * heads * dim, 8), torch.bfloat16, (2 * state_page_stride, 1, 3 * heads * dim))
        conv.copy_(torch.arange(8, dtype=torch.bfloat16, device="npu").view(1, 1, 8) + layer_no * 4)
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
            if not real_pool_layout:
                conv = conv.contiguous()
        entries.append(
            SimpleNamespace(
                kv_cache=(conv, state, correction, kg),
                A_log=torch.zeros(heads, device="npu"),
                dt_bias=torch.zeros(heads * dim, device="npu"),
                gate_lower_bound=-5.0,
                local_num_heads=heads,
                head_dim=dim,
            )
        )
        seeds.append(seed)
    context = KDARecoverSSMCommitContext.create(entries, spec_query_len=span, max_num_reqs=batch)

    def reset():
        for layer_no, (entry, seed) in enumerate(zip(entries, seeds)):
            conv, state, _, _ = entry.kv_cache
            state.copy_(seed.npu()[:, None, None, None])
            logical = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
            logical.copy_(torch.arange(8, dtype=torch.bfloat16, device="npu").view(1, 1, 8) + layer_no * 4)

    context.commit(counts, indices, starts)
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            context.commit(counts, indices, starts)
        reset()
        graph.replay()
    torch.npu.synchronize()
    factors = torch.ones(slots, dtype=torch.float64)
    factors[1 : batch + 1] = torch.exp(-2.5 * counts_cpu.double())
    for layer_no, (entry, seed) in enumerate(zip(entries, seeds)):
        conv, state, _, _ = entry.kv_cache
        golden = (seed.double() * factors)[:, None, None, None].expand(slots, heads, dim, dim)
        assert_accuracy(state, golden, record_property, f"layer{layer_no}_full_context")
        assert torch.equal(state[0].cpu(), seed[0].expand(heads, dim, dim))
        assert torch.equal(state[-1].cpu(), seed[-1].expand(heads, dim, dim))
        expected = (torch.arange(8, dtype=torch.bfloat16) + layer_no * 4).view(1, 1, 8)
        expected = expected.expand(slots, 3 * heads * dim, 8).clone()
        original = expected.clone()
        for row, count in enumerate(counts_cpu.tolist(), 1):
            expected[row, :, :3] = original[row, :, count - 1 : count + 2]
        logical = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
        assert torch.equal(logical.cpu(), expected)
    record_property("full_context_layers", num_layers)


@pytest.mark.parametrize(
    "span,heads,activation_dtype,lower_bound",
    [(span, heads, torch.bfloat16, -5.0) for heads in (32, 64) for span in (1, 2, 4, 6, 8)]
    + [
        (4, 64, torch.float16, -5.0),
        (8, 64, torch.float16, None),
        (2, 32, torch.bfloat16, None),
        (8, 64, torch.bfloat16, None),
        (6, 16, torch.bfloat16, -5.0),
    ],
)
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_commit_window_uses_device_path(
    span, heads, activation_dtype, lower_bound, graph_mode, monkeypatch, record_property
):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(19133)
    slots, dim, batch = 6, 128, 4
    blocks, accepted = [2, 1, 3, 0], [span, max(span // 2, 1), 0, 0]
    starts = [row * span for row in range(batch + 1)]
    indices, offsets, counts = (
        torch.tensor(values, dtype=torch.int32, device="npu") for values in (blocks, starts, accepted)
    )
    layers, expected, originals = [], [], []
    for layer_no in range(2):
        state = torch.randn(slots, heads, dim, dim, device="npu") * 0.03
        initial = state.clone()
        correction = torch.full((slots, heads, span, dim), float("nan"), device="npu")
        kg = torch.full((slots, heads, span, 2 * dim), float("nan"), dtype=activation_dtype, device="npu")
        projections = torch.randn(1, batch * span, 4 * heads * dim, dtype=activation_dtype, device="npu")
        q, k, v, gate = (view.view(1, batch * span, heads, dim) for view in projections.split(heads * dim, dim=-1))
        beta = torch.randn(1, batch * span, heads, dtype=activation_dtype, device="npu")
        a_log = torch.randn(heads, device="npu") * 0.1
        bias = torch.randn(heads * dim, device="npu") * 0.1
        golden_output, golden_state, records = reference(
            q, k, v, gate, beta, a_log, bias, lower_bound, initial, starts, blocks, accepted
        )
        output = kda_recoverssm_verify(
            q, k, v, gate, beta, a_log, bias, lower_bound, state, correction, kg, offsets, indices, span
        )
        assert_accuracy(output, golden_output, record_property, f"window_layer{layer_no}_verify")
        assert_verify_records(correction, kg, records, record_property)
        torch.testing.assert_close(state, initial, atol=0, rtol=0)
        for block, count in zip(blocks[:3], accepted[:3]):
            correction[block, :, count:] = float("nan")
            kg[block, :, count:] = float("nan")
        conv = torch.randn(slots, 3 * heads * dim, span + 2, dtype=activation_dtype, device="npu")
        golden_conv = conv.clone()
        for block, count in zip(blocks[:2], accepted[:2]):
            golden_conv[block, :, :3] = conv[block, :, count - 1 : count + 2]
        stored_conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
        layers.append(
            SimpleNamespace(
                kv_cache=(stored_conv, state, correction, kg),
                A_log=a_log,
                dt_bias=bias,
                gate_lower_bound=lower_bound,
                local_num_heads=heads,
                head_dim=dim,
            )
        )
        expected.append((golden_state, golden_conv))
        originals.append((initial, conv.clone()))
    context = KDARecoverSSMCommitContext.create(layers, spec_query_len=span, max_num_reqs=batch)
    grouped_commit = is_950() and heads in (32, 64)
    kernel_name = "_commit_kda_state_grouped_kernel" if grouped_commit else "_commit_kda_state_kernel"
    original_kernel, launches = getattr(recoverssm_module, kernel_name), []

    class RecordingKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def invoke(*args, **kwargs):
                launches.append(kernel_name)
                if not grouped_commit:
                    matrix_buffers = 2
                    assert (
                        kwargs["BV"] * kwargs["BK"] * torch.float32.itemsize * matrix_buffers
                        <= recoverssm_module.get_ub_size_bytes()
                    )
                    assert kwargs["UNROLL_COMMIT"] == is_950()
                return launch(*args, **kwargs)

            return invoke

    # Assert the selected path so a fixed-window fallback cannot hide missing coverage.
    monkeypatch.setattr(recoverssm_module, kernel_name, RecordingKernel())
    context.commit(counts, indices, offsets)
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            context.commit(counts, indices, offsets)
        for layer, (initial, conv) in zip(layers, originals):
            layer.kv_cache[1].copy_(initial)
            logical = layer.kv_cache[0] if is_conv_state_dim_first() else layer.kv_cache[0].transpose(-1, -2)
            logical.copy_(conv)
        graph.replay()
    assert launches and all(name == kernel_name for name in launches)
    for layer_no, (layer, (state, conv)) in enumerate(zip(layers, expected)):
        assert_accuracy(layer.kv_cache[1], state, record_property, f"window_layer{layer_no}_commit")
        logical = layer.kv_cache[0] if is_conv_state_dim_first() else layer.kv_cache[0].transpose(-1, -2)
        torch.testing.assert_close(logical, conv, atol=0, rtol=0)
        torch.testing.assert_close(
            layer.kv_cache[1][[0, 3, 4, 5]], originals[layer_no][0][[0, 3, 4, 5]], atol=0, rtol=0
        )
    record_property("commit_kernel", kernel_name)
    record_property("spec_query_len", span)
