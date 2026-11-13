# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-rank GDN verify, replay and commit tests for Ascend."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.gdn import recoverssm_gdn as gdn  # noqa: E402
from vllm_ascend.ops.triton.mamba.recoverssm import compact_conv_state  # noqa: E402

# Qwen/Qwen3.6-35B-A3B and Qwen/Qwen3.8-27B config.json, respectively.
# Both have 16 key heads, K=V=128, and a convolution kernel width of four.
QWEN_35B = 32
QWEN_27B = 48
HISTORY = 3


def check_accuracy(actual, expected, record_property, label):
    """Apply the mixed-tolerance and worst-element gates to each output."""
    tolerances = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }
    rtol, atol, floor = tolerances[actual.dtype]
    assert actual.shape == expected.shape
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float64)
    expected_cpu = expected.detach().to(device="cpu", dtype=torch.float64)
    assert torch.isfinite(actual_cpu).all() and torch.isfinite(expected_cpu).all(), label
    error = (actual_cpu - expected_cpu).abs()
    passed = (error <= atol + rtol * expected_cpu.abs()).count_nonzero().item()
    ratio = passed / error.numel() if error.numel() else 1.0
    maximum = error.max().item() if error.numel() else 0.0
    limit = max(floor, 32 * torch.finfo(actual.dtype).eps)
    metrics = dict(
        dtype=str(actual.dtype),
        elements=error.numel(),
        matched=passed,
        rtol=rtol,
        atol=atol,
        pass_ratio=ratio,
        max_abs_error=maximum,
        max_abs_error_limit=limit,
    )
    record_property(f"accuracy/{label}", json.dumps(metrics))
    assert ratio >= 0.99, f"{label}: mixed tolerance: {metrics}"
    assert maximum <= limit, f"{label}: maximum error: {metrics}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_gates(dtype, record_property):
    expected = torch.zeros(100, dtype=torch.float64)
    actual = torch.zeros(100, dtype=dtype)
    actual[:1] = 0.005 if dtype == torch.float32 else 0.05
    check_accuracy(actual, expected, record_property, "99_percent")
    actual[:2] = 0.005 if dtype == torch.float32 else 0.05
    with pytest.raises(AssertionError, match="mixed tolerance"):
        check_accuracy(actual, expected, record_property, "98_percent")
    actual.zero_()
    actual[-1] = 4
    with pytest.raises(AssertionError, match="maximum error"):
        check_accuracy(actual, expected, record_property, "outlier")


@pytest.fixture
def npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(419)
    return torch.device("npu")


def packed_layer(device, pages, hk, hv, key, value, spec_query_len, state_dtype, channel_first, padding=256):
    """Build typed views over a common byte-strided Mamba allocation."""
    channels = 2 * hk * key + hv * value
    span = spec_query_len + HISTORY - 1
    conv_shape = (channels, span) if channel_first else (span, channels)
    shapes = (conv_shape, (hv, value, key), (hv, spec_query_len, value + key + 1))
    dtypes = (torch.bfloat16, state_dtype, torch.float32)
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in zip(shapes, dtypes)]
    page_bytes = sum(sizes) + padding
    assert page_bytes % 4 == 0
    raw = torch.full((pages, page_bytes), 165, dtype=torch.uint8, device=device)
    views = []
    offset = 0
    for shape, dtype, size in zip(shapes, dtypes, sizes):
        strides = tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
        views.append(
            raw.view(dtype).as_strided(
                (pages, *shape), (page_bytes // dtype.itemsize, *strides), offset // dtype.itemsize
            )
        )
        offset += size
    conv, state, replay = views
    conv.copy_(torch.randn_like(conv))
    state.copy_(torch.randn_like(state) * 0.02)
    replay.fill_(float("nan"))
    return SimpleNamespace(num_k_heads=hk, tp_size=1, kv_cache=tuple(views)), raw


def logical_conv(layer, channel_first):
    conv = layer.kv_cache[0]
    return conv if channel_first else conv.transpose(-2, -1)


def model_inputs(device, tokens, hk, hv, key, value):
    packed = torch.randn(1, tokens, 2 * hk * key + hv * value, dtype=torch.bfloat16, device=device)
    q, k, v = packed.split((hk * key, hk * key, hv * value), dim=-1)
    a_log = torch.linspace(-0.4, 0.3, hv, device=device)
    bias = torch.linspace(-0.2, 0.2, hv, device=device)
    a = torch.randn(tokens, hv, device=device, dtype=torch.bfloat16)
    b = torch.randn(1, tokens, hv, device=device, dtype=torch.bfloat16)
    return a_log, a, b, bias, q.unflatten(-1, (hk, key)), k.unflatten(-1, (hk, key)), v.unflatten(-1, (hv, value))


def cpu_reference(inputs, checkpoint, offsets, pages, *, normalize=True, scale=None, sp_beta=1.0, threshold=20.0):
    """Forward token recurrence with per-value-head GDN decay and GQA."""
    a_log, a, b, bias, q, k, v = (tensor.detach().cpu().double() for tensor in inputs)
    initial = checkpoint.detach().cpu().double()
    ratio = v.shape[2] // k.shape[2]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if normalize:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = (q[0] * scale).repeat_interleave(ratio, dim=1)
    k = k[0].repeat_interleave(ratio, dim=1)
    x = a.reshape_as(b[0]) + bias
    softplus = torch.nn.functional.softplus(x, beta=sp_beta, threshold=threshold)
    decay = (-a_log.exp() * softplus).exp()
    beta = b[0].sigmoid()
    output = torch.zeros_like(v)
    histories, records = {}, {}
    for row, page in enumerate(pages):
        if page <= 0:
            continue
        state = initial[page]
        states, tokens = [state], []
        for token in range(offsets[row], offsets[row + 1]):
            decayed = state * decay[token, :, None, None]
            key = k[token]
            residual = beta[token, :, None] * (v[0, token] - torch.einsum("hvk,hk->hv", decayed, key))
            state = decayed + torch.einsum("hv,hk->hvk", residual, key)
            output[0, token] = torch.einsum("hvk,hk->hv", state, q[token])
            tokens.append(torch.cat((residual, key, decay[token, :, None]), dim=-1))
            states.append(state)
        histories[row] = states
        if tokens:
            records[page] = torch.stack(tokens, dim=1)
    return output, histories, records


def verify(inputs, layer, offsets, pages, window, **options):
    _, checkpoint, replay = layer.kv_cache
    return gdn.gdn_recoverssm_verify(
        *inputs,
        checkpoint_state=checkpoint,
        replay_cache=replay,
        query_start_loc=offsets,
        state_indices=pages,
        spec_query_len=window,
        **options,
    )


def check_records(replay, records, value, record_property, label):
    for page, expected in records.items():
        observed = replay[page, :, : expected.shape[1]]
        for name, fields in (("correction", slice(0, value)), ("key", slice(value, -1)), ("decay", slice(-1, None))):
            check_accuracy(observed[..., fields], expected[..., fields], record_property, f"{label}/{page}/{name}")


def replay_commit(launch, backings, graph_mode):
    """Capture commit after warmup and replay from the original pages."""
    if not graph_mode:
        launch()
        return
    saved = [backing.clone() for backing in backings]
    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for backing, original in zip(backings, saved):
        backing.copy_(original)
    graph.replay()
    torch.npu.synchronize()


@pytest.mark.parametrize(
    "global_hv,tp,spec_query_len,state_dtype,channel_first",
    [
        pytest.param(QWEN_35B, 4, 5, torch.float32, True, id="qwen36-tp4-mtp4"),
        pytest.param(QWEN_27B, 2, 9, torch.bfloat16, False, id="qwen38-tp2-dflash8"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_multilayer_verify_and_commit_across_steps(
    npu, monkeypatch, record_property, global_hv, tp, spec_query_len, state_dtype, channel_first
):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: channel_first)
    hk, hv = 16 // tp, global_hv // tp
    pairs = [
        packed_layer(npu, 5, hk, hv, 128, 128, spec_query_len, state_dtype, channel_first, gap) for gap in (256, 512)
    ]
    layers, backings = zip(*pairs)
    for layer in layers:
        layer.num_k_heads, layer.tp_size = 16, tp
    context = gdn.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=4)
    query_start_offsets = [0, spec_query_len, spec_query_len + 2, spec_query_len + 3, spec_query_len + 3]
    query_start_loc = torch.tensor(query_start_offsets, dtype=torch.int64, device=npu)
    spec_to_sampler_rows = torch.tensor([2, 0, 1, 3], dtype=torch.int32, device=npu)
    # Spec row 1 reads sampler row 0 and clamps 99 to its two-token query span.
    sampler_accepted_counts = torch.tensor([99, 1, spec_query_len - 1, -1], dtype=torch.int32, device=npu)
    for step, pages in enumerate(([3, 1, 0, 2], [1, 3, 0, 2])):
        indices = torch.tensor(pages, dtype=torch.int32, device=npu)
        expected_layers = []
        for layer_no, layer in enumerate(layers):
            label = f"step{step}/layer{layer_no}"
            _, state, replay = layer.kv_cache
            initial = state.clone()
            conv = logical_conv(layer, channel_first)
            original_conv = conv.clone()
            inputs = model_inputs(npu, query_start_offsets[-1], hk, hv, 128, 128)
            expected, histories, records = cpu_reference(inputs, state, query_start_offsets, pages)
            out = torch.empty_like(inputs[-1], memory_format=torch.contiguous_format)
            assert verify(inputs, layer, query_start_loc, indices, spec_query_len, out=out) is out
            check_accuracy(out, expected, record_property, f"{label}/verify")
            check_records(replay, records, 128, record_property, label)
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            torch.testing.assert_close(out[:, -1], torch.zeros_like(out[:, -1]), atol=0, rtol=0)
            expected_state, expected_conv = initial.clone(), original_conv.clone()
            for row, count in enumerate((spec_query_len - 1, 2)):
                page = pages[row]
                expected_state[page].copy_(histories[row][count])
                expected_conv[page, :, :HISTORY] = original_conv[page, :, count - 1 : count - 1 + HISTORY]
                replay[page, :, count:].fill_(float("nan"))
            expected_layers.append((expected_state, expected_conv))
        context.commit(sampler_accepted_counts, indices, query_start_loc, request_indices=spec_to_sampler_rows)
        for layer_no, (layer, backing, (state, conv)) in enumerate(zip(layers, backings, expected_layers)):
            check_accuracy(layer.kv_cache[1], state, record_property, f"step{step}/layer{layer_no}/commit")
            torch.testing.assert_close(layer.kv_cache[1][[0, 2, 4]], state[[0, 2, 4]], atol=0, rtol=0)
            torch.testing.assert_close(logical_conv(layer, channel_first), conv, atol=0, rtol=0)
            assert torch.all(backing[:, -(256 if layer_no == 0 else 512) :] == 165)


def test_head_padding_survives_verify_and_commit(npu, monkeypatch, record_property):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    hk, hv, key, value, spec_query_len = 4, 8, 128, 128, 3
    layer, _ = packed_layer(npu, 3, hk, hv, key, value, spec_query_len, torch.float32, True)
    conv = layer.kv_cache[0]
    checkpoint_storage = torch.full((3, hv, value * key + 16), -5.0, device=npu)
    replay_storage = torch.full((3, hv, spec_query_len * (value + key + 1) + 16), -7.0, device=npu)
    checkpoint = checkpoint_storage[..., :-16].unflatten(-1, (value, key))
    replay = replay_storage[..., :-16].unflatten(-1, (spec_query_len, value + key + 1))
    checkpoint.copy_(torch.randn_like(checkpoint) * 0.02)
    layer.kv_cache = (conv, checkpoint, replay)
    indices = torch.tensor([1, 0], dtype=torch.int32, device=npu)[::2]
    offsets = torch.tensor([0, spec_query_len], dtype=torch.int32, device=npu)
    inputs = model_inputs(npu, spec_query_len, hk, hv, key, value)
    expected, histories, records = cpu_reference(inputs, checkpoint, [0, spec_query_len], [1])
    before = checkpoint_storage.clone()
    output = verify(inputs, layer, offsets, indices, spec_query_len)
    check_accuracy(output, expected, record_property, "head_padding_verify")
    check_records(replay, records, value, record_property, "head_padding_records")
    torch.testing.assert_close(checkpoint_storage, before, atol=0, rtol=0)
    context = gdn.GDNRecoverSSMCommitContext.create([layer], spec_query_len=spec_query_len, max_num_reqs=1)
    context.commit(torch.tensor([2], dtype=torch.int32, device=npu), indices, offsets)
    check_accuracy(checkpoint[1], histories[0][2], record_property, "head_padding_commit")
    assert (checkpoint_storage[..., -16:] == -5).all()
    assert (replay_storage[..., -16:] == -7).all()
    torch.testing.assert_close(checkpoint_storage[[0, 2]], before[[0, 2]], atol=0, rtol=0)


@pytest.mark.parametrize(
    "channels,channel_first,align,batch,layers",
    [
        (127, True, False, 4, 2),
        (2048, False, False, 11, 11),  # Qwen3.6 TP4 channels; synthetic B11/L11.
        (2560, True, True, 4, 2),  # Qwen3.8 TP4
        (5120, False, True, 4, 2),  # Qwen3.8 TP2
    ],
)
def test_compaction_page_routing(npu, channels, channel_first, align, batch, layers):
    """Check the shared convolution wrapper independently of state recovery."""
    pages, span = 7, 7
    physical = (channels, span) if channel_first else (span, channels)
    backings = [
        torch.full((pages, channels * span + gap), -3, dtype=torch.bfloat16, device=npu)
        for gap in (128 + 128 * layer for layer in range(layers))
    ]
    views, expected = [], []
    for backing in backings:
        golden = backing.clone()
        page_stride = backing.stride(0)
        inner = (physical[1], 1)
        view = backing.as_strided((pages, *physical), (page_stride, *inner))
        view.copy_(torch.randn_like(view))
        golden.copy_(backing)
        target = golden.as_strided(view.shape, view.stride())
        if not channel_first:
            view, target = view.transpose(-1, -2), target.transpose(-1, -2)
        target[3, :, :HISTORY] = view[1, :, 3:6]
        target[2, :, :HISTORY] = view[2, :, :HISTORY]
        if align:
            target[4, :, :HISTORY] = view[1, :, 1:4]
        views.append(view)
        expected.append(golden)
    address = torch.tensor([view.data_ptr() for view in views], dtype=torch.int64, device=npu)
    page_stride, channel_stride, token_stride = (
        torch.tensor([view.stride(axis) for view in views], dtype=torch.int64, device=npu) for axis in range(3)
    )
    source, lengths, finals, boundary, boundary_len = (
        torch.tensor([*values, *([0] * (batch - 4))], dtype=torch.int32, device=npu)
        for values in ([1, 2, 0, 5], [4, 1, 3, 0], [3, 2, 6, 5], [4, 0, 0, 0], [2, 0, 0, 0])
    )
    compact_conv_state(
        views[0],
        address,
        page_stride,
        channel_stride,
        token_stride,
        source,
        lengths,
        finals,
        boundary,
        boundary_len,
        conv_history_len=HISTORY,
        align_mode=align,
        dim_stride=views[0].stride(1),
        null_block_id=0,
    )
    for backing, golden in zip(backings, expected):
        torch.testing.assert_close(backing, golden, atol=0, rtol=0)


@pytest.mark.parametrize(
    "key,value,amplitude,normalize",
    [
        (7, 5, 1, False),
        (7, 8, 1, True),
        (8, 5, 1, True),
        (128, 128, 1e-3, True),
        (128, 128, 1e-8, True),
        (128, 128, 0, True),
    ],
)
def test_verify_ragged_tiles_and_small_norms(npu, monkeypatch, record_property, key, value, amplitude, normalize):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    hk, hv, spec_query_len = 2, 6, 4
    layer, _ = packed_layer(npu, 3, hk, hv, key, value, spec_query_len, torch.float32, False)
    _, state, replay = layer.kv_cache
    a_log, a, b, bias, q, k, v = model_inputs(npu, spec_query_len, hk, hv, key, value)
    a[0], a[1] = -8, 8
    inputs = (a_log, a, b, bias, q * amplitude, k * amplitude, v)
    offsets = torch.tensor([0, -1, spec_query_len], device=npu, dtype=torch.int64)[::2]
    pages = torch.tensor([1], device=npu, dtype=torch.int64)
    expected, states, records = cpu_reference(
        inputs,
        state,
        [0, spec_query_len],
        [1],
        normalize=normalize,
        sp_beta=0.5,
        threshold=2.0,
        scale=0.25,
    )
    before = state.clone()
    out = torch.empty(1, spec_query_len * 2, hv, value, dtype=v.dtype, device=npu)[:, ::2]
    actual = verify(
        inputs,
        layer,
        offsets,
        pages,
        spec_query_len,
        out=out,
        scale=0.25,
        softplus_beta=0.5,
        softplus_threshold=2.0,
        use_qk_l2norm_in_kernel=normalize,
    )
    assert actual is out
    check_accuracy(actual, expected, record_property, "ragged_or_small_norm")
    check_records(replay, records, value, record_property, "replay")
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    context = gdn.GDNRecoverSSMCommitContext.create([layer], spec_query_len=spec_query_len, max_num_reqs=1)
    conv = layer.kv_cache[0].clone()
    for length in (1, spec_query_len):
        state.copy_(before)
        context.commit(torch.full_like(pages, length), pages, offsets, commit_conv=False)
        check_accuracy(state[1], states[0][length], record_property, f"prefix{length}")
        torch.testing.assert_close(layer.kv_cache[0], conv, atol=0, rtol=0)


@pytest.mark.parametrize("amplitude", [1.0, 1e-3, 1e-8, 0.0])
def test_small_norms_against_ascendc(npu, monkeypatch, record_property, amplitude):
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    hk, hv, dim, spec_query_len = 4, 12, 128, 5
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    layer, _ = packed_layer(npu, 2, hk, hv, dim, dim, spec_query_len, torch.float32, True)
    _, checkpoint, _ = layer.kv_cache
    a_log, a, b, bias, q, k, v = model_inputs(npu, spec_query_len, hk, hv, dim, dim)
    q, k = q * amplitude, k * amplitude
    b.zero_()
    native_state = checkpoint.clone().contiguous()
    before = checkpoint.clone()
    page = torch.ones(1, dtype=torch.int32, device=npu)
    span = torch.tensor([0, spec_query_len], dtype=torch.int32, device=npu)
    one_token = torch.tensor([0, 1], dtype=torch.int32, device=npu)

    # Use the same BF16 Q/K on both paths. The FP64 test checks internal normalization.
    q_norm, k_norm = (
        (tensor.float() * (tensor.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16()
        for tensor in (q[0], k[0])
    )
    decay_log = -a_log.exp() * torch.nn.functional.softplus(a.float() + bias)
    beta = b[0].float().sigmoid()
    outputs, prefixes = [], []
    for pos in range(spec_query_len):
        step = slice(pos, pos + 1)
        outputs.append(
            recurrent_gated_delta_rule(
                q_norm[step].contiguous(),
                k_norm[step].contiguous(),
                v[0, step].contiguous(),
                native_state,
                g=decay_log[step],
                beta=beta[step].bfloat16(),
                scale=dim**-0.5,
                actual_seq_lengths=one_token,
                ssm_state_indices=page,
            )
        )
        prefixes.append(native_state[1].clone())
    inputs = (a_log, a, b, bias, q_norm.unsqueeze(0), k_norm.unsqueeze(0), v)
    actual = verify(inputs, layer, span, page, spec_query_len, use_qk_l2norm_in_kernel=False)
    check_accuracy(actual, torch.cat(outputs).unsqueeze(0), record_property, "ascendc_verify")
    context = gdn.GDNRecoverSSMCommitContext.create([layer], spec_query_len=spec_query_len, max_num_reqs=1)
    for count in (1, 3, spec_query_len):
        checkpoint.copy_(before)
        context.commit(torch.full_like(page, count), page, span, commit_conv=False)
        check_accuracy(checkpoint[1], prefixes[count - 1], record_property, f"ascendc_prefix{count}")


@pytest.mark.parametrize("offsets,pages", [([0], []), ([0, 3], [0]), ([0, 3], [-1])])
def test_empty_and_null_rows(npu, monkeypatch, offsets, pages):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    layer, backing = packed_layer(npu, 3, 2, 4, 16, 16, 3, torch.float32, True)
    before = backing.clone()
    inputs = model_inputs(npu, offsets[-1], 2, 4, 16, 16)
    starts = torch.tensor(offsets, dtype=torch.int32, device=npu)
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    out = verify(inputs, layer, starts, indices, 3)
    torch.testing.assert_close(out, torch.zeros_like(inputs[-1]), atol=0, rtol=0)
    context = gdn.GDNRecoverSSMCommitContext.create([layer], spec_query_len=3, max_num_reqs=1)
    context.commit(torch.full_like(indices, 3), indices, starts)
    torch.testing.assert_close(backing, before, atol=0, rtol=0)


@pytest.mark.parametrize("window,capacity", [(0, 2), (4, 0), (4, 2)])
def test_context_rejects_invalid_capacity_or_no_layers(window, capacity):
    layers = [object()] if min(window, capacity) == 0 else []
    with pytest.raises(ValueError, match="requires layers and positive capacities"):
        gdn.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=capacity)


@pytest.mark.parametrize(
    "prior,count,table,graph_mode,state_dtype",
    [
        pytest.param(6, 4, [3, 4], False, torch.float32, id="cross"),
        pytest.param(6, 2, [3, 4], True, torch.bfloat16, id="exact"),
        pytest.param(1, 2, [3, 4], False, torch.bfloat16, id="relocate"),
        pytest.param(1, 2, [1, 4], True, torch.float32, id="in-place"),
        pytest.param(6, 0, [3, 4], False, torch.float32, id="no-acceptance"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_align_final_and_boundary_routes(
    npu, monkeypatch, record_property, prior, count, table, graph_mode, state_dtype
):
    hk, hv, dim, spec_query_len, block_size = 4, 12, 128, 5, 8
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    pairs = [packed_layer(npu, 8, hk, hv, dim, dim, spec_query_len, state_dtype, layout) for layout in (True, False)]
    layers, backings = zip(*pairs)
    layers[1].kv_cache = (layers[1].kv_cache[0].transpose(-1, -2), *layers[1].kv_cache[1:])
    context = gdn.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=3)
    assert context.conv_shared_dim_stride is None
    pages, offsets = [1, 2, 0], [0, 5, 7, 8]
    starts = torch.tensor([0, -1, 5, -1, 7, -1, 8], dtype=torch.int64, device=npu)[::2]
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    counts = torch.tensor([9, -1, 7, -1, count, -1], dtype=torch.int32, device=npu)[::2]
    mapping = torch.tensor([2, -1, 0, -1, 1, -1], dtype=torch.int32, device=npu)[::2]
    computed = torch.tensor([1, 0, prior], dtype=torch.int32, device=npu)
    table_data = [[6, 7], [0, 0], table]
    block_table = torch.tensor(table_data, dtype=torch.int64, device=npu).repeat_interleave(2, dim=1)[:, ::2]
    expected_layers = []
    for layer_no, layer in enumerate(layers):
        conv, state, replay = layer.kv_cache
        inputs = model_inputs(npu, offsets[-1], hk, hv, dim, dim)
        golden, history, records = cpu_reference(inputs, state, offsets, pages)
        output = verify(inputs, layer, starts, indices, spec_query_len)
        check_accuracy(output, golden, record_property, f"align_verify{layer_no}")
        check_records(replay, records, dim, record_property, f"align_replay{layer_no}")
        expected_state, expected_conv = state.clone(), conv.clone()
        untouched = set(range(state.shape[0]))
        for row, (previous, accepted, destinations) in enumerate(((prior, count, table), (1, 2, [6, 7]))):
            if not accepted:
                continue
            writes = [(destinations[(previous + accepted - 1) // block_size], accepted)]
            boundary = block_size - previous % block_size
            if accepted >= boundary:
                writes.append((destinations[previous // block_size], boundary))
            for destination, length in writes:
                expected_state[destination].copy_(history[row][length])
                expected_conv[destination, :, :HISTORY] = conv[pages[row], :, length - 1 : length - 1 + HISTORY]
                untouched.discard(destination)
        expected_layers.append((expected_state, expected_conv, sorted(untouched)))

    def launch():
        context.commit(counts, indices, starts, mapping, block_table, computed, block_size)

    replay_commit(launch, backings, graph_mode)
    plan = context.commit_plan
    torch.testing.assert_close(plan.commit_lens[:3].cpu(), torch.tensor([count, 2, 0], dtype=torch.int32))
    torch.testing.assert_close(
        plan.state_commit_lens[:3] + plan.boundary_commit_lens[:3], plan.commit_lens[:3], atol=0, rtol=0
    )
    assert not (plan.state_commit_lens[:3].bool() & plan.boundary_commit_lens[:3].bool()).any()
    for layer_no, (layer, (state, conv, untouched_pages)) in enumerate(zip(layers, expected_layers)):
        check_accuracy(layer.kv_cache[1], state, record_property, f"align_commit{layer_no}")
        torch.testing.assert_close(layer.kv_cache[1][untouched_pages], state[untouched_pages], atol=0, rtol=0)
        torch.testing.assert_close(layer.kv_cache[0], conv, atol=0, rtol=0)


@pytest.mark.parametrize(
    "global_hv,tp,spec_query_len,batch,layer_count,graph_mode",
    [
        pytest.param(QWEN_35B, 4, 5, 11, 30, False, id="qwen36-mtp4"),
        pytest.param(QWEN_27B, 4, 9, 12, 48, True, id="qwen38-dflash8"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_model_packed_page_commit(
    npu, monkeypatch, record_property, global_hv, tp, spec_query_len, batch, layer_count, graph_mode
):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    hk, hv = 16 // tp, global_hv // tp
    pairs = [
        packed_layer(npu, batch + 2, hk, hv, 128, 128, spec_query_len, torch.float32, False) for _ in range(layer_count)
    ]
    layers, backings = zip(*pairs)
    count_cpu = torch.arange(batch, dtype=torch.int32) % spec_query_len + 1
    starts = torch.arange(batch + 1, dtype=torch.int32, device=npu) * spec_query_len
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=npu)
    counts = count_cpu.to(npu)
    for layer_no, layer in enumerate(layers):
        conv, state, replay = layer.kv_cache
        state.fill_(0.02 * (1 + layer_no % 5))
        replay[..., :-1].zero_()
        replay[..., -1].fill_(0.75)
        conv.copy_(torch.arange(spec_query_len + HISTORY - 1, device=npu, dtype=torch.bfloat16)[None, :, None])

    context = gdn.GDNRecoverSSMCommitContext.create(layers, spec_query_len=spec_query_len, max_num_reqs=batch)

    def launch():
        context.commit(counts, indices, starts)

    replay_commit(launch, backings, graph_mode)
    # Zero effective key and correction reduce recovery to scalar decay.
    factors = torch.ones(batch + 2, dtype=torch.float64)
    factors[1:-1] = 0.75 ** count_cpu.double()
    for layer_no, (layer, raw) in enumerate(zip(layers, backings)):
        conv, state, replay = layer.kv_cache
        initial = torch.tensor(0.02 * (1 + layer_no % 5), dtype=torch.float32).double()
        golden = (initial * factors)[:, None, None, None].expand(state.shape)
        check_accuracy(state, golden, record_property, f"packed_layer{layer_no}")
        torch.testing.assert_close(state[[0, batch + 1]].cpu().double(), golden[[0, batch + 1]], atol=0, rtol=0)
        sequence = torch.arange(spec_query_len + HISTORY - 1, dtype=torch.bfloat16)
        expected_conv = sequence.expand(batch + 2, -1).clone()
        for page, count in enumerate(count_cpu.tolist(), start=1):
            expected_conv[page, :HISTORY] = sequence[count - 1 : count - 1 + HISTORY]
        torch.testing.assert_close(conv.cpu(), expected_conv[..., None].expand(conv.shape), atol=0, rtol=0)
        assert replay[..., :-1].count_nonzero() == 0
        assert (replay[..., -1] == 0.75).all()
        assert (raw[:, -256:] == 165).all()


@pytest.mark.parametrize(
    "hk,hv,key,value,spec_query_len,state_dtype,graph_mode",
    [
        (4, 8, 128, 128, 1, torch.float32, False),
        (4, 8, 128, 128, 2, torch.bfloat16, True),
        (4, 8, 128, 128, 5, torch.float32, True),
        (8, 24, 128, 128, 9, torch.bfloat16, False),
        (1, 3, 128, 128, 5, torch.float32, False),
        (2, 6, 33, 47, 4, torch.float32, True),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_configured_window_dispatch(
    npu, monkeypatch, record_property, hk, hv, key, value, spec_query_len, state_dtype, graph_mode
):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: True)
    # Enough rows to hit the large verify tile on A5 for the production shapes.
    batch = max(3, math.ceil(96 / hv)) if key == value == 128 and hv % 2 == 0 else 3
    layer, backing = packed_layer(npu, batch + 1, hk, hv, key, value, spec_query_len, state_dtype, True)
    conv, state, _ = layer.kv_cache
    pages = list(range(1, batch + 1))
    offsets = [row * spec_query_len for row in range(batch + 1)]
    starts = torch.tensor(offsets, dtype=torch.int32, device=npu)
    indices = torch.tensor(pages, dtype=torch.int32, device=npu)
    counts = torch.full_like(indices, spec_query_len)
    inputs = model_inputs(npu, offsets[-1], hk, hv, key, value)
    # Reuse the public entry points, recording rather than replacing kernel execution.
    launches = []

    class TraceKernel:
        def __init__(self, name):
            self.name = name
            self.kernel = getattr(gdn, name)

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append((self.name, grid, kwargs))
                return self.kernel[grid](*args, **kwargs)

            return run

    verify_name = "_gdn_recoverssm_verify_kernel"
    ordinary = "_commit_gdn_state_kernel"
    grouped = "_commit_gdn_state_grouped_kernel"
    for name in (verify_name, ordinary, grouped):
        monkeypatch.setattr(gdn, name, TraceKernel(name))
    golden, histories, _ = cpu_reference(inputs, state, offsets, pages)
    actual = verify(inputs, layer, starts, indices, spec_query_len)
    check_accuracy(actual, golden, record_property, "dispatch_verify")
    expected_state = state.clone()
    for row, page in enumerate(pages):
        expected_state[page].copy_(histories[row][spec_query_len])
    expected_conv = conv.clone()
    expected_conv[1:, :, :HISTORY] = conv[1:, :, spec_query_len - 1 : spec_query_len - 1 + HISTORY]
    context = gdn.GDNRecoverSSMCommitContext.create([layer], spec_query_len=spec_query_len, max_num_reqs=batch)

    def launch():
        context.commit(counts, indices, starts)

    replay_commit(launch, [backing], graph_mode)
    a5 = gdn.is_950()
    selected = grouped if a5 and key == value == 128 and hv % 2 == 0 else ordinary
    assert {name for name, _, _ in launches} == {verify_name, selected}
    for name, grid, args in launches:
        tiles = math.ceil(value / args["BV"])
        if name == verify_name:
            expected_tile = 128 if a5 and batch * hv >= 96 else 64
            assert args["BV"] == min(1 << (value - 1).bit_length(), expected_tile)
            assert grid == (tiles, hv, batch)
        else:
            assert args["SPEC_QUERY_LEN"] == spec_query_len
            heads_per_program = 2 if name == grouped else 1
            assert grid == (tiles * (hv // heads_per_program) * batch,)
            assert args["BV"] * args["BK"] * heads_per_program * 2 * 4 <= gdn.get_ub_size_bytes()
            if name == ordinary:
                assert args["UNROLL_COMMIT"] == a5
    check_accuracy(state, expected_state, record_property, "dispatch_commit")
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    record_property("commit_kernel", selected)
