# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared GDN RecoverSSM test inputs and references."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.gdn.gdn_recoverssm_verify import gdn_recoverssm_verify
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# Qwen/Qwen3.6-35B-A3B and Qwen/Qwen3.8-27B config.json, respectively.
# Both have 16 key heads, K=V=128, and a convolution kernel width of four.
QWEN_35B = 32
QWEN_27B = 48
HISTORY = 3

WINDOW_CASES = [
    (4, 8, 128, 128, 1, torch.float32, False),
    (4, 8, 128, 128, 2, torch.bfloat16, True),
    (4, 8, 128, 128, 5, torch.float32, True),
    (8, 24, 128, 128, 9, torch.bfloat16, False),
    (1, 3, 128, 128, 5, torch.float32, False),
    (2, 6, 33, 47, 4, torch.float32, True),
]


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


def npu_device():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    init_device_properties_triton()
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


def padded_layer(npu, hv, key, value, spec_query_len):
    checkpoint_storage = torch.full((3, hv, value * key + 16), -5.0, device=npu)
    replay_storage = torch.full((3, hv, spec_query_len * (value + key + 1) + 16), -7.0, device=npu)
    checkpoint = checkpoint_storage[..., :-16].unflatten(-1, (value, key))
    replay = replay_storage[..., :-16].unflatten(-1, (spec_query_len, value + key + 1))
    checkpoint.copy_(torch.randn_like(checkpoint) * 0.02)
    layer = SimpleNamespace(kv_cache=(None, checkpoint, replay))
    return layer, checkpoint_storage, replay_storage


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
    return gdn_recoverssm_verify(
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


def ascendc_reference(npu, amplitude):
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    hk, hv, dim, spec_query_len = 4, 12, 128, 5
    layer, _ = packed_layer(npu, 2, hk, hv, dim, dim, spec_query_len, torch.float32, True)
    _, checkpoint, _ = layer.kv_cache
    a_log, a, b, bias, q, k, v = model_inputs(npu, spec_query_len, hk, hv, dim, dim)
    q, k = q * amplitude, k * amplitude
    b.zero_()
    native_state = checkpoint.clone().contiguous()
    page = torch.ones(1, dtype=torch.int32, device=npu)
    span = torch.tensor([0, spec_query_len], dtype=torch.int32, device=npu)
    one_token = torch.tensor([0, 1], dtype=torch.int32, device=npu)

    # Both references receive the same pre-normalized BF16 Q/K.
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
    return inputs, layer, span, page, torch.cat(outputs).unsqueeze(0), prefixes


def trace_launches(monkeypatch, operator, *names):
    """Record grids and options while still running each kernel."""
    launches = []

    class TraceKernel:
        def __init__(self, name):
            self.name = name
            self.kernel = getattr(operator, name)

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append((self.name, grid, kwargs))
                return self.kernel[grid](*args, **kwargs)

            return run

    for name in names:
        monkeypatch.setattr(operator, name, TraceKernel(name))
    return launches
