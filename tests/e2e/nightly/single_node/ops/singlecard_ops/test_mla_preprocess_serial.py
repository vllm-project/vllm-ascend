# SPDX-License-Identifier: Apache-2.0
"""Device checks for the complete serial MLA API and its cache side effects."""

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.mla_preprocess_serial import (
    mla_preprocess_serial,
    prepare_mla_preprocess_serial_weights,
    prepare_mla_preprocess_single_token_weights,
)
from vllm_ascend.utils import enable_custom_op

HIDDEN = 7168
Q_RANK = 1536
KV_RANK = 512
Q_NOPE = 128
ROPE_DIM = 64
BLOCK_SIZE = 128
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
HEAD_COUNTS = (8, 16, 32, 64, 128)
# Retain all27 original business points while adding realistic decode/TP sizes.
BUSINESS_SHAPES = tuple((t, h) for t in TOKEN_COUNTS for h in HEAD_COUNTS) + tuple(
    (t, h) for t in (257, 1023) for h in (16, 32, 128)
)
CASES = tuple((t, h, False, 1) for t, h in BUSINESS_SHAPES) + (
    (1, 32, True, 2),
    (2, 16, True, 2),
    (64, 32, True, 2),
    (257, 128, True, 2),
)


@pytest.fixture(scope="module", autouse=True)
def load_custom_ops():
    if not torch.npu.get_device_name(0).startswith("Ascend910_93"):
        pytest.skip("Serial MLA Vector operators currently target A3")
    enable_custom_op()


def _make_inputs(tokens, heads, no_rope, cache_stride):
    torch.manual_seed(211)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape) * scale).to(torch.bfloat16).npu()

    slots = torch.arange(tokens, dtype=torch.int32) * 7 + 17
    if cache_stride > 1:
        slots[-1] = -1
    blocks = (tokens * 7 + 17 + BLOCK_SIZE - 1) // BLOCK_SIZE
    angles = torch.randn(tokens, ROPE_DIM // 2).repeat(1, 2)
    cache_base = torch.full((blocks * cache_stride, BLOCK_SIZE, 1, KV_RANK), -3.25, dtype=torch.bfloat16, device="npu")
    rope_base = torch.full((blocks * cache_stride, BLOCK_SIZE, 1, ROPE_DIM), -4.5, dtype=torch.bfloat16, device="npu")
    inputs = (
        rand(tokens, HIDDEN),
        rand(Q_RANK + KV_RANK + ROPE_DIM, HIDDEN, scale=1 / math.sqrt(HIDDEN)),
        rand(heads * (Q_NOPE + ROPE_DIM), Q_RANK, scale=1 / math.sqrt(Q_RANK)),
        rand(heads, KV_RANK, Q_NOPE, scale=1 / math.sqrt(Q_NOPE)),
        (1 + torch.randn(Q_RANK) * 0.1).to(torch.bfloat16).npu(),
        (1 + torch.randn(KV_RANK) * 0.1).to(torch.bfloat16).npu(),
        None if no_rope else angles.cos().to(torch.bfloat16).npu(),
        None if no_rope else angles.sin().to(torch.bfloat16).npu(),
        slots.npu(),
        cache_base[::cache_stride],
        rope_base[::cache_stride],
    )
    return inputs, (cache_base, rope_base)


def _rope(value, cos, sin):
    if cos is None:
        return value
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos + rotated * sin


def _reference(inputs, heads):
    """Independent FP32 CPU specification; no original fused implementation."""
    x, down_w, up_w, key_w, gamma_q, gamma_kv, cos, sin, slots, cache, cache_rope = (
        v.cpu().float() if v is not None and v.is_floating_point() else v.cpu() if v is not None else None
        for v in inputs
    )

    def rms(value, gamma):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * gamma

    down = x @ down_w.T
    latent = rms(down[:, :Q_RANK], gamma_q)
    kv = rms(down[:, Q_RANK : Q_RANK + KV_RANK], gamma_kv)
    up = (latent @ up_w.T).reshape(-1, heads, Q_NOPE + ROPE_DIM)
    query_nope = torch.einsum("thp,hrp->thr", up[..., :Q_NOPE], key_w)
    query_rope = _rope(up[..., Q_NOPE:], None if cos is None else cos[:, None], None if sin is None else sin[:, None])
    key_rope = _rope(down[:, Q_RANK + KV_RANK :], cos, sin)
    cache, cache_rope = cache.clone(), cache_rope.clone()
    written = torch.zeros(cache.shape[:2], dtype=torch.bool)
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, BLOCK_SIZE)
            cache[block, offset, 0] = kv[token]
            cache_rope[block, offset, 0] = key_rope[token]
            written[block, offset] = True
    return (query_nope, query_rope, cache, cache_rope), written


def _assert_close(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    value = actual.cpu().float()
    assert torch.isfinite(value).all() and torch.isfinite(expected).all()
    if value.numel():
        error = (value - expected).abs()
        assert (error <= 2**-6 * (1 + expected.abs())).float().mean() >= 0.99
        assert error.max() <= 1


@pytest.mark.parametrize("tokens,heads,no_rope,cache_stride", CASES)
@pytest.mark.parametrize("graph", [False, True])
def test_mla_preprocess_serial(tokens, heads, no_rope, cache_stride, graph):
    inputs, bases = _make_inputs(tokens, heads, no_rope, cache_stride)
    expected, written = _reference(inputs, heads)
    down, up, key = prepare_mla_preprocess_serial_weights(inputs[1], inputs[2], inputs[3])
    values = (inputs[0], down, up, key, *inputs[4:])
    if graph:
        mla_preprocess_serial(*values)
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            actual = mla_preprocess_serial(*values)
        capture.replay()
    else:
        actual = mla_preprocess_serial(*values)
    torch.npu.synchronize()
    assert actual[2] is inputs[9] and actual[3] is inputs[10]
    assert all(value.device.type == "npu" for value in actual)
    for value, golden in zip(actual[:2], expected[:2]):
        _assert_close(value, golden)
    for value, golden in zip(actual[2:], expected[2:]):
        _assert_close(value.cpu()[written], golden[written])
        assert torch.equal(value.cpu()[~written], golden.to(torch.bfloat16)[~written])
    if cache_stride > 1:
        for base, sentinel in zip(bases, (-3.25, -4.5)):
            assert base[1::2].cpu().eq(sentinel).all()


@pytest.mark.parametrize("invalid", ["dtype", "unpaired_rope"])
def test_mla_preprocess_serial_invalid_input(invalid):
    inputs, _ = _make_inputs(1, 16, False, 1)
    down, up, key = prepare_mla_preprocess_serial_weights(inputs[1], inputs[2], inputs[3])
    values = [inputs[0], down, up, key, *inputs[4:]]
    if invalid == "dtype":
        values[0] = values[0].float()
        message = "BF16 NPU"
    else:
        values[6] = None
        message = "cos and sin"
    with pytest.raises(ValueError, match=message):
        mla_preprocess_serial(*values)


@pytest.mark.parametrize(
    "heads,no_rope,cache_stride", [(h, False, 1) for h in HEAD_COUNTS] + [(32, True, 2), (128, False, 2)]
)
@pytest.mark.parametrize("graph", [False, True])
def test_mla_preprocess_single_token_weights(heads, no_rope, cache_stride, graph):
    inputs, bases = _make_inputs(1, heads, no_rope, cache_stride)
    down, up, key = prepare_mla_preprocess_serial_weights(inputs[1], inputs[2], inputs[3])
    single = prepare_mla_preprocess_single_token_weights(down, up)
    for original, prepared in zip((down, up), single):
        assert prepared.shape == original.shape and prepared.dtype == original.dtype
        assert torch.equal(prepared.cpu().contiguous().view(torch.int16), original.cpu().contiguous().view(torch.int16))
    values = (inputs[0], down, up, key, *inputs[4:])

    def run():
        return mla_preprocess_serial(*values, single_token_weights=single)

    if graph:
        run()
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            actual = run()
    for update in (0, 0.125) if graph else (0,):
        if update:
            inputs[0].add_(update)
        for base, sentinel in zip(bases, (-3.25, -4.5)):
            base.fill_(sentinel)
        expected, written = _reference(inputs, heads)
        if graph:
            capture.replay()
        else:
            actual = run()
        torch.npu.synchronize()
        assert actual[2] is inputs[9] and actual[3] is inputs[10]
        for value, golden in zip(actual[:2], expected[:2]):
            _assert_close(value, golden)
        for value, golden in zip(actual[2:], expected[2:]):
            _assert_close(value.cpu()[written], golden[written])
            assert torch.equal(value.cpu()[~written], golden.to(torch.bfloat16)[~written])
        if cache_stride > 1:
            for base, sentinel in zip(bases, (-3.25, -4.5)):
                assert base[1::2].cpu().eq(sentinel).all()
