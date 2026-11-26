# SPDX-License-Identifier: Apache-2.0
"""Fuse the HY4 Prefill gate with CANN reduction order and BF16 rounding."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _add_halves(source, width: tl.constexpr):
    lane = tl.arange(0, width // 2)
    return tl.gather(source, lane, 0) + tl.gather(source, lane + width // 2, 0)


@triton.jit
def _fold_1024(source):
    source = _add_halves(source, 1024)
    source = _add_halves(source, 512)
    source = _add_halves(source, 256)
    return _add_halves(source, 128)


@triton.jit
def _reduce_3072(a, b, c):
    # Combine the first and last 1024 elements before folding to 64 lanes.
    return tl.sum(_fold_1024(a + c) + _fold_1024(b), 0)


@triton.jit
def _rstd(a, b, c, eps: tl.constexpr):
    mean = _reduce_3072(a * a, b * b, c * c) * (1.0 / 3072)
    return tl.div_rn(1.0, tl.sqrt(mean + eps))


@triton.jit
def _prod(key, query, k_inv, q_inv, key_weight, query_weight, index):
    kw = tl.load(key_weight + index).to(tl.float32)
    qw = tl.load(query_weight + index).to(tl.float32)
    # Round both RMSNorm outputs and their product to BF16.
    kn = ((key * k_inv) * kw).to(tl.bfloat16).to(tl.float32)
    qn = ((query * q_inv) * qw).to(tl.bfloat16).to(tl.float32)
    return (kn * qn).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["key_stride"])
def _engram_gate_dot_ordered(
    key, hidden, key_weight, query_weight, dot, key_stride,
    CHANNEL: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 1024)
    k = key + row * key_stride + col
    q = hidden + (row * 4 + CHANNEL) * 3072 + col
    k0 = tl.load(k).to(tl.float32)
    k1 = tl.load(k + 1024).to(tl.float32)
    k2 = tl.load(k + 2048).to(tl.float32)
    q0 = tl.load(q).to(tl.float32)
    q1 = tl.load(q + 1024).to(tl.float32)
    q2 = tl.load(q + 2048).to(tl.float32)
    kinv = _rstd(k0, k1, k2, EPS)
    qinv = _rstd(q0, q1, q2, EPS)
    p0 = _prod(k0, q0, kinv, qinv, key_weight, query_weight, col)
    p1 = _prod(k1, q1, kinv, qinv, key_weight, query_weight, col + 1024)
    p2 = _prod(k2, q2, kinv, qinv, key_weight, query_weight, col + 2048)
    dot_value = _reduce_3072(p0, p1, p2)
    tl.store(dot + row, dot_value.to(tl.bfloat16))


def gate_dot(key, hidden, key_weight, query_weight, channel, epsilon):
    tokens = hidden.shape[0]
    dot = hidden.new_empty(tokens)
    _engram_gate_dot_ordered[(tokens,)](
        key, hidden, key_weight, query_weight, dot, key.stride(0),
        channel, epsilon, enable_fp_fusion=False,
    )
    return dot


@triton.jit
def _load_square(value, offset):
    x = tl.load(value + offset).to(tl.float32)
    return x * x


@triton.jit(do_not_specialize=["value_stride"])
def _engram_gate_value_inverse(value, inverse, value_stride, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 1024)
    offset = row * value_stride + col

    # CANN's 3072-wide row reduction adds the tail to the first half,
    # folds each half into a 64-lane vector, then reduces their sum.
    left = _load_square(value, offset) + _load_square(value, offset + 2048)
    right = _load_square(value, offset + 1024)
    mean = tl.sum(_fold_1024(left) + _fold_1024(right), 0) * (1.0 / 3072)
    tl.store(inverse + row, tl.div_rn(1.0, tl.sqrt(mean + EPS)))


def gate_value_inverse(value, epsilon):
    tokens = value.shape[0]
    inverse = value.new_empty(tokens, dtype=torch.float32)
    _engram_gate_value_inverse[(tokens,)](
        value, inverse, value.stride(0), epsilon, enable_fp_fusion=False,
    )
    return inverse
