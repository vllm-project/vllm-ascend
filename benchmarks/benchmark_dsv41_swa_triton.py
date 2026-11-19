"""Standalone Ascend benchmark for DeepSeek V4.1 SWA cache preparation."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from subprocess import CalledProcessError
from typing import NamedTuple

import torch
import torch_npu
import triton
import triton.language as tl
from triton.compiler import errors as triton_errors

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
EPS = 1e-6
SENTINEL = -31.0
ATOL = 0.035
RTOL = 0.01
BASELINE_REV = "5568aa082d609e7079e4e9a4c35f38c3af9c99ad"
COMPILER_ERRORS = (
    triton_errors.CompilationError,
    getattr(triton_errors, "MLIRCompilationError", triton_errors.CompilationError),
    CalledProcessError,
)
KERNEL_HEAD_DIM = tl.constexpr(HEAD_DIM)
KERNEL_NOPE_DIM = tl.constexpr(NOPE_DIM)
KERNEL_ROPE_DIM = tl.constexpr(ROPE_DIM)
KERNEL_EPS = tl.constexpr(EPS)
try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice
except ImportError:
    extract_slice = getattr(tl, "extract_slice", None)
    insert_slice = getattr(tl, "insert_slice", None)


@triton.jit
def _pr21183_kernel(
    projected_kv,
    norm_weight,
    cos,
    sin,
    cache,
    slot_mapping,
    KV_TOKEN_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    COS_DIM_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SIN_DIM_STRIDE: tl.constexpr,
    SLOT_TOKEN_STRIDE: tl.constexpr,
    SLOT_DIM_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_DIM_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, HEAD_DIM)
    row = projected_kv + token * KV_TOKEN_STRIDE

    value = tl.load(row + dims).to(tl.float32)
    weight = tl.load(norm_weight + dims).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)

    normalized = (value * inverse_rms * weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    rotary_index = dims - NOPE_DIM
    rotary_mask = dims >= NOPE_DIM
    even = (rotary_index % 2) == 0
    pair_dims = tl.where(rotary_mask, dims + tl.where(even, 1, -1), 0)
    pair_value = tl.load(row + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(norm_weight + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_normalized = (pair_value * inverse_rms * pair_weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    cos_value = tl.load(
        cos + token * COS_TOKEN_STRIDE + rotary_index * COS_DIM_STRIDE,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    sin_value = tl.load(
        sin + token * SIN_TOKEN_STRIDE + rotary_index * SIN_DIM_STRIDE,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    sin_sign = tl.where(even, -1.0, 1.0)
    rotated = normalized * cos_value + pair_normalized * sin_value * sin_sign
    output = tl.where(rotary_mask, rotated, normalized)

    slot_row = slot_mapping + token * SLOT_TOKEN_STRIDE
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + SLOT_DIM_STRIDE).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    cache_offsets = block * CACHE_BLOCK_STRIDE + offset * CACHE_TOKEN_STRIDE + dims * CACHE_DIM_STRIDE
    tl.store(cache + cache_offsets, output, mask=valid)


@triton.jit
def _pr21581_single_layer_partial_kernel(
    kv,
    weight,
    cos,
    sin,
    cache,
    slots,
    token,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    dims = tl.arange(0, KERNEL_HEAD_DIM)
    value = tl.load(kv + token * KV_STRIDE + dims).to(tl.float32)
    norm_weight = tl.load(weight + dims).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / KERNEL_HEAD_DIM + KERNEL_EPS)
    normalized = (value * inv_rms * norm_weight).to(tl.bfloat16).to(tl.float32)
    rope_dims = tl.arange(0, KERNEL_ROPE_DIM)
    rope_value = tl.load(kv + token * KV_STRIDE + KERNEL_NOPE_DIM + rope_dims).to(tl.float32)
    rope_weight = tl.load(weight + KERNEL_NOPE_DIM + rope_dims).to(tl.float32)
    rope = (rope_value * inv_rms * rope_weight).to(tl.bfloat16).to(tl.float32)
    even = rope_dims % 2 == 0
    paired = tl.gather(rope, tl.where(even, rope_dims + 1, rope_dims - 1), 0)
    cos_value = tl.load(cos + token * COS_STRIDE + rope_dims).to(tl.float32)
    sin_value = tl.load(sin + token * SIN_STRIDE + rope_dims).to(tl.float32)
    rotated = rope * cos_value + tl.where(even, -paired, paired) * sin_value

    slot_row = slots + token * SLOT_STRIDE
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + 1).to(tl.int64)
    if (block >= 0) & (block < CACHE_BLOCKS) & (offset >= 0) & (offset < CACHE_BLOCK_SIZE):
        cache_row = cache + block * PAGE_STRIDE + offset * CACHE_ROW_STRIDE
        tl.store(cache_row + dims, normalized.to(tl.bfloat16), dims < KERNEL_NOPE_DIM)
        tl.store(cache_row + KERNEL_NOPE_DIM + rope_dims, rotated.to(tl.bfloat16))


@triton.jit(do_not_specialize=["num_tokens"])
def _pr21581_single_layer_kernel(
    kv,
    weight,
    cos,
    sin,
    cache,
    slots,
    num_tokens,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    for token in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        _pr21581_single_layer_partial_kernel(
            kv,
            weight,
            cos,
            sin,
            cache,
            slots,
            token.to(tl.int64),
            KV_STRIDE,
            COS_STRIDE,
            SIN_STRIDE,
            SLOT_STRIDE,
            PAGE_STRIDE,
            CACHE_ROW_STRIDE,
            CACHE_BLOCKS,
            CACHE_BLOCK_SIZE,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _current_cyclic_kernel(
    projected_kv,
    norm_weight,
    cos,
    sin,
    cache,
    slot_mapping,
    num_tokens,
    KV_TOKEN_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SLOT_TOKEN_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    nope_dim: tl.constexpr = HEAD_DIM - ROPE_DIM
    dims = tl.arange(0, HEAD_DIM)
    rotary_dims = tl.arange(0, ROPE_DIM)
    weight = tl.load(norm_weight + dims).to(tl.float32)
    for row_index in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        token = row_index.to(tl.int64)
        slot_row = slot_mapping + token * SLOT_TOKEN_STRIDE
        block = tl.load(slot_row).to(tl.int64)
        offset = tl.load(slot_row + 1).to(tl.int64)
        valid = (block >= 0) & (offset >= 0)
        cache_row = cache + tl.where(valid, block * CACHE_BLOCK_STRIDE + offset * CACHE_TOKEN_STRIDE, 0)

        value = tl.load(projected_kv + token * KV_TOKEN_STRIDE + dims).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)
        normalized = (value * inverse_rms * weight).to(tl.bfloat16)
        rotary = extract_slice(normalized, (nope_dim,), (ROPE_DIM,), (1,)).to(tl.float32)
        cos_value = tl.load(cos + token * COS_TOKEN_STRIDE + rotary_dims).to(tl.float32)
        sin_value = tl.load(sin + token * SIN_TOKEN_STRIDE + rotary_dims).to(tl.float32)
        partner = tl.gather(rotary, rotary_dims ^ 1, 0)
        rotated = rotary * cos_value + partner * sin_value * tl.where(rotary_dims % 2 == 0, -1.0, 1.0)

        tl.store(cache_row + dims, normalized, valid & (dims < nope_dim))
        tl.store(cache_row + nope_dim + rotary_dims, rotated, valid)


@triton.jit
def _write_row(cache_row, dims, normalized, rotary_dim, rotated, valid, SPLIT: tl.constexpr, SLOT_POLICY: tl.constexpr):
    if SLOT_POLICY != "bounded" or valid:
        write_mask = True if SLOT_POLICY == "bounded" else valid
        if SPLIT:
            tl.store(cache_row + dims, normalized, write_mask & (dims < KERNEL_NOPE_DIM))
            tl.store(cache_row + KERNEL_NOPE_DIM + rotary_dim, rotated, write_mask)
        else:
            tl.store(cache_row + dims, normalized, write_mask)


@triton.jit
def _store_row(
    row,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    cache_row,
    valid,
    dims,
    weights,
    ROPE_IMPL: tl.constexpr,
    MERGE: tl.constexpr,
    SLOT_POLICY: tl.constexpr,
):
    value = tl.load(row + dims).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, 0) / KERNEL_HEAD_DIM + KERNEL_EPS)
    normalized = (value * inverse_rms * weights).to(tl.bfloat16)
    if ROPE_IMPL == "gather512" or ROPE_IMPL == "masked512":
        rotary_dim = dims - KERNEL_NOPE_DIM
        rotary_mask = dims >= KERNEL_NOPE_DIM
        if ROPE_IMPL == "masked512":
            mate_dim = tl.where(rotary_mask, dims ^ 1, 0)
            mate_value = tl.load(row + mate_dim, rotary_mask, other=0).to(tl.float32)
            mate_weight = tl.load(weight_ptr + mate_dim, rotary_mask, other=0).to(tl.float32)
            mate = (mate_value * inverse_rms * mate_weight).to(tl.bfloat16).to(tl.float32)
        else:
            mate = tl.gather(normalized.to(tl.float32), dims ^ 1, 0)
            rotary_dim = tl.minimum(tl.maximum(rotary_dim, 0), KERNEL_ROPE_DIM - 1)
        cos = tl.load(cos_ptr + rotary_dim, rotary_mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + rotary_dim, rotary_mask, other=0.0).to(tl.float32)
        rotated = normalized.to(tl.float32) * cos + mate * sin * tl.where(dims % 2 == 0, -1.0, 1.0)
        output = tl.where(rotary_mask, rotated, normalized.to(tl.float32))
        _write_row(cache_row, dims, output, 0, 0, valid, False, SLOT_POLICY)
    elif ROPE_IMPL == "pair512":
        pairs = tl.arange(0, KERNEL_HEAD_DIM // 2)
        rope_mask = pairs >= KERNEL_NOPE_DIM // 2
        pair_offset = 2 * tl.maximum(pairs - KERNEL_NOPE_DIM // 2, 0)
        even, odd = tl.split(tl.reshape(normalized.to(tl.float32), (KERNEL_HEAD_DIM // 2, 2)))
        cos_even = tl.load(cos_ptr + pair_offset, rope_mask, other=1.0).to(tl.float32)
        cos_odd = tl.load(cos_ptr + pair_offset + 1, rope_mask, other=1.0).to(tl.float32)
        sin_even = tl.load(sin_ptr + pair_offset, rope_mask, other=0.0).to(tl.float32)
        sin_odd = tl.load(sin_ptr + pair_offset + 1, rope_mask, other=0.0).to(tl.float32)
        rotated = tl.reshape(
            tl.join(even * cos_even - odd * sin_even, odd * cos_odd + even * sin_odd), (KERNEL_HEAD_DIM,)
        )
        output = tl.where(dims >= KERNEL_NOPE_DIM, rotated, normalized.to(tl.float32))
        _write_row(cache_row, dims, output, 0, 0, valid, False, SLOT_POLICY)
    else:
        rotary_dim = tl.arange(0, KERNEL_ROPE_DIM)
        if ROPE_IMPL == "gather64" or ROPE_IMPL == "load64":
            rope_value = tl.load(row + KERNEL_NOPE_DIM + rotary_dim).to(tl.float32)
            rope_weight = tl.load(weight_ptr + KERNEL_NOPE_DIM + rotary_dim).to(tl.float32)
            rope = (rope_value * inverse_rms * rope_weight).to(tl.bfloat16).to(tl.float32)
        else:
            rope = extract_slice(normalized, (KERNEL_NOPE_DIM,), (KERNEL_ROPE_DIM,), (1,)).to(tl.float32)
        cos = tl.load(cos_ptr + rotary_dim).to(tl.float32)
        sin = tl.load(sin_ptr + rotary_dim).to(tl.float32)
        if ROPE_IMPL == "slice-pair64":
            even, odd = tl.split(tl.reshape(rope, (KERNEL_ROPE_DIM // 2, 2)))
            cos_even, cos_odd = tl.split(tl.reshape(cos, (KERNEL_ROPE_DIM // 2, 2)))
            sin_even, sin_odd = tl.split(tl.reshape(sin, (KERNEL_ROPE_DIM // 2, 2)))
            rotated = tl.reshape(
                tl.join(even * cos_even - odd * sin_even, odd * cos_odd + even * sin_odd), (KERNEL_ROPE_DIM,)
            )
        else:
            mate_dim = rotary_dim ^ 1
            if ROPE_IMPL == "load64":
                mate_value = tl.load(row + KERNEL_NOPE_DIM + mate_dim).to(tl.float32)
                mate_weight = tl.load(weight_ptr + KERNEL_NOPE_DIM + mate_dim).to(tl.float32)
                mate = (mate_value * inverse_rms * mate_weight).to(tl.bfloat16).to(tl.float32)
            else:
                mate = tl.gather(rope, mate_dim, 0)
            rotated = rope * cos + mate * sin * tl.where(rotary_dim % 2 == 0, -1.0, 1.0)
        if MERGE:
            output = insert_slice(normalized, rotated.to(tl.bfloat16), (KERNEL_NOPE_DIM,), (KERNEL_ROPE_DIM,), (1,))
            _write_row(cache_row, dims, output, 0, 0, valid, False, SLOT_POLICY)
        else:
            _write_row(cache_row, dims, normalized, rotary_dim, rotated, valid, True, SLOT_POLICY)


@triton.jit
def _store_token(
    kv,
    weight,
    cos,
    sin,
    cache,
    slots,
    token,
    dims,
    weights,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    ROPE_IMPL: tl.constexpr,
    MERGE: tl.constexpr,
    SLOT_POLICY: tl.constexpr,
):
    slot_row = slots + token * SLOT_STRIDE
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + 1).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    if SLOT_POLICY == "bounded":
        valid = valid & (block < CACHE_BLOCKS) & (offset < CACHE_BLOCK_SIZE)
    cache_row = cache + tl.where(valid, block * PAGE_STRIDE + offset * CACHE_ROW_STRIDE, 0)
    if SLOT_POLICY != "early" or valid:
        _store_row(
            kv + token * KV_STRIDE,
            weight,
            cos + token * COS_STRIDE,
            sin + token * SIN_STRIDE,
            cache_row,
            valid,
            dims,
            weights,
            ROPE_IMPL,
            MERGE,
            SLOT_POLICY,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _row_kernel(
    kv,
    weight,
    cos,
    sin,
    cache,
    slots,
    num_tokens,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    ROPE_IMPL: tl.constexpr,
    MERGE: tl.constexpr,
    SLOT_POLICY: tl.constexpr,
    SCHEDULE: tl.constexpr,
):
    dims = tl.arange(0, KERNEL_HEAD_DIM)
    weights = tl.load(weight + dims).to(tl.float32)
    program = tl.program_id(0)
    if SCHEDULE == "direct":
        _store_token(
            kv,
            weight,
            cos,
            sin,
            cache,
            slots,
            program.to(tl.int64),
            dims,
            weights,
            KV_STRIDE,
            COS_STRIDE,
            SIN_STRIDE,
            SLOT_STRIDE,
            PAGE_STRIDE,
            CACHE_ROW_STRIDE,
            CACHE_BLOCKS,
            CACHE_BLOCK_SIZE,
            ROPE_IMPL,
            MERGE,
            SLOT_POLICY,
        )
    else:
        if SCHEDULE == "blocked":
            rows = tl.cdiv(num_tokens, tl.num_programs(0))
            start = program * rows
            stop = tl.minimum(start + rows, num_tokens)
            step = 1
        else:
            start = program
            stop = num_tokens
            step = tl.num_programs(0)
        for token in range(start, stop, step):
            _store_token(
                kv,
                weight,
                cos,
                sin,
                cache,
                slots,
                token.to(tl.int64),
                dims,
                weights,
                KV_STRIDE,
                COS_STRIDE,
                SIN_STRIDE,
                SLOT_STRIDE,
                PAGE_STRIDE,
                CACHE_ROW_STRIDE,
                CACHE_BLOCKS,
                CACHE_BLOCK_SIZE,
                ROPE_IMPL,
                MERGE,
                SLOT_POLICY,
            )


@triton.jit(do_not_specialize=["num_tokens"])
def _tile_kernel(
    kv,
    weight,
    cos,
    sin,
    cache,
    slots,
    num_tokens,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    dims = tl.arange(0, KERNEL_HEAD_DIM)
    weights = tl.load(weight + dims).to(tl.float32)
    rope_dim = tl.arange(0, KERNEL_ROPE_DIM)
    program = tl.program_id(0)
    stop = tl.cdiv(num_tokens, ROWS) if PERSISTENT else program + 1
    step = tl.num_programs(0) if PERSISTENT else 1
    for tile in range(program, stop, step):
        tokens = (tile * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        in_bounds = tokens < num_tokens
        tokens = tl.minimum(tokens, num_tokens - 1)
        block = tl.load(slots + tokens * SLOT_STRIDE).to(tl.int64)
        offset = tl.load(slots + tokens * SLOT_STRIDE + 1).to(tl.int64)
        valid = in_bounds & (block >= 0) & (offset >= 0)
        cache_rows = tl.where(valid, block * PAGE_STRIDE + offset * CACHE_ROW_STRIDE, 0)
        value = tl.load(kv + tokens[:, None] * KV_STRIDE + dims[None, :]).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(value * value, 1) / KERNEL_HEAD_DIM + KERNEL_EPS)
        normalized = (value * inverse_rms[:, None] * weights[None, :]).to(tl.bfloat16)
        rope = extract_slice(normalized, (0, KERNEL_NOPE_DIM), (ROWS, KERNEL_ROPE_DIM), (1, 1)).to(tl.float32)
        cos_value = tl.load(cos + tokens[:, None] * COS_STRIDE + rope_dim[None, :]).to(tl.float32)
        sin_value = tl.load(sin + tokens[:, None] * SIN_STRIDE + rope_dim[None, :]).to(tl.float32)
        even, odd = tl.split(tl.reshape(rope, (ROWS, KERNEL_ROPE_DIM // 2, 2)))
        cos_even, cos_odd = tl.split(tl.reshape(cos_value, (ROWS, KERNEL_ROPE_DIM // 2, 2)))
        sin_even, sin_odd = tl.split(tl.reshape(sin_value, (ROWS, KERNEL_ROPE_DIM // 2, 2)))
        rotated = tl.reshape(
            tl.join(even * cos_even - odd * sin_even, odd * cos_odd + even * sin_odd), (ROWS, KERNEL_ROPE_DIM)
        )
        output = insert_slice(
            normalized, rotated.to(tl.bfloat16), (0, KERNEL_NOPE_DIM), (ROWS, KERNEL_ROPE_DIM), (1, 1)
        )
        tl.store(cache + cache_rows[:, None] + dims[None, :], output, valid[:, None])


class Variant(NamedTuple):
    name: str
    rope: str
    merge: bool = False
    slot_policy: str = "masked"
    rows: int = 1


VARIANTS = (
    Variant("masked512", "masked512"),
    Variant("gather512", "gather512"),
    Variant("gather512-bounded", "gather512", slot_policy="bounded"),
    Variant("gather64", "gather64"),
    Variant("gather64-bounded", "gather64", slot_policy="bounded"),
    Variant("load64", "load64"),
    Variant("pair512", "pair512"),
    Variant("slice-gather64-split", "slice-gather64"),
    Variant("slice-gather64", "slice-gather64", merge=True),
    Variant("slice-pair64", "slice-pair64", merge=True),
    Variant("slice-pair64-skip", "slice-pair64", merge=True, slot_policy="early"),
    Variant("tile2", "slice-pair64", merge=True, rows=2),
    Variant("tile4", "slice-pair64", merge=True, rows=4),
    Variant("tile8", "slice-pair64", merge=True, rows=8),
)
DEFAULT_VARIANTS = ("masked512", "gather64-bounded", "slice-gather64-split", "slice-pair64")
SCHEDULES = ("direct", "cyclic", "blocked", "cyclic-half", "cyclic2")
MODES = ("eager", "graph", "graph-batch")
LAYOUTS = ("paged", "shuffled-padded", "graph-tail")
TOKEN_COUNTS = (1024, 4096)
PROGRAM_COUNTS = (20, 40, 80)
DISPLAY_TOP = 10
BASELINE = "pr21183-masked512/direct"
NATIVE = "native-3op"
ADAPTED = "pr21581-gather64/single-layer"
CURRENT = "current-gather64-split/cyclic"


class Case(NamedTuple):
    projected: torch.Tensor
    weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    storage: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    replay_slots: torch.Tensor
    expected: torch.Tensor
    written: torch.Tensor
    replay_expected: torch.Tensor
    replay_written: torch.Tensor


class ValidationError(Exception):
    pass


def args_from_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only compare PR21183, single-layer PR21581 and the cyclic-only current kernel in eager mode",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=TOKEN_COUNTS,
        help="Row counts to test (default: 1024 4096)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=[variant.name for variant in VARIANTS],
        default=DEFAULT_VARIANTS,
        help="Candidates to test (default: %(default)s); tile4 and tile8 may hit Ascend UB and layout constraints",
    )
    parser.add_argument("--schedules", nargs="+", choices=SCHEDULES, default=["cyclic", "blocked"])
    parser.add_argument(
        "--programs",
        nargs="+",
        type=int,
        default=PROGRAM_COUNTS,
        help="Requested 1D program counts for cyclic/blocked, capped at the number of work items (default: 20 40 80)",
    )
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=["paged"], help="Default layout: paged")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["eager", "graph-batch"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vector-cores", type=int, help="Override Triton's detected number of vector cores")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--page-padding", type=int, default=8, help="Extra physical rows for padded layouts")
    parser.add_argument("--row-padding", type=int, default=0, help="Extra elements per cache row in padded layouts")
    parser.add_argument("--slot-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12, help="Calls per timing sample")
    parser.add_argument("--samples", type=int, default=5, help="Interleaved timing rounds")
    parser.add_argument(
        "--check-only", action="store_true", help="Validate eager and selected graph modes without timing"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Try other candidates after backend exceptions"
    )
    parser.add_argument("--native-library", type=Path, help="Optional trusted local library registering _C_ascend ops")
    parser.add_argument("--json", type=Path, dest="output", help="Write metadata, validation and raw timing samples")
    parser.add_argument("--verbose", action="store_true", help="Show ranked tables instead of concise RESULT lines")
    args = parser.parse_args()
    if min(*args.tokens, *args.programs, args.block_size, args.repeats, args.samples) < 1:
        parser.error("sizes, repeats and samples must be positive")
    if min(args.warmups, args.page_padding, args.row_padding) < 0:
        parser.error("warmups and padding must be nonnegative")
    if args.vector_cores is not None and args.vector_cores < 1:
        parser.error("--vector-cores must be positive")
    if args.quick:
        args.layouts = ["paged"]
        args.modes = ["eager"]
    return args


def reference(projected, weight, cos, sin):
    value = projected.float()
    normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS) * weight.float()).bfloat16()
    rope = normalized[:, NOPE_DIM:].float()
    rotated = torch.empty_like(rope)
    rotated[:, 0::2] = rope[:, 0::2] * cos[:, 0::2].float() - rope[:, 1::2] * sin[:, 0::2].float()
    rotated[:, 1::2] = rope[:, 1::2] * cos[:, 1::2].float() + rope[:, 0::2] * sin[:, 1::2].float()
    return torch.cat((normalized[:, :NOPE_DIM], rotated.bfloat16()), dim=1)


def expected_storage(shape, slots, values):
    expected = torch.full(shape, SENTINEL, dtype=torch.bfloat16)
    written = torch.zeros(shape[:2], dtype=torch.bool)
    valid = (slots >= 0).all(dim=1)
    blocks, offsets = slots[valid].long().unbind(dim=1)
    written[blocks, offsets] = True
    expected[blocks, offsets] = values[valid]
    return expected, written


def make_case(tokens, layout, args, device):
    rand = torch.Generator().manual_seed(args.seed + tokens)
    page_size = args.block_size
    padded = layout != "paged"
    blocks = 2 * triton.cdiv(tokens, page_size) + 1
    if layout == "shuffled-padded":
        flat = torch.randperm((blocks - 1) * page_size, generator=rand)[:tokens]
    else:
        pages = torch.randperm(blocks - 1, generator=rand)
        logical = torch.arange(tokens)
        flat = pages[logical // page_size] * page_size + logical % page_size
    slots = torch.stack((flat // page_size, flat % page_size), dim=1).to(getattr(torch, args.slot_dtype))
    if layout == "graph-tail":
        first_padding = 3 * tokens // 4
        slots[first_padding:] = -1
    replay_slots = slots.roll(1, dims=0)
    valid = (replay_slots >= 0).all(dim=1)
    replay_slots[valid, 0] += 1
    if layout == "graph-tail" and tokens == 1:
        replay_slots[0] = torch.tensor([blocks - 1, 0], dtype=slots.dtype)

    projected = torch.randn(tokens, HEAD_DIM, generator=rand, dtype=torch.bfloat16)
    weight = (1 + 0.25 * torch.randn(HEAD_DIM, generator=rand)).bfloat16()
    angles = torch.randn(tokens, ROPE_DIM // 2, generator=rand)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    shape = (blocks, page_size + args.page_padding * padded, HEAD_DIM)
    expected, written = expected_storage(shape, slots, reference(projected, weight, cos, sin))
    replay_expected, replay_written = expected_storage(shape, replay_slots, reference(-projected, weight, cos, sin))

    def padded_rows(tensor, extra):
        view = torch.empty((tokens, tensor.shape[1] + extra * padded), dtype=tensor.dtype, device=device)
        view = view[:, : tensor.shape[1]]
        return view.copy_(tensor)

    storage = torch.empty((*shape[:2], HEAD_DIM + args.row_padding * padded), device=device, dtype=torch.bfloat16)
    return Case(
        padded_rows(projected, 16),
        weight.to(device),
        padded_rows(cos, 8),
        padded_rows(sin, 16),
        storage,
        storage[:, :page_size, :HEAD_DIM],
        padded_rows(slots, 2),
        replay_slots.to(device),
        expected,
        written,
        replay_expected,
        replay_written,
    )


def launchers(case, args, vector_cores):
    projected, weight, cos, sin, _, cache, slots, *_ = case
    tokens = projected.shape[0]
    pointers = (projected, weight, cos, sin, cache, slots)
    strides = (projected.stride(0), cos.stride(0), sin.stride(0), slots.stride(0), cache.stride(0), cache.stride(1))

    def baseline():
        _pr21183_kernel[(tokens,)](
            *pointers,
            strides[0],
            strides[1],
            1,
            strides[2],
            1,
            strides[3],
            1,
            strides[4],
            strides[5],
            1,
            HEAD_DIM,
            NOPE_DIM,
            EPS,
            enable_fp_fusion=False,
        )

    yield BASELINE, tokens, baseline
    if args.native_library or args.quick:

        def native():
            value, _ = torch_npu.npu_rms_norm(projected, weight, EPS)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                value.view(tokens, 1, 1, HEAD_DIM),
                cos.view(tokens, 1, 1, ROPE_DIM),
                sin.view(tokens, 1, 1, ROPE_DIM),
                rotary_mode="interleave",
                partial_slice=[NOPE_DIM, HEAD_DIM],
            )
            torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, slots, value.contiguous())

        yield NATIVE, None, native
    if args.quick:
        programs = vector_cores
        blocks, block_size = cache.shape[:2]

        def adapted():
            _pr21581_single_layer_kernel[(programs,)](
                *pointers, tokens, *strides, blocks, block_size, enable_fp_fusion=False
            )

        def current():
            _current_cyclic_kernel[(programs,)](
                *pointers, tokens, *strides, HEAD_DIM, ROPE_DIM, EPS, enable_fp_fusion=False
            )

        yield ADAPTED, programs, adapted
        yield CURRENT, programs, current
        return
    for variant in VARIANTS:
        if variant.name not in args.variants:
            continue
        tasks = triton.cdiv(tokens, variant.rows)
        seen = set()
        for schedule in args.schedules:
            if variant.rows > 1 and schedule == "blocked":
                continue
            for scheduling, programs in program_grid(schedule, tasks, args.programs, vector_cores):
                key = (scheduling, programs)
                if key in seen:
                    continue
                seen.add(key)
                if variant.rows == 1:

                    def run(variant=variant, scheduling=scheduling, programs=programs):
                        _row_kernel[(programs,)](
                            *pointers,
                            tokens,
                            *strides,
                            cache.shape[0] if variant.slot_policy == "bounded" else 0,
                            cache.shape[1] if variant.slot_policy == "bounded" else 0,
                            variant.rope,
                            variant.merge,
                            variant.slot_policy,
                            scheduling,
                            enable_fp_fusion=False,
                        )
                else:

                    def run(variant=variant, scheduling=scheduling, programs=programs):
                        _tile_kernel[(programs,)](
                            *pointers,
                            tokens,
                            *strides,
                            variant.rows,
                            scheduling != "direct",
                            enable_fp_fusion=False,
                        )

                yield f"{variant.name}/{scheduling}@{programs}", programs, run


def program_grid(schedule, tasks, requested, vector_cores):
    if schedule == "direct":
        return [(schedule, tasks)]
    if schedule == "cyclic-half":
        return [("cyclic", min(tasks, max(1, vector_cores // 2)))]
    if schedule == "cyclic2":
        return [("cyclic", min(tasks, 2 * vector_cores))]
    return [(schedule, min(tasks, count)) for count in requested]


def validate(storage, expected, written, baseline_rows=None):
    actual = storage.cpu()
    if not (actual[..., HEAD_DIM:] == SENTINEL).all():
        raise ValidationError("physical cache row padding was overwritten")
    actual = actual[..., :HEAD_DIM]
    if not (actual[~written] == SENTINEL).all():
        raise ValidationError("cache hole, negative slot or physical page padding was overwritten")
    actual_rows = actual[written]
    reference_rows = expected[written]
    stats = {"written_rows": len(actual_rows)}
    for name, dim_slice in (("nope", slice(None, NOPE_DIM)), ("rope", slice(NOPE_DIM, None))):
        found = actual_rows[:, dim_slice].float()
        wanted = reference_rows[:, dim_slice].float()
        error = (found - wanted).abs()
        if not torch.isfinite(found).all() or not (error <= ATOL + RTOL * wanted.abs()).all():
            raise ValidationError(f"{name} values differ from the BF16 reference (max error {float(error.max()):.6g})")
        stats[name] = {
            "max_abs_error": float(error.max()) if error.numel() else None,
            "exact_vs_cpu": float((found == wanted).float().mean()) if error.numel() else None,
            "exact_vs_pr21183": float((found == baseline_rows[:, dim_slice].float()).float().mean())
            if error.numel() and baseline_rows is not None
            else None,
        }
    return stats, actual_rows


def capture(run, calls):
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            run()
    return graph


def measure(run, calls, divisor, start, end):
    clock = time.perf_counter_ns
    begin = clock()
    start.record()
    for _ in range(calls):
        run()
    end.record()
    submitted = clock()
    end.synchronize()
    finished = clock()
    return {
        "event_us": start.elapsed_time(end) * 1000 / divisor,
        "wall_us": (finished - begin) / 1000 / divisor,
        "submit_us": (submitted - begin) / 1000 / divisor,
    }


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def native_library_path(explicit_path):
    if explicit_path:
        return explicit_path.resolve(strict=True)
    spec = find_spec("vllm_ascend")
    directories = spec.submodule_search_locations if spec else None
    for directory in directories or ():
        libraries = sorted(Path(directory).glob("vllm_ascend_C*.so"))
        if len(libraries) == 1:
            return libraries[0].resolve()
    raise SystemExit("Original path requires vllm_ascend_C and its CANN custom ops; pass --native-library /path/to/.so")


def load_native_library(path):
    vendor = path.parent / "_cann_ops_custom" / "vendors" / "custom_transformer"
    if vendor.is_dir():
        current_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
            dict.fromkeys([str(vendor), *filter(None, current_paths)])
        )
    torch.ops.load_library(str(path))
    for operator in ("inplace_partial_rotary_mul", "npu_scatter_nd_update_sk"):
        if not hasattr(torch.ops._C_ascend, operator):
            raise SystemExit(f"{path} did not register _C_ascend.{operator}; select the matching Ascend extension")


def compare_native(actual, native, written):
    found_rows = actual[..., :HEAD_DIM][written].float()
    native_rows = native[..., :HEAD_DIM][written].float()
    stats = {
        "untouched_ok": bool((actual[..., :HEAD_DIM][~written] == SENTINEL).all())
        and bool((actual[..., HEAD_DIM:] == SENTINEL).all())
    }
    for name, features in (("nope", slice(None, NOPE_DIM)), ("rope", slice(NOPE_DIM, None))):
        found = found_rows[:, features]
        expected = native_rows[:, features]
        error = (found - expected).abs()
        stats[name] = {
            "exact_pct": 100 * float((found == expected).sum()) / found.numel(),
            "max_abs_error": float(error.max()) if torch.isfinite(error).all() else None,
            "within_tolerance": bool((error <= ATOL + RTOL * expected.abs()).all()),
        }
    stats["passed"] = stats["untouched_ok"] and all(stats[name]["within_tolerance"] for name in ("nope", "rope"))
    return stats


def run_quick(tokens, args, vector_cores, device):
    case = make_case(tokens, "paged", args, device)
    launches = {name: (programs, run) for name, programs, run in launchers(case, args, vector_cores)}
    results = {name: {"programs": programs} for name, (programs, _) in launches.items()}

    def snapshot(run):
        case.storage.fill_(SENTINEL)
        run()
        torch.npu.synchronize()
        return case.storage.cpu()

    native = snapshot(launches[NATIVE][1])
    actual_rows = native[..., :HEAD_DIM][case.written]
    if (actual_rows == SENTINEL).all(dim=1).any() or not compare_native(native, native, case.written)["passed"]:
        raise ValidationError("original three-op path left invalid cache contents")
    runnable = {}
    for name, (_, run) in launches.items():
        if name != NATIVE:
            try:
                results[name]["check_vs_native"] = compare_native(snapshot(run), native, case.written)
            except COMPILER_ERRORS as exc:
                results[name]["compile_error"] = str(exc)
                continue
        runnable[name] = run

    if args.check_only:
        return results
    for run in runnable.values():
        for _ in range(args.warmups):
            run()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    end.synchronize()
    samples = {name: [] for name in runnable}
    order = list(runnable)
    rng = random.Random(args.seed + tokens)
    for _ in range(args.samples):
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(runnable[name], args.repeats, args.repeats, start, end))
    for name, values in samples.items():
        results[name]["samples"] = values
        for metric in ("event_us", "wall_us"):
            results[name][metric] = quantile([sample[metric] for sample in values], 0.5)
    return results


def format_native_error(check, feature):
    if check is None:
        return "reference"
    stats = check[feature]
    maximum = stats["max_abs_error"]
    max_error = f"{maximum:.6g}" if maximum is not None else "nonfinite"
    return f"{stats['exact_pct']:.6f}%/{max_error}"


def print_quick(tokens, results):
    print(f"T={tokens} eager; error vs {NATIVE} (exact %, max abs)", flush=True)
    print(f"{'version':32} {'grid':>6} {'event us':>10} {'wall us':>10} {'NoPE':>21} {'RoPE':>21}  status")
    passed = True
    for name, result in results.items():
        if "compile_error" in result:
            detail = result["compile_error"].strip().splitlines()
            print(f"{name:32} COMPILE FAIL: {detail[-1] if detail else 'see --json'}", flush=True)
            passed = False
            continue

        event = f"{result['event_us']:.3f}" if "event_us" in result else "-"
        wall = f"{result['wall_us']:.3f}" if "wall_us" in result else "-"
        check = result.get("check_vs_native")
        status = "REF" if check is None else "PASS" if check["passed"] else "FAIL"
        passed &= status != "FAIL"
        if check is not None and not check["untouched_ok"]:
            status += " (cache holes modified)"
        print(
            f"{name:32} {str(result['programs'] or '-'):>6} {event:>10} {wall:>10} "
            f"{format_native_error(check, 'nope'):>21} {format_native_error(check, 'rope'):>21}  {status}",
            flush=True,
        )
    return passed


def run_case(tokens, layout, args, vector_cores, device):
    case = make_case(tokens, layout, args, device)
    slots_before = case.slots.clone()
    timings = {mode: {} for mode in args.modes}
    results = {}
    failures = []
    baseline_rows = None
    for name, programs, run in launchers(case, args, vector_cores):
        entry = {"programs": programs, "validation": {}, "modes": {}}
        results[name] = entry
        graphs = {}
        stage = "eager"
        try:
            case.storage.fill_(SENTINEL)
            run()
            torch.npu.synchronize()
            stats, written_rows = validate(case.storage, case.expected, case.written, baseline_rows)
            entry["validation"]["eager"] = stats
            if name == BASELINE:
                baseline_rows = written_rows
            for mode in args.modes:
                if mode != "eager":
                    stage = mode
                    graphs[mode] = capture(run, args.repeats if mode == "graph-batch" else 1)
            if graphs:
                case.projected.neg_()
                case.slots.copy_(case.replay_slots)
                try:
                    for mode, graph in graphs.items():
                        stage = mode
                        case.storage.fill_(SENTINEL)
                        graph.replay()
                        torch.npu.synchronize()
                        stats, _ = validate(case.storage, case.replay_expected, case.replay_written)
                        entry["validation"][mode] = stats
                finally:
                    case.projected.neg_()
                    case.slots.copy_(slots_before)
            for mode in args.modes:
                timings[mode][name] = run if mode == "eager" else graphs[mode].replay
        except Exception as exc:
            compile_failure = isinstance(exc, COMPILER_ERRORS)
            if not compile_failure and not isinstance(exc, ValidationError) and not args.continue_on_error:
                raise
            failure = {"candidate": name, "stage": stage, "type": type(exc).__name__, "message": str(exc)}
            compiler_process = exc if isinstance(exc, CalledProcessError) else exc.__cause__
            if compile_failure and isinstance(compiler_process, CalledProcessError) and compiler_process.stderr:
                stderr = compiler_process.stderr
                failure["compiler_stderr"] = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            failures.append(failure)
            lines = (failure.get("compiler_stderr") or str(exc)).strip().splitlines()
            summary = next(
                (line.strip() for line in lines if "ub overflow" in line.lower() or "cannot align" in line.lower()),
                lines[0].strip() if lines else type(exc).__name__,
            )
            status = "FAIL" if name == BASELINE or isinstance(exc, ValidationError) else "SKIP"
            print(f"  {status} {name} ({stage}, {type(exc).__name__}): {summary}", flush=True)
            if name == BASELINE:
                break

    if args.check_only or failures and failures[0]["candidate"] == BASELINE:
        return results, failures
    rng = random.Random(args.seed + tokens)
    for mode, calls_by_name in timings.items():
        for run in calls_by_name.values():
            for _ in range(args.warmups):
                run()
        torch.npu.synchronize()
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        calls = 1 if mode == "graph-batch" else args.repeats
        samples = {name: [] for name in calls_by_name}
        order = list(samples)
        for _ in range(args.samples):
            rng.shuffle(order)
            for name in order:
                samples[name].append(measure(calls_by_name[name], calls, args.repeats, start, end))
        for name, values in samples.items():
            mode_stats = {"samples": values}
            for metric in ("event_us", "wall_us", "submit_us"):
                measured = [sample[metric] for sample in values]
                paired = [
                    base[metric] / value
                    for base, value in zip(samples[BASELINE], measured, strict=True)
                    if value > 0 and base[metric] > 0
                ]
                mode_stats[metric] = {
                    "p50": quantile(measured, 0.5),
                    "p90": quantile(measured, 0.9),
                    "paired_speedup_p50": quantile(paired, 0.5) if paired else None,
                    "paired_speedup_p10": quantile(paired, 0.1) if paired else None,
                }
            results[name]["modes"][mode] = mode_stats
    return results, failures


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def compact_results(rows):
    fastest_by_variant = {}
    for name, timing in rows:
        fastest_by_variant.setdefault(name.partition("/")[0], (name, timing))
    baseline = fastest_by_variant.pop(BASELINE.partition("/")[0])
    control = fastest_by_variant.pop("masked512", None)
    finalists = list(fastest_by_variant.values())[:2]
    shown = [baseline, *([control] if control else []), *finalists]
    parts = []
    for name, timing in shown:
        event = timing["event_us"]
        label = "base" if name == BASELINE else name
        part = f"{label}={event['p50']:.2f}/{timing['wall_us']['p50']:.2f}us"
        ratio = event["paired_speedup_p50"]
        if name != BASELINE and ratio is not None:
            part += f" ({ratio:.2f}x/{event['paired_speedup_p10']:.2f}x)"
        parts.append(part)
    return " | ".join(parts)


@torch.inference_mode()
def main():
    args = args_from_cli()
    torch.set_num_threads(1)
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU visible: run on an Ascend server with CANN, torch-npu and Triton-Ascend")
    if args.quick or args.native_library:
        args.native_library = native_library_path(args.native_library)
        load_native_library(args.native_library)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    properties = triton.runtime.driver.active.utils.get_device_properties(args.device)
    vector_cores = args.vector_cores or properties.get("num_vectorcore", 0)
    if vector_cores <= 0:
        raise SystemExit("Triton did not report num_vectorcore; set --vector-cores to the device's vector core count")
    report = {
        "device": torch.npu.get_device_name(args.device),
        "vector_cores": vector_cores,
        "baseline_revision": BASELINE_REV,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: package_version(name) for name in ("torch", "torch-npu", "triton", "triton-ascend")},
        "args": vars(args),
        "cases": [],
    }
    print(f"{report['device']} | vector cores: {vector_cores} | BF16 512D, RoPE 64D | {report['versions']}", flush=True)
    if args.quick:
        print(
            f"Native reference: {args.native_library} | block size: {args.block_size} | atol={ATOL}, rtol={RTOL}",
            flush=True,
        )
        report["mode"] = "eager"
        passed = True
        for tokens in args.tokens:
            results = run_quick(tokens, args, vector_cores, device)
            report["cases"].append({"tokens": tokens, "layout": "paged", "candidates": results})
            passed &= print_quick(tokens, results)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if not passed:
            raise SystemExit("At least one Triton version failed compilation or cache validation")
        return
    invalid = False
    for tokens in args.tokens:
        for layout in args.layouts:
            print(f"T={tokens}, layout={layout}, block={args.block_size}", flush=True)
            entries, failures = run_case(tokens, layout, args, vector_cores, device)
            report["cases"].append({"tokens": tokens, "layout": layout, "candidates": entries, "failures": failures})
            invalid |= any(row["candidate"] == BASELINE or row["type"] == "ValidationError" for row in failures)
            for mode in args.modes:
                rows = [(name, entry["modes"][mode]) for name, entry in entries.items() if mode in entry["modes"]]
                if not rows:
                    continue
                rows.sort(key=lambda row: row[1]["event_us"]["p50"])
                if not args.verbose:
                    print(
                        f"RESULT T={tokens} {layout} {mode} {len(rows)}/{len(entries)} timed | {compact_results(rows)}",
                        flush=True,
                    )
                    continue
                print(
                    f"  {mode:11} {'candidate':32} {'grid':>6} {'event p50/p90 us':>20} "
                    f"{'wall p50 us':>12} {'paired event p50/p10':>23}"
                )
                shown = rows[:DISPLAY_TOP] + [row for row in rows[DISPLAY_TOP:] if row[0] == BASELINE]
                for name, timing in shown:
                    event = timing["event_us"]
                    ratio = event["paired_speedup_p50"]
                    speedup = f"{ratio:.3f}x/{event['paired_speedup_p10']:.3f}x" if ratio is not None else "n/a"
                    print(
                        f"  {'':11} {name:32} {str(entries[name]['programs']):>6} "
                        f"{event['p50']:9.3f}/{event['p90']:<9.3f} {timing['wall_us']['p50']:12.3f} {speedup:>23}"
                    )
                if len(shown) < len(rows):
                    print(f"  Showing {len(shown)} of {len(rows)} rows; use --json for all results.")
            if args.check_only or args.verbose or failures:
                print(f"  {len(entries) - len(failures)}/{len(entries)} passed cache validation", flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if invalid:
        raise SystemExit("Baseline failed or at least one candidate wrote incorrect cache contents; inspect the report")


if __name__ == "__main__":
    main()
