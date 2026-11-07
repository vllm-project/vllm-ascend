# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact EngramHashIds correctness against Triton and a scalar CPU reference.

Run with pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_engram_hash_ids.py.
Direct execution with python also starts pytest and accepts its arguments.
Requires the rebuilt custom operator package and Torch extension on A3 or A5.
No timing, model initialization, output files or benchmark imports are involved.
"""

from __future__ import annotations

import importlib
import itertools
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

SEED = 20261021
DEVICE = "npu:0"
TRITON_TILE = 32
ASCEND_TILE = 128
HEAD_TILE = 8
VOCABULARY = 31
OUTPUT_SENTINEL = -1234567
GUARD_ELEMENTS = 17
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MULTIPLIER_EDGES = (0, 1, -1, INT64_MIN, INT64_MAX, 1 << 32, -(1 << 40), 0x123456789ABCDEF)
DIVISOR_EDGES = (1, -1, 2, -2, 257, -257, INT32_MAX, 1 << 32, INT64_MIN, INT64_MAX)
OFFSET_EDGES = (0, -1, INT32_MIN, INT32_MAX, 1 << 31, 1 << 32, INT64_MIN, INT64_MAX)
DTYPE_CONFIGURATIONS = tuple(itertools.product((torch.int32, torch.int64), repeat=4))


@dataclass(frozen=True)
class Case:
    tokens: int = ASCEND_TILE + 1
    layers: int = 2
    ngram: int = 4
    heads: int = HEAD_TILE
    requests: int = 3
    depth: int = 3
    history: str = "mixed"
    dtypes: tuple = (torch.int32, torch.int32, torch.int64, torch.int32)
    layout: str = "contiguous"
    pattern: str = "mixed"
    pad: int = 2
    dead: int = -1
    block_size: int = 4
    blocks: int = 5
    table_rows: int = 2
    slots: int = 29
    capacity: int = 0
    position: str = "normal"
    arithmetic: bool = False
    seed: int = SEED
    repeats: int = 1
    ragged: bool = False
    divisor: int | None = None
    multiplier: int | None = None
    offset: int | None = None

    @property
    def use_cache(self):
        return self.history not in ("none", "window_only")


def signed(value, bits):
    return ((value + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1))


def cpu_reference(data, case):
    """Python integers model addressing, dtype wrapping and C-style remainder.

    Read only logical inputs/effective sizes. Do not call either implementation
    or use PyTorch remainder (which has different negative-value semantics).
    """
    input_ids, mapping, positions = (data[name] for name in ("input_ids", "token_map", "positions"))
    position_bits = 32 if case.dtypes[2] == torch.int32 else 64
    expected = []
    for token in range(case.tokens):
        request = data["req_ids"][token]
        chunk_index = min(data["query_start_loc"][request], case.tokens - 1)
        chunk_start = positions[chunk_index]
        blocked, sources = False, []
        for shift in range(case.ngram):
            lookback = signed(positions[token] - shift, 64)
            if lookback >= chunk_start:
                index = max(token - shift, 0)
                source = case.dead if data["dead_mask"][index] else mapping[input_ids[index]]
            else:
                previous = signed(chunk_start - 1, position_bits)
                column = signed(previous - lookback, 64)
                raw = data["lookback_token_ids"][request][column] if 0 <= column < case.depth else -1
                if raw >= 0:
                    source = case.dead if data["lookback_dead_mask"][request][column] else mapping[raw]
                elif case.use_cache:
                    position = min(max(lookback, 0), case.blocks * case.block_size - 1)
                    block = data["block_table"][min(request, case.table_rows - 1)][position // case.block_size]
                    slot = min(max(block * case.block_size + position % case.block_size, 0), case.slots - 1)
                    source = data["cache"][slot]
                else:
                    source = signed(case.pad, 32)
            blocked = blocked or lookback < 0 or source == case.dead
            sources.append(case.pad if blocked else source)
        for layer in range(case.layers):
            rolling = 0
            for shift, source in enumerate(sources):
                multiplier = data["multipliers"][layer * case.ngram + shift]
                rolling = signed(rolling ^ signed(source * multiplier, 64), 64)
                if shift == 0:
                    continue
                for head in range(case.heads):
                    index = (layer * (case.ngram - 1) + shift - 1) * case.heads + head
                    remainder = abs(rolling) % abs(data["primes"][index])
                    if rolling < 0:
                        remainder = -remainder
                    expected.append(signed(remainder + data["offsets"][index], 32))
    return expected


def make_data(case):
    rng = random.Random(case.seed)
    lengths = [
        case.tokens // case.requests + (request < case.tokens % case.requests) for request in range(case.requests)
    ]
    if case.ragged and case.requests >= 3 and case.tokens:
        lengths = [1] + [0] * (case.requests - 2) + [case.tokens - 1]
    starts: list[int] = [0]
    requests: list[int] = []
    positions: list[int] = []
    position_bits = 32 if case.dtypes[2] == torch.int32 else 64
    for request, length in enumerate(lengths):
        start = 12 + request
        if case.position == "zero":
            start = 0
        elif case.position == "negative":
            start = -2
        elif case.position == "block_boundary":
            start = case.block_size
        elif case.position == "upper_clamp":
            start = case.blocks * case.block_size + case.ngram
        elif case.position == "dtype_min":
            start = -(1 << (position_bits - 1))
        elif case.position == "dtype_max":
            start = (1 << (position_bits - 1)) - 1
        positions.extend(signed(start + index, position_bits) for index in range(length))
        requests.extend([request] * length)
        starts.append(starts[-1] + length)
    mapping = [rng.randrange(-1000, 1001) for _ in range(VOCABULARY)]
    mapping[0], mapping[1], mapping[-1] = case.dead, INT32_MIN, INT32_MAX
    if case.dtypes[1] == torch.int64 and case.arithmetic:
        mapping[2:4] = [INT64_MIN, INT64_MAX]
    ids = [index % VOCABULARY for index in range(case.tokens)]
    dead_mask = [
        case.pattern == "all_dead" or (case.pattern == "mixed" and index % 7 == 3) for index in range(case.tokens)
    ]
    if case.pattern == "clean":
        ids = [1 + index % (VOCABULARY - 1) for index in range(case.tokens)]
    window = [[rng.randrange(VOCABULARY) for _ in range(case.depth)] for _ in range(case.requests)]
    for row in window:
        for column in range(case.depth):
            if case.history in ("cache", "none") or (case.history == "mixed" and column % 2):
                row[column] = -1 if column % 2 else -7
    window_dead = [
        [case.pattern == "window_dead" or (case.pattern == "mixed" and column % 3 == 1) for column in range(case.depth)]
        for _ in range(case.requests)
    ]
    table = [[(row * case.blocks + col) % 7 for col in range(case.blocks)] for row in range(case.table_rows)]
    # Both physical slot clamps, shared blocks and a row-clamped request.
    for row in table:
        row[0], row[-1] = -1, case.slots + 1
    cache = [rng.randrange(-1000, 1001) for _ in range(case.slots)]
    if cache:
        cache[0], cache[-1] = INT32_MIN, INT32_MAX
        if case.pattern == "cache_dead":
            cache = [case.dead] * case.slots
    parameter_count = case.layers * (case.ngram - 1) * case.heads
    multipliers = [rng.randrange(1, 1 << 40) for _ in range(case.layers * case.ngram)]
    primes = [257 + index * 2 for index in range(parameter_count)]
    offsets = [index * 1009 for index in range(parameter_count)]
    if case.arithmetic:
        multipliers = [MULTIPLIER_EDGES[index % len(MULTIPLIER_EDGES)] for index in range(len(multipliers))]
        primes = [DIVISOR_EDGES[index % len(DIVISOR_EDGES)] for index in range(parameter_count)]
        offsets = [OFFSET_EDGES[index % len(OFFSET_EDGES)] for index in range(parameter_count)]
    if case.divisor is not None:
        primes = [case.divisor] * parameter_count
    if case.multiplier is not None:
        multipliers = [case.multiplier] * len(multipliers)
    if case.offset is not None:
        offsets = [case.offset] * parameter_count
    return dict(
        input_ids=ids,
        token_map=mapping,
        dead_mask=dead_mask,
        positions=positions,
        query_start_loc=starts,
        req_ids=requests,
        lookback_token_ids=window,
        lookback_dead_mask=window_dead,
        block_table=table,
        cache=cache,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
    )


def input_strides(name, shape, layout):
    vectors = ("input_ids", "dead_mask", "positions", "query_start_loc")
    if len(shape) == 1:
        stride = 3 if (layout == "strided" and name in vectors) or layout == name else 1
        if layout == "zero_stride" and name in vectors:
            stride = 0
        return (stride,)
    column = 3 if layout == "strided" or layout == name + "_col" else 1
    row = max(shape[1] * column + (5 if layout == "strided" or layout == name + "_row" else 0), 1)
    if layout == "zero_stride":
        column = 0
    if layout == name + "_zero_row":
        row = 0
    return row, column


def device_inputs(case):
    """Use real storage offsets, independent matrix strides and excess capacity."""
    data = make_data(case)
    input_dtype, map_dtype, position_dtype, lookback_dtype = case.dtypes
    dtypes = dict(
        input_ids=input_dtype,
        token_map=map_dtype,
        positions=position_dtype,
        lookback_token_ids=lookback_dtype,
        dead_mask=torch.bool,
        lookback_dead_mask=torch.bool,
        multipliers=torch.int64,
        primes=torch.int64,
        offsets=torch.int64,
    )
    matrix_shapes = dict(
        block_table=(case.table_rows, case.blocks),
        lookback_token_ids=(case.requests, case.depth),
        lookback_dead_mask=(case.requests, case.depth),
    )
    tensors = {}
    for name, values in data.items():
        cpu = torch.tensor(values, dtype=dtypes.get(name, torch.int32))
        if name in matrix_shapes:
            cpu = cpu.reshape(matrix_shapes[name])
        shape = tuple(cpu.shape)
        strides = input_strides(name, shape, case.layout)
        offset = GUARD_ELEMENTS if case.layout != "contiguous" else 0
        span = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1 if cpu.numel() else 0
        storage = torch.zeros(offset + span + case.capacity + GUARD_ELEMENTS, dtype=cpu.dtype)
        view = storage.as_strided(shape, strides, offset)
        # Explicit scalar assignment also supports overlapping zero-stride views.
        if cpu.ndim == 1:
            for index in range(shape[0]):
                view[index] = cpu[index]
        else:
            for row in range(shape[0]):
                for col in range(shape[1]):
                    view[row, col] = cpu[row, col]
        data[name] = view.tolist()
        exposed_shape = shape
        if case.capacity and name in (
            "input_ids",
            "dead_mask",
            "positions",
            "req_ids",
            "cache",
            "multipliers",
            "primes",
            "offsets",
        ):
            exposed_shape = (shape[0] + case.capacity,)
            # Strided effective inputs need enough backing storage for exposed capacity.
            required = offset + max(exposed_shape[0] - 1, 0) * strides[0] + 1 + GUARD_ELEMENTS
            if required > storage.numel():
                storage = torch.cat((storage, torch.zeros(required - storage.numel(), dtype=storage.dtype)))
        tensors[name] = storage.to(DEVICE).as_strided(exposed_shape, strides, offset)
    return data, tensors


@pytest.fixture(scope="module")
def implementations():
    if not torch.npu.is_available():
        pytest.skip("EngramHashIds requires an Ascend NPU")
    utils = importlib.import_module("vllm_ascend.utils")
    if utils.is_950():
        utils.bootstrap_custom_op_env()
        try:
            importlib.import_module("vllm_ascend.vllm_ascend_C")
        except ImportError as exc:
            if "libcust_opapi.so" not in str(exc):
                raise
            utils.bootstrap_custom_op_env(include_vendor_lib=True)
            importlib.import_module("vllm_ascend.vllm_ascend_C")
        importlib.import_module("vllm_ascend.meta_registration")
    elif not utils.enable_custom_op():
        pytest.fail("Custom-op loading failed; rebuild the operator package and extension")
    kernel = importlib.import_module("vllm_ascend.ops.triton.engram_hash")
    return torch.ops._C_ascend.engram_hash_ids, kernel._hash_ids_kernel


def run_case(case, implementations):
    ascendc, triton = implementations
    data, tensors = device_inputs(case)
    expected = torch.tensor(cpu_reference(data, case), dtype=torch.int32)
    output_elements = case.tokens * case.layers * (case.ngram - 1) * case.heads
    assert expected.numel() == output_elements
    storage_size = GUARD_ELEMENTS + output_elements + case.capacity + GUARD_ELEMENTS
    storages = [torch.full((storage_size,), OUTPUT_SENTINEL, dtype=torch.int32, device=DEVICE) for _ in range(2)]
    outputs = [storage[GUARD_ELEMENTS : GUARD_ELEMENTS + output_elements + case.capacity] for storage in storages]
    snapshots = {name: tensor.cpu().clone() for name, tensor in tensors.items()}
    t = tensors
    arguments = [t[name] for name in ("input_ids", "token_map", "dead_mask", "positions")]
    arguments += [t["block_table"] if case.use_cache else None]
    arguments += [t[name] for name in ("query_start_loc", "req_ids", "multipliers", "primes", "offsets")]
    arguments += [t["cache"] if case.use_cache else None, t["lookback_token_ids"], t["lookback_dead_mask"]]
    scalars = dict(
        input_stride=t["input_ids"].stride(0),
        mask_stride=t["dead_mask"].stride(0),
        position_stride=t["positions"].stride(0),
        table_stride=t["block_table"].stride(0),
        table_col_stride=t["block_table"].stride(1),
        query_stride=t["query_start_loc"].stride(0),
        num_table_rows=case.table_rows,
        max_blocks=case.blocks,
        cache_block_size=case.block_size,
        MAX_NGRAM=case.ngram,
        num_heads=case.heads,
        dead_id=case.dead,
        lookback_depth=case.depth,
        lookback_row_stride=t["lookback_token_ids"].stride(0),
        lookback_col_stride=t["lookback_token_ids"].stride(1),
        lookback_mask_row_stride=t["lookback_dead_mask"].stride(0),
        lookback_mask_col_stride=t["lookback_dead_mask"].stride(1),
    )

    def check_result(index, name):
        torch.npu.synchronize()
        result = outputs[index][:output_elements].cpu()
        torch.testing.assert_close(result, expected, rtol=0, atol=0, msg=f"{name} != CPU: {case}")
        storage = storages[index]
        assert torch.all(storage[:GUARD_ELEMENTS] == OUTPUT_SENTINEL).item(), f"{name}: output prefix overwritten"
        assert torch.all(storage[GUARD_ELEMENTS + output_elements :] == OUTPUT_SENTINEL).item(), (
            f"{name}: output tail overwritten"
        )
        for input_name, tensor in tensors.items():
            torch.testing.assert_close(
                tensor.cpu(), snapshots[input_name], rtol=0, atol=0, msg=f"{name}: read-only {input_name} changed"
            )
        print(f"{name} vs CPU: PASS; output guards and read-only inputs: PASS", flush=True)
        return result

    for _ in range(case.repeats):
        for storage in storages:
            storage.fill_(OUTPUT_SENTINEL)
        print("Running Ascend C", flush=True)
        ascendc(*arguments, outputs[0], case.tokens, case.slots, case.pad, **scalars, num_layers=case.layers)
        ascendc_result = check_result(0, "Ascend C")
        print("Running Triton" if case.tokens else "Triton: zero-token no-op", flush=True)
        # A zero-token launch has no programs; compare the untouched buffers.
        if case.tokens:
            triton[(math.ceil(case.tokens / TRITON_TILE), case.layers)](
                *arguments,
                outputs[1],
                case.tokens,
                case.slots,
                case.pad,
                **scalars,
                BLOCK_T=TRITON_TILE,
                BLOCK_H=1 << (case.heads - 1).bit_length(),
                num_warps=4,
            )
        triton_result = check_result(1, "Triton")
        torch.testing.assert_close(ascendc_result, triton_result, rtol=0, atol=0, msg=f"Ascend C != Triton: {case}")


def correctness_cases():
    """Separate axes keep failures identifiable without an impractical Cartesian product."""

    def entry(name, **changes):
        return pytest.param(replace(Case(), **changes), id=name)

    cases = []
    for tokens, history in itertools.product(
        (0, 1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257), ("none", "window", "cache", "mixed")
    ):
        cases.append(entry(f"tokens-{tokens}-{history}", tokens=tokens, history=history))
    for layers, ngram, heads in itertools.product((1, 2, 5), (1, 2, 4, 7), (1, 3, 7, 8, 9, 17, 33)):
        cases.append(entry(f"shape-L{layers}-N{ngram}-H{heads}", layers=layers, ngram=ngram, heads=heads))
    # Exercise full and partial SIMT history groups, including persistent blocked state.
    for ngram, history in itertools.product((8, 9, 16, 17), ("none", "window_only", "cache", "mixed")):
        cases.append(
            entry(
                f"history-groups-N{ngram}-{history}",
                tokens=33,
                ngram=ngram,
                heads=1,
                depth=ngram,
                history=history,
                pattern="window_dead",
                layout="strided",
                arithmetic=True,
                repeats=2,
            )
        )
    for index, dtypes in enumerate(DTYPE_CONFIGURATIONS):
        for history, layout in itertools.product(("none", "window", "cache", "mixed"), ("contiguous", "strided")):
            cases.append(
                entry(
                    f"dtype-{index}-{history}-{layout}", dtypes=dtypes, history=history, layout=layout, arithmetic=True
                )
            )
    for depth, history, position in itertools.product(
        (1, 2, 3, 6, 8), ("none", "window", "cache", "mixed"), ("zero", "block_boundary", "upper_clamp")
    ):
        cases.append(entry(f"history-D{depth}-{history}-{position}", depth=depth, history=history, position=position))
    for pattern, history, pad in itertools.product(
        ("clean", "mixed", "all_dead", "window_dead", "cache_dead"),
        ("none", "window", "cache", "mixed"),
        (0, -1, 1 << 32, INT64_MIN, INT64_MAX),
    ):
        cases.append(
            entry(f"dead-{pattern}-{history}-pad{pad}", pattern=pattern, history=history, pad=pad, arithmetic=True)
        )
    for position, dtype, history in itertools.product(
        ("negative", "dtype_min", "dtype_max"), (torch.int32, torch.int64), ("none", "cache", "mixed")
    ):
        cases.append(
            entry(
                f"position-{position}-{dtype}-{history}",
                position=position,
                history=history,
                dtypes=(torch.int64, torch.int64, dtype, torch.int64),
                arithmetic=True,
            )
        )
    for layout, tokens, heads in itertools.product(("strided", "zero_stride"), (1, 33, 129), (1, 8, 9)):
        cases.append(
            entry(
                f"layout-{layout}-T{tokens}-H{heads}",
                layout=layout,
                tokens=tokens,
                heads=heads,
                capacity=ASCEND_TILE,
                repeats=2,
            )
        )
    for block_size, slots, rows in itertools.product((1, 3, 4, 127, 128, 129), (1, 17, 129), (1, 2, 4)):
        cases.append(
            entry(
                f"cache-block{block_size}-slots{slots}-rows{rows}",
                block_size=block_size,
                slots=slots,
                table_rows=rows,
                history="cache",
                position="upper_clamp",
            )
        )
    for tokens, requests in itertools.product((1, 33, 129), (1, 2, 7, 33)):
        cases.append(entry(f"requests-T{tokens}-R{requests}", tokens=tokens, requests=requests, depth=7))
    for dead in (0, INT32_MIN, INT32_MAX):
        cases.append(entry(f"dead-id-{dead}", dead=dead, arithmetic=True))
    for tokens in (8192, 16384):
        cases.append(entry(f"large-{tokens}", tokens=tokens, pattern="clean", history="mixed", capacity=ASCEND_TILE))
    independent_layouts = (
        "input_ids",
        "dead_mask",
        "positions",
        "query_start_loc",
        "block_table_row",
        "block_table_col",
        "block_table_zero_row",
        "lookback_token_ids_row",
        "lookback_token_ids_col",
        "lookback_token_ids_zero_row",
        "lookback_dead_mask_row",
        "lookback_dead_mask_col",
        "lookback_dead_mask_zero_row",
    )
    for layout, history in itertools.product(independent_layouts, ("none", "window", "cache", "mixed")):
        cases.append(entry(f"independent-{layout}-{history}", layout=layout, history=history, arithmetic=True))
    for divisor, multiplier in itertools.product(DIVISOR_EDGES, (INT64_MIN, -1, INT64_MAX)):
        cases.append(
            entry(
                f"remainder-p{divisor}-m{multiplier}",
                divisor=divisor,
                multiplier=multiplier,
                arithmetic=True,
                dtypes=(torch.int64,) * 4,
            )
        )
    for offset in OFFSET_EDGES:
        cases.append(entry(f"offset-{offset}", offset=offset, arithmetic=True))
    for tokens, history in itertools.product((1, 33, 129), ("none", "window", "cache", "mixed")):
        cases.append(entry(f"ragged-{tokens}-{history}", tokens=tokens, requests=7, ragged=True, history=history))
    for index, dtypes in enumerate(DTYPE_CONFIGURATIONS):
        for depth in (1, 7):
            cases.append(
                entry(
                    f"no-cache-window-{index}-D{depth}",
                    dtypes=dtypes,
                    depth=depth,
                    history="window_only",
                    arithmetic=True,
                    capacity=ASCEND_TILE,
                )
            )
    for heads in (63, 64, 65, 127, 128, 129):
        cases.append(entry(f"large-heads-{heads}", heads=heads, tokens=33, arithmetic=True))
    rng = random.Random(SEED)
    for index in range(64):
        cases.append(
            entry(
                f"random-{index}",
                tokens=rng.randrange(1, 300),
                layers=rng.randrange(1, 5),
                ngram=rng.randrange(2, 9),
                heads=rng.choice((1, 3, 8, 9, 17)),
                depth=rng.randrange(1, 9),
                requests=rng.randrange(1, 10),
                history=rng.choice(("none", "window", "cache", "mixed")),
                dtypes=rng.choice(DTYPE_CONFIGURATIONS),
                layout=rng.choice(independent_layouts),
                arithmetic=True,
                capacity=ASCEND_TILE,
                seed=SEED + index,
            )
        )
    return cases


class TestEngramHashIds:
    @pytest.mark.parametrize("case", correctness_cases())
    def test_exact_correctness(self, case, implementations):
        run_case(case, implementations)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-sv", *sys.argv[1:]]))
