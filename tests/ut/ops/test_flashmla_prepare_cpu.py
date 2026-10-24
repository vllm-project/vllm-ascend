# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interpret production D02 address calculations on CPU, without importing Triton.

The production wrappers and kernel body execute unchanged, apart from removing
the JIT decorator. NumPy implements the small pointer/load/store surface used by
this copy kernel. This checks layout, masks and wrapper arguments; it does not
compile Triton, validate an NPU backend, or establish performance.
"""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/dcp/flashmla_prepare.py"
NOPE_DIM = 512
PE_DIM = 64
PACKED_DIM = NOPE_DIM + PE_DIM
LOCAL_HEADS = 12
DCP_SIZE = 8
CACHE_BLOCK_SIZE = 128
SENTINEL = -32256


class Array(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Array)


class Scalar(int):
    def __floordiv__(self, other):
        return Scalar(int(self) // int(other))

    def __mod__(self, other):
        return Scalar(int(self) % int(other))

    def to(self, dtype):
        return Scalar(dtype(self))


class Tensor:
    """Logical tensor view and its entire sentinel-padded storage."""

    def __init__(self, array, storage, writes):
        self.array, self.storage, self.writes = array, storage, writes
        self.shape, self.ndim, self.dtype = array.shape, array.ndim, array.dtype
        self.device = "cpu"
        self.offset = (array.ctypes.data - storage.ctypes.data) // array.itemsize

    def stride(self, dimension=None):
        strides = tuple(value // self.array.itemsize for value in self.array.strides)
        return strides if dimension is None else strides[dimension]

    def __getitem__(self, item):
        return Tensor(self.array[item], self.storage, self.writes)


def strided(shape, strides, *, offset=19, dtype=np.float16):
    size = offset + sum(max(length - 1, 0) * stride for length, stride in zip(shape, strides)) + 23
    storage = np.full(size, SENTINEL, dtype=dtype)
    array: npt.NDArray[Any] = np.ndarray(
        shape,
        dtype=dtype,
        buffer=storage,
        offset=offset * storage.itemsize,
        strides=tuple(stride * storage.itemsize for stride in strides),
    )
    return Tensor(array, storage, np.zeros(size, dtype=np.int32))


class Pointer:
    def __init__(self, tensor, offsets=None):
        self.tensor = tensor
        self.offsets = np.asarray(tensor.offset if offsets is None else offsets, dtype=np.int64)

    def __add__(self, offsets):
        return Pointer(self.tensor, self.offsets + offsets)


class Language:
    constexpr = object
    int64 = np.int64

    def __init__(self):
        self.program = 0
        self.programs = 1

    def program_id(self, axis):
        assert axis == 0
        return self.program

    def num_programs(self, axis):
        assert axis == 0
        return self.programs

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop).view(Array)

    @staticmethod
    def where(condition, left, right):
        return np.asarray(np.where(condition, left, right)).view(Array)

    @staticmethod
    def addresses(pointer, mask):
        offsets = np.asarray(pointer.offsets, dtype=np.int64)
        live = np.broadcast_to(np.asarray(mask, dtype=bool), offsets.shape)
        selected = offsets[live]
        assert np.all((selected >= 0) & (selected < pointer.tensor.storage.size)), "live out-of-bounds address"
        return offsets, live

    @classmethod
    def load(cls, pointer, mask=True, other=0):
        offsets, live = cls.addresses(pointer, mask)
        result = np.full(offsets.shape, other, dtype=pointer.tensor.dtype)
        result[live] = pointer.tensor.storage[offsets[live]]
        return result.view(Array)

    @classmethod
    def store(cls, pointer, values, mask=True):
        offsets, live = cls.addresses(pointer, mask)
        values = np.broadcast_to(np.asarray(values), offsets.shape)
        pointer.tensor.storage[offsets[live]] = values[live]
        np.add.at(pointer.tensor.writes, offsets[live], 1)


class Launcher:
    def __init__(self, kernel, language):
        self.kernel, self.language = kernel, language
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1
            self.language.programs = grid[0]
            pointers = tuple(Pointer(value) if isinstance(value, Tensor) else value for value in args)
            for program in builtins.range(grid[0]):
                self.language.program = program
                self.kernel(*pointers, **kwargs)

        return launch


@pytest.fixture
def production():
    language = Language()
    scope = {
        "tl": language,
        "torch": SimpleNamespace(
            Tensor=Tensor,
            float16=np.dtype(np.float16),
            bfloat16=object(),
            int32=np.dtype(np.int32),
            int64=np.dtype(np.int64),
        ),
        "triton": SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
        "init_device_properties_triton": lambda: None,
        "get_vectorcore_num": lambda: 7,
        "range": lambda *args: (Scalar(value) for value in builtins.range(*args)),
    }
    functions = [
        node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)
    ]
    for node in functions:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), scope)
    launcher = Launcher(scope["_prepare_flashmla_kernel"], language)
    scope["_prepare_flashmla_kernel"] = launcher
    return SimpleNamespace(
        query=scope["prepare_flashmla_query"],
        current=scope["prepare_flashmla_current"],
        launcher=launcher,
    )


def components(tokens, heads, dimension_stride=1):
    head_stride = PACKED_DIM * dimension_stride
    packed = strided((tokens, heads, PACKED_DIM), (heads * head_stride + 13, head_stride, dimension_stride))
    token = np.arange(tokens)[:, None, None]
    head = np.arange(heads)[None, :, None]
    dimension = np.arange(PACKED_DIM)[None, None, :]
    packed.array[...] = ((token * 19 + head * 7 + dimension * 3) % 509 - 254) / 4
    # A nonzero raw component distinct from the latent component catches loss
    # of the 64 NoPE dimensions and use of the wrong split storage offset.
    packed.array[..., NOPE_DIM:] += 256
    return packed, packed[..., :NOPE_DIM], packed[..., NOPE_DIM:]


def query_output(tokens, heads):
    head_stride = PACKED_DIM * 2 + 11
    return strided((tokens, heads, PACKED_DIM), (heads * head_stride + 17, head_stride, 2))


def check_whole_query_storage(output, reference):
    expected = np.full_like(output.storage, SENTINEL)
    expected_view: npt.NDArray[Any] = np.ndarray(
        output.shape,
        dtype=output.dtype,
        buffer=expected,
        offset=output.offset * expected.itemsize,
        strides=output.array.strides,
    )
    expected_view[...] = reference
    np.testing.assert_array_equal(output.storage, expected)
    assert np.count_nonzero(output.writes) == reference.size
    assert np.all(output.writes[output.writes > 0] == 1)


@pytest.mark.parametrize("tokens", [96, 104])
@pytest.mark.parametrize("dimension_stride", [1, 2])
def test_history_query_exact_strided_layout(production, tokens, dimension_stride):
    packed, q_nope, q_pe = components(tokens, LOCAL_HEADS * DCP_SIZE, dimension_stride)
    before = packed.storage.copy()
    output = query_output(tokens, LOCAL_HEADS * DCP_SIZE)
    reference = np.concatenate((q_nope.array, q_pe.array), axis=-1)
    production.query(q_nope, q_pe, output)
    assert production.launcher.launches == 1
    check_whole_query_storage(output, reference)
    np.testing.assert_array_equal(packed.storage, before)


@pytest.mark.parametrize("tokens,requests", [(96, 24), (104, 25)])
@pytest.mark.parametrize("slot_dtype", [np.int32, np.int64])
def test_current_query_and_kv_exact_with_padding(production, tokens, requests, slot_dtype):
    packed, _, _ = components(tokens, LOCAL_HEADS * DCP_SIZE)
    # This rank's retained local heads correspond to a nonzero gathered shard.
    rank = 3
    local = packed[:, rank * LOCAL_HEADS : (rank + 1) * LOCAL_HEADS]
    q_nope, q_pe = local[..., :NOPE_DIM], local[..., NOPE_DIM:]
    source_before = packed.storage.copy()
    kv, k_nope, k_pe = components(tokens, 1, dimension_stride=2)
    kv_before = kv.storage.copy()
    slots = strided((tokens,), (2,), dtype=slot_dtype)
    slots.array.fill(-1)
    # Four live tokens per request; T=104 has four trailing padding tokens.
    for request in range(requests):
        slots.array[request * 4 : request * 4 + 4] = ((request + 7) % requests) * CACHE_BLOCK_SIZE + np.arange(4)
    slots_before = slots.storage.copy()
    token_stride = PACKED_DIM * 2 + 9
    page_stride = CACHE_BLOCK_SIZE * token_stride + 37
    cache = strided(
        (requests, CACHE_BLOCK_SIZE, 1, PACKED_DIM),
        (page_stride, token_stride, token_stride, 2),
    )
    expected = cache.storage.copy()
    expected_cache: npt.NDArray[Any] = np.ndarray(
        cache.shape,
        dtype=cache.dtype,
        buffer=expected,
        offset=cache.offset * expected.itemsize,
        strides=cache.array.strides,
    )
    for token, slot in enumerate(slots.array):
        if slot >= 0:
            expected_cache[slot // CACHE_BLOCK_SIZE, slot % CACHE_BLOCK_SIZE, 0] = kv.array[token, 0]
    output = query_output(tokens, LOCAL_HEADS)
    production.current(q_nope, q_pe, k_nope, k_pe, output, cache, slots)
    assert production.launcher.launches == 1
    check_whole_query_storage(output, np.concatenate((q_nope.array, q_pe.array), axis=-1))
    np.testing.assert_array_equal(cache.storage, expected)
    assert np.count_nonzero(cache.writes) == requests * 4 * PACKED_DIM
    assert np.all(cache.writes[cache.writes > 0] == 1), "each KV element must have exactly one head owner"
    np.testing.assert_array_equal(packed.storage, source_before)
    np.testing.assert_array_equal(kv.storage, kv_before)
    np.testing.assert_array_equal(slots.storage, slots_before)


def test_invalid_slots_skip_cache_but_still_copy_query(production):
    _, q_nope, q_pe = components(4, LOCAL_HEADS)
    kv, k_nope, k_pe = components(4, 1)
    cache = strided((1, CACHE_BLOCK_SIZE, 1, PACKED_DIM), (CACHE_BLOCK_SIZE * PACKED_DIM, PACKED_DIM, PACKED_DIM, 1))
    slots = strided((4,), (1,), dtype=np.int64)
    slots.array[:] = [0, -1, CACHE_BLOCK_SIZE, np.iinfo(np.int64).max]
    output = query_output(4, LOCAL_HEADS)
    production.current(q_nope, q_pe, k_nope, k_pe, output, cache, slots)
    np.testing.assert_array_equal(output.array, np.concatenate((q_nope.array, q_pe.array), axis=-1))
    np.testing.assert_array_equal(cache.array[0, 0, 0], kv.array[0, 0])
    assert np.count_nonzero(cache.writes) == PACKED_DIM
    assert np.all(cache.writes[cache.writes > 0] == 1)


def test_empty_inputs_do_not_launch(production):
    _, q_nope, q_pe = components(0, LOCAL_HEADS)
    _, k_nope, k_pe = components(0, 1)
    output = query_output(0, LOCAL_HEADS)
    cache = strided((1, CACHE_BLOCK_SIZE, 1, PACKED_DIM), (CACHE_BLOCK_SIZE * PACKED_DIM, PACKED_DIM, PACKED_DIM, 1))
    slots = strided((0,), (1,), dtype=np.int64)
    before = cache.storage.copy()
    production.query(q_nope, q_pe, output)
    production.current(q_nope, q_pe, k_nope, k_pe, output, cache, slots)
    assert production.launcher.launches == 0
    np.testing.assert_array_equal(cache.storage, before)
