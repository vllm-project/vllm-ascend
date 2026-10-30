# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the preparation kernel body; NPU compilation is tested separately."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.models.qwen4_exp.gbsa_reference import reference_prepare


class _Value(np.ndarray):
    def to(self, dtype, bitcast=False):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(_Value)


def _value(value):
    return np.asarray(value).view(_Value)


class _Pointer:
    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, offset

    def __add__(self, offset):
        return _Pointer(self.storage, self.offset + offset)


class _Language:
    int32, int64, float32 = np.int32, np.int64, np.float32
    constexpr = object
    static_range = staticmethod(range)

    def program_id(self, axis):
        return _value(self.row)

    @staticmethod
    def arange(start, end):
        return _value(np.arange(start, end, dtype=np.int32))

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offset, mask)
        result = np.full(offsets.shape, other, dtype=pointer.storage.dtype)
        result[mask] = pointer.storage[offsets[mask]]
        return _value(result)

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, values, mask = np.broadcast_arrays(pointer.offset, value, mask)
        pointer.storage[offsets[mask]] = values[mask]

    @staticmethod
    def sum(value, axis):
        return _value(np.sum(value, axis=axis, dtype=value.dtype))

    @staticmethod
    def cumsum(value, axis):
        return _value(np.cumsum(value, axis=axis, dtype=value.dtype))

    @staticmethod
    def minimum(a, b):
        return _value(np.minimum(a, b))

    @staticmethod
    def maximum(a, b):
        return _value(np.maximum(a, b))

    @staticmethod
    def where(condition, a, b):
        return _value(np.where(condition, a, b))


class _Kernel:
    def __init__(self, function, language):
        self.function, self.language = function, language

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop("num_warps")
            pointers = []
            for argument in args:
                if isinstance(argument, torch.Tensor):
                    storage = argument.as_strided(
                        (argument.untyped_storage().nbytes() // argument.element_size(),), (1,), storage_offset=0
                    ).numpy()
                    argument = _Pointer(storage, argument.storage_offset())
                pointers.append(argument)
            for row in range(grid[0]):
                self.language.row = row
                self.function(*pointers, **kwargs)

        return launch


def load_cpu_preparation():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    language = _Language()
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    namespace = {
        "torch": torch,
        "tl": language,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    exec(compile(module, str(source), "exec"), namespace)
    namespace["_prepare_qsa_gbsa_rows_kernel"] = _Kernel(namespace["_prepare_qsa_gbsa_rows_kernel"], language)
    return namespace["prepare_qsa_gbsa_inputs"]


def make_inputs(groups, tail, pattern, dtype, rows=7):
    generator = torch.Generator().manual_seed(803)
    compact = torch.full((rows, 2 * (groups + 3)), -1, dtype=dtype)[:, ::2]
    selections = torch.randperm(groups, generator=generator).to(dtype)
    compact[:, :groups] = selections
    if pattern == "holes":
        compact[:, 1:groups:3] = -1
        compact[:, 2:groups:5] = groups + 1
    elif pattern == "prefix":
        compact[:, groups // 2 : groups] = -1
    elif pattern == "invalid":
        compact[:, :groups] = -1
    lengths = torch.tensor([groups * 4 + tail, 0, 3, 17], dtype=dtype).repeat_interleave(2)[::2]
    requests = torch.tensor([0, 1, 2, 3, -1, 0, 0], dtype=dtype)[:rows].repeat_interleave(2)[::2]
    positions = (lengths[requests.long()] - 1).repeat_interleave(2)[::2]
    table = torch.randint(0, 100, (4, 86), dtype=dtype, generator=generator)[:, ::2]
    return compact, positions, lengths, requests, table


@pytest.mark.parametrize("groups", [1, 2, 31, 128, 511, 512])
@pytest.mark.parametrize("tail", range(4))
@pytest.mark.parametrize("pattern", ["full", "prefix", "holes", "invalid"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("heads", [1, 2])
def test_kernel_integer_semantics(groups, tail, pattern, dtype, heads):
    args = make_inputs(groups, tail, pattern, dtype)
    expected = reference_prepare(*args, heads)
    actual = load_cpu_preparation()(*args, heads)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_empty_preparation():
    args = make_inputs(512, 1, "holes", torch.int32, rows=0)
    for result, reference in zip(load_cpu_preparation()(*args, 2), reference_prepare(*args, 2), strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pack_wide_positions_and_strided_buffers(dtype):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/qwen4_exp/qsa.py"
    module = ast.parse(source.read_text())
    module.body = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pack_qsa_group_indices_kernel"
    ]
    module.body[0].decorator_list = []
    language = _Language()
    namespace = {"tl": language}
    exec(compile(module, str(source), "exec"), namespace)
    kernel = _Kernel(namespace["_pack_qsa_group_indices_kernel"], language)
    positions = torch.tensor([-2, -1, 0, 1, 2, 3, 4, 7, 2047, 4095, 16383, 2147483646], dtype=dtype)
    rows = positions.numel()
    indices = torch.arange(rows * 1024, dtype=torch.int32).reshape(rows, 1024)[:, ::2]
    lengths = torch.tensor([0, 4096, 32768], dtype=dtype)
    for request_id in [-1, 0, 1, 2, 3]:
        requests = torch.full((rows,), request_id, dtype=torch.int32)
        storage = torch.full((rows, 2051), -777, dtype=torch.int32)
        output = storage[:, :515]
        kernel[(rows,)](
            indices,
            positions,
            lengths,
            requests,
            output,
            *indices.stride(),
            *output.stride(),
            rows,
            3,
            BLOCK_TOPK=512,
            COMPRESS_RATIO=4,
            OUTPUT_WIDTH=515,
            num_warps=1,
        )
        expected = torch.full_like(storage, -777)
        for row, position in enumerate(positions.tolist()):
            length = int(lengths[request_id]) if 0 <= request_id < 3 else 0
            complete = max(0, min((position + 1) // 4, length // 4, 512))
            expected[row, :515] = -1
            expected[row, :complete] = indices[row, :complete]
            start = ((position + 1) // 4) * 4
            for offset in range(3):
                if offset < (position + 1) - start and start + offset < length:
                    expected[row, 512 + offset] = start + offset
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)
