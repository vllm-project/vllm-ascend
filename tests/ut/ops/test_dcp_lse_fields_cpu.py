# SPDX-License-Identifier: Apache-2.0
"""Check actual encoder arithmetic by a separate FP64 reconstruction."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class Array(np.ndarray):
    def to(self, dtype, bitcast=False):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(Array)


def array(value):
    return np.asarray(value).view(Array)


def test_lse_fields_roundtrip_all_exponents_and_random_mantissas():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_encode_lse_fields"
    )
    function.decorator_list = []
    scope = {
        "tl": SimpleNamespace(
            int32=np.int32,
            float32=np.float32,
            where=lambda *args: array(np.where(*args)),
            maximum=lambda *args: array(np.maximum(*args)),
        )
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    rng = np.random.default_rng(55)
    random = rng.integers(0, 1 << 32, 1 << 18, dtype=np.uint32)
    exponents = np.arange(256, dtype=np.uint32)[:, None] << 23
    mantissas = np.array([0, 1, 0x7FFFFE, 0x7FFFFF], dtype=np.uint32)[None, :]
    boundary = (exponents | mantissas).reshape(-1)
    words = np.concatenate((random, boundary, boundary | np.uint32(1 << 31)))
    values = array(words.view(np.float32))
    code, hi, mid, lo = scope["_encode_lse_fields"](values)
    for field in (code, hi, mid, lo):
        assert np.all(np.abs(field) <= 255)
        np.testing.assert_array_equal(field, field.astype(np.int32))
    finite = np.isfinite(values)
    assert np.all(code[~finite] == 0)
    assert np.all(hi[~finite] == 0)
    assert np.all(mid[~finite] == 0)
    assert np.all(lo[~finite] == 0)
    significand = hi.astype(np.float64) * 65536 + mid * 256 + lo
    restored = np.sign(code) * np.ldexp(significand, (np.abs(code) - 151).astype(np.int32))
    expected = np.where(values == 0, np.float32(0), values)
    np.testing.assert_array_equal(restored[finite].astype(np.float32).view(np.uint32), expected[finite].view(np.uint32))
