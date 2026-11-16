# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the BLOCK_T heuristic of the FLA/KDA chunk-local cumsum.

``chunk_local_cumsum_scalar`` sizes its tile with
``BLOCK_T = max(next_power_of_2(2**18 // (H * chunk_size)), chunk_size)``. The scalar
kernel reshapes a tile into ``(N_CHUNKS, CHUNK_SIZE, H)`` with
``N_CHUNKS = BLOCK_T // CHUNK_SIZE``, so a heuristic value below ``chunk_size`` produces
``N_CHUNKS == 0`` and the reshape fails: with ``chunk_size=64`` every v-head count above
64 used to crash on the first prefill, regardless of the sequence length.

The module cannot be imported here (it pulls in ``vllm``/``triton``/device code), so the
launcher function is loaded from source and executed with a stub triton namespace. That
keeps the regression test CPU-only and fast while still exercising the real heuristic.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
CUMSUM_MODULES = (
    "vllm_ascend/ops/triton/fla/cumsum.py",
    "vllm_ascend/ops/triton/kda/cumsum.py",
)
# Smallest legal tile for the scalar kernel, expressed as the number of chunks it holds.
MIN_CHUNKS_PER_BLOCK = 1


def _reference_next_power_of_2(value):
    """Mirror ``triton.next_power_of_2`` bit for bit, including its non-positive input.

    Triton rounds by propagating the sign bit downwards, so 0 stays 0 instead of raising.
    """
    value -= 1
    for shift in (1, 2, 4, 8, 16, 32):
        value |= value >> shift
    return value + 1


def _legacy_block_t(num_heads, chunk_size):
    """Pre-fix heuristic, kept to prove that H <= 64 configurations are untouched."""
    return _reference_next_power_of_2((2**18) // (num_heads * chunk_size))


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "float32"


class _RecordingKernel:
    """Stands in for the triton kernel; records every launch keyword argument."""

    def __init__(self):
        self.grid = None
        self.launches = []

    def __getitem__(self, grid):
        self.grid = grid

        def launch(**kwargs):
            self.launches.append(kwargs)

        return launch


class _FakeTriton:
    @staticmethod
    def next_power_of_2(value):
        return _reference_next_power_of_2(value)

    @staticmethod
    def cdiv(numerator, denominator):
        return -(-numerator // denominator)


def _load_scalar_cumsum(relative_path):
    """Execute ``chunk_local_cumsum_scalar`` from the repository source in isolation."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "chunk_local_cumsum_scalar"
    )
    constants = {}
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        ):
            expression = compile(ast.Expression(node.value), str(relative_path), "eval")
            constants[node.targets[0].id] = eval(expression, {"__builtins__": {}})

    kernel = _RecordingKernel()
    prepared_chunk_sizes = []

    def prepare_chunk_indices(cu_seqlens, chunk_size):
        prepared_chunk_sizes.append(chunk_size)
        return []

    namespace = {
        # Only ``empty_like``, ``float`` and the ``Tensor`` annotations are touched by the
        # launcher; annotations are evaluated at def time, hence the extra placeholders.
        "torch": SimpleNamespace(
            empty_like=lambda g, dtype=None: _FakeTensor(g.shape),
            float=None,
            Tensor=object,
            dtype=object,
        ),
        "triton": _FakeTriton(),
        "prepare_chunk_indices": prepare_chunk_indices,
        "chunk_local_cumsum_scalar_kernel": kernel,
    }
    namespace.update(constants)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(relative_path), "exec"), namespace)

    launcher = namespace["chunk_local_cumsum_scalar"]
    return launcher, kernel, prepared_chunk_sizes, namespace


def _launch(launcher, kernel, num_heads, chunk_size, seq_len=128, cu_seqlens=None, head_first=False):
    if head_first:
        gate = _FakeTensor((1, num_heads, seq_len))
    else:
        gate = _FakeTensor((1, seq_len, num_heads))
    kernel.launches.clear()
    launcher(
        g=gate,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=None,
    )
    assert len(kernel.launches) == 1, f"expected exactly one kernel launch, got {kernel.launches}"
    return kernel.launches[0]


class TestChunkLocalCumsumBlockT(unittest.TestCase):
    def _load(self, relative_path):
        launcher, kernel, prepared_chunk_sizes, namespace = _load_scalar_cumsum(relative_path)
        self.addCleanup(kernel.launches.clear)
        return launcher, kernel, prepared_chunk_sizes, namespace

    def test_element_budget_constant_is_documented(self):
        for relative_path in CUMSUM_MODULES:
            with self.subTest(module=relative_path):
                _, _, _, namespace = self._load(relative_path)
                self.assertEqual(namespace["OPTIM_BLOCK_ELEMENT_BUDGET"], 2**18)

    def test_block_t_covers_at_least_one_chunk(self):
        # H > 64 with chunk_size 64 used to collapse BLOCK_T to 32/16/8/... and crash.
        cases = [
            (num_heads, chunk_size)
            for num_heads in (1, 2, 16, 32, 64, 65, 96, 128, 256, 512, 1024)
            for chunk_size in (16, 32, 64, 128)
        ]
        for relative_path in CUMSUM_MODULES:
            launcher, kernel, _, _ = self._load(relative_path)
            for num_heads, chunk_size in cases:
                with self.subTest(module=relative_path, num_heads=num_heads, chunk_size=chunk_size):
                    launch = _launch(launcher, kernel, num_heads, chunk_size)
                    block_t = launch["BLOCK_T"]
                    self.assertGreaterEqual(
                        block_t // chunk_size,
                        MIN_CHUNKS_PER_BLOCK,
                        f"BLOCK_T={block_t} yields zero chunks for chunk_size={chunk_size}",
                    )
                    self.assertEqual(block_t % chunk_size, 0)
                    self.assertEqual(launch["CHUNK_SIZE"], chunk_size)

    def test_block_t_is_exactly_chunk_size_when_heuristic_collapses(self):
        # H = 125 is the smallest v-head count whose raw heuristic value (32) is below the
        # GDN chunk_size of 64; before the clamp that meant N_CHUNKS == 0 and a crash.
        # The last two cases push H * chunk_size past the element budget, where the raw
        # division round-trips through a non-positive value (0 or 1).
        for relative_path in CUMSUM_MODULES:
            launcher, kernel, _, _ = self._load(relative_path)
            cases = ((125, 64), (128, 64), (256, 64), (128, 128), (2048, 64), (4096, 64), (8192, 64))
            for num_heads, chunk_size in cases:
                with self.subTest(module=relative_path, num_heads=num_heads, chunk_size=chunk_size):
                    launch = _launch(launcher, kernel, num_heads, chunk_size)
                    self.assertEqual(launch["BLOCK_T"], chunk_size)
                    self.assertLess(_legacy_block_t(num_heads, chunk_size), chunk_size)

    def test_element_budget_underflow_is_clamped(self):
        # H * chunk_size beyond the budget makes the raw division zero, and Triton's
        # next_power_of_2 keeps zero as zero, so the value passed to the kernel has to be
        # clamped rather than relying on the rounding helper.
        for relative_path in CUMSUM_MODULES:
            launcher, kernel, _, _ = self._load(relative_path)
            for num_heads, chunk_size in ((8192, 64), (16384, 32)):
                with self.subTest(module=relative_path, num_heads=num_heads, chunk_size=chunk_size):
                    self.assertGreater(num_heads * chunk_size, 2**18)
                    self.assertEqual(_reference_next_power_of_2(2**18 // (num_heads * chunk_size)), 0)
                    launch = _launch(launcher, kernel, num_heads, chunk_size)
                    self.assertEqual(launch["BLOCK_T"], chunk_size)

    def test_block_t_matches_legacy_heuristic_for_small_head_counts(self):
        # H <= 64 must keep the historical tile size: no grid/block change for existing models.
        for relative_path in CUMSUM_MODULES:
            launcher, kernel, _, _ = self._load(relative_path)
            for num_heads in (1, 4, 8, 16, 32, 64):
                for chunk_size in (16, 32, 64):
                    with self.subTest(module=relative_path, num_heads=num_heads, chunk_size=chunk_size):
                        launch = _launch(launcher, kernel, num_heads, chunk_size)
                        self.assertEqual(launch["BLOCK_T"], _legacy_block_t(num_heads, chunk_size))
                        self.assertGreaterEqual(launch["BLOCK_T"], chunk_size)

    def test_varlen_chunk_indices_use_the_clamped_block_size(self):
        # ``prepare_chunk_indices`` must be called with the same tile size as the kernel,
        # otherwise varlen sequences are cut into chunks the kernel cannot represent.
        for relative_path in CUMSUM_MODULES:
            for num_heads, chunk_size, expected_chunk_size in ((64, 64, 64), (128, 64, 64), (256, 64, 64)):
                launcher, kernel, prepared_chunk_sizes, _ = self._load(relative_path)
                with self.subTest(module=relative_path, num_heads=num_heads):
                    cu_seqlens = _FakeTensor((4,))
                    launch = _launch(
                        launcher,
                        kernel,
                        num_heads,
                        chunk_size,
                        seq_len=192,
                        cu_seqlens=cu_seqlens,
                        head_first=False,
                    )
                    self.assertEqual(prepared_chunk_sizes, [expected_chunk_size])
                    self.assertEqual(launch["BLOCK_T"], expected_chunk_size)

    def test_head_first_layout_uses_the_same_clamped_block_size(self):
        for relative_path in CUMSUM_MODULES:
            for num_heads in (64, 128):
                launcher, kernel, _, _ = self._load(relative_path)
                with self.subTest(module=relative_path, num_heads=num_heads):
                    launch = _launch(launcher, kernel, num_heads, 64, seq_len=192, head_first=True)
                    self.assertGreaterEqual(launch["BLOCK_T"], 64)
                    self.assertTrue(launch["HEAD_FIRST"])


if __name__ == "__main__":
    unittest.main()
