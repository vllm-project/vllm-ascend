# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host-side unit tests for the MRV2 penalties launcher.

These run on CPU. ``_penalties_kernel`` is replaced by a fake that records the
grid and arguments instead of launching, so what is under test is everything
``apply_penalties`` decides before the launch -- the grid, the derived tile
split, the empty-batch guard -- plus the partition arithmetic the kernel
mirrors and the specialization policy of the kernel signature.

Device numerics (bit-exactness, dtypes, vocabulary tails, speculative decoding)
are covered by
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py``,
which needs an NPU and so cannot live here.
"""

import ast
import inspect
from collections.abc import Callable

import pytest
import torch

from vllm_ascend.worker.v2.sample import penalties
from vllm_ascend.worker.v2.sample.penalties import VOCAB_BLOCK_SIZE, apply_penalties

# Positional arguments the launcher passes, after the pointer/stride block.
_VOCAB_SIZE_ARG = -3
_TOTAL_TILES_ARG = -2
_TILES_PER_WORKER_ARG = -1
_NUM_POSITIONAL_ARGS = 15


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _FakeKernel:
    """Records each launch instead of running it."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[object, ...], dict[str, object]]] = []

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        def launch(*args: object, **kwargs: object) -> None:
            self.calls.append((grid, args, kwargs))

        return launch

    @property
    def grid(self) -> tuple[int, ...] | None:
        return self.calls[-1][0] if self.calls else None

    @property
    def args(self) -> tuple[object, ...] | None:
        return self.calls[-1][1] if self.calls else None

    @property
    def kwargs(self) -> dict[str, object] | None:
        return self.calls[-1][2] if self.calls else None


def _inputs(num_tokens: int, vocab_size: int) -> tuple[torch.Tensor, ...]:
    """CPU tensors shaped the way ``apply_penalties`` expects them."""
    rows = max(num_tokens, 1)
    return (
        torch.zeros(num_tokens, vocab_size),
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.zeros(rows, 8, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.ones(rows),
        torch.zeros(rows),
        torch.zeros(rows),
        torch.zeros(rows, _cdiv(vocab_size, 32), dtype=torch.int32),
        torch.zeros(rows, vocab_size, dtype=torch.int32),
    )


@pytest.fixture
def fake_kernel(monkeypatch: pytest.MonkeyPatch) -> _FakeKernel:
    fake = _FakeKernel()
    monkeypatch.setattr(penalties, "_penalties_kernel", fake)
    return fake


def _set_cores(monkeypatch: pytest.MonkeyPatch, num_cores: int) -> None:
    """get_vectorcore_num() asserts unless device properties were initialised,
    which does not happen on CPU, so pin it the way the other worker UTs do."""
    monkeypatch.setattr(penalties, "get_vectorcore_num", lambda: num_cores)


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_cores", [1, 48, 64])
@pytest.mark.parametrize("num_tokens", [1, 256])
def test_grid_is_one_program_per_vector_core(
    fake_kernel: _FakeKernel,
    monkeypatch: pytest.MonkeyPatch,
    num_cores: int,
    num_tokens: int,
) -> None:
    """The grid is the core count, independent of batch size.

    This is the property that makes the launch shape stable across serving
    inputs; the work split moves into the kernel instead of the grid.
    """
    _set_cores(monkeypatch, num_cores)
    apply_penalties(*_inputs(num_tokens, 151936))
    assert fake_kernel.grid == (num_cores,)


def test_empty_batch_does_not_launch(fake_kernel: _FakeKernel, monkeypatch: pytest.MonkeyPatch) -> None:
    """num_tokens == 0 must return before launching, not launch zero work."""
    _set_cores(monkeypatch, 48)
    apply_penalties(*_inputs(0, 151936))
    assert fake_kernel.calls == []


# --------------------------------------------------------------------------
# Derived arguments
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_tokens,vocab_size",
    [
        (1, 151936),  # 38 blocks, fewer tiles than cores
        (8, 151936),
        (256, 151936),
        (8, 152064),  # 38 blocks, different tail
        (8, 16384),  # 4 blocks, exact multiple of the tile width
        (8, 1000),  # single block
    ],
)
def test_tile_split_matches_the_vocab_blocking(
    fake_kernel: _FakeKernel,
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    vocab_size: int,
) -> None:
    _set_cores(monkeypatch, 48)
    apply_penalties(*_inputs(num_tokens, vocab_size))

    n_blocks = _cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    args, kwargs = fake_kernel.args, fake_kernel.kwargs
    assert args is not None and kwargs is not None
    assert len(args) == _NUM_POSITIONAL_ARGS
    assert args[_VOCAB_SIZE_ARG] == vocab_size
    assert args[_TOTAL_TILES_ARG] == num_tokens * n_blocks
    assert args[_TILES_PER_WORKER_ARG] == _cdiv(num_tokens * n_blocks, 48)
    assert kwargs == {"NUM_VOCAB_BLOCKS": n_blocks, "BLOCK_SIZE": VOCAB_BLOCK_SIZE}


# --------------------------------------------------------------------------
# Partition
# --------------------------------------------------------------------------
def _worker_tiles(start: int, end: int, n_blocks: int) -> list[tuple[int, int]]:
    """Replay one program's loops, mirroring ``_penalties_kernel``."""
    if start >= end:
        return []
    out: list[tuple[int, int]] = []
    for token_idx in range(start // n_blocks, (end - 1) // n_blocks + 1):
        row_base = token_idx * n_blocks
        lo = max(start - row_base, 0)
        hi = min(end - row_base, n_blocks)
        out.extend((token_idx, block) for block in range(lo, hi))
    return out


@pytest.mark.parametrize("num_cores", [1, 7, 48])
@pytest.mark.parametrize("num_tokens,vocab_size", [(1, 151936), (5, 151936)])
def test_the_launchers_split_covers_every_tile_exactly_once(
    fake_kernel: _FakeKernel,
    monkeypatch: pytest.MonkeyPatch,
    num_cores: int,
    num_tokens: int,
    vocab_size: int,
) -> None:
    """Replay the kernel's loops over the split the launcher actually chose.

    Every tile must be owned by one program: that is why no atomics are needed.
    The bounds are taken from the recorded launch rather than recomputed, so a
    change to ``tiles_per_worker`` fails here. The kernel's own indexing is
    still only mirrored, and is proven on device by the worker-invariance test
    in tests/e2e/nightly.
    """
    _set_cores(monkeypatch, num_cores)
    apply_penalties(*_inputs(num_tokens, vocab_size))

    args, kwargs, grid = fake_kernel.args, fake_kernel.kwargs, fake_kernel.grid
    assert args is not None and kwargs is not None and grid is not None
    # The recorded arguments are opaque, so narrow them before doing arithmetic.
    total_tiles = args[_TOTAL_TILES_ARG]
    tiles_per_worker = args[_TILES_PER_WORKER_ARG]
    n_blocks = kwargs["NUM_VOCAB_BLOCKS"]
    assert isinstance(total_tiles, int)
    assert isinstance(tiles_per_worker, int)
    assert isinstance(n_blocks, int)

    covered: list[tuple[int, int]] = []
    for program_idx in range(grid[0]):
        lo = program_idx * tiles_per_worker
        covered.extend(_worker_tiles(lo, min(lo + tiles_per_worker, total_tiles), n_blocks))

    expected = {(t, b) for t in range(num_tokens) for b in range(n_blocks)}
    assert len(covered) == len(expected), "a tile is covered twice or not at all"
    assert set(covered) == expected


# --------------------------------------------------------------------------
# Specialization policy
# --------------------------------------------------------------------------
def _kernel_params() -> dict[str, str | None]:
    """Argument name -> annotation source, read from the module's own AST.

    Read from source rather than from the JITFunction so the assertion does
    not depend on triton internals, which differ between versions.
    """
    tree = ast.parse(inspect.getsource(penalties))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_penalties_kernel":
            return {arg.arg: (ast.unparse(arg.annotation) if arg.annotation else None) for arg in node.args.args}
    raise AssertionError("_penalties_kernel not found")


def test_only_structural_arguments_are_constexpr() -> None:
    """Workload values stay runtime; only tile geometry is compile-time.

    Every distinct ``tl.constexpr`` value compiles its own kernel, so a value
    that moves with batch size must not be annotated. ``NUM_VOCAB_BLOCKS`` and
    ``BLOCK_SIZE`` are fixed per model and shape the loops, so they stay.
    """
    params = _kernel_params()
    constexpr = {n for n, ann in params.items() if ann and ann.endswith("constexpr")}
    assert constexpr == {"NUM_VOCAB_BLOCKS", "BLOCK_SIZE"}


@pytest.mark.parametrize("name", ["vocab_size", "total_tiles", "tiles_per_worker"])
def test_batch_dependent_arguments_are_runtime(name: str) -> None:
    """These change with the serving input; annotating one would recompile."""
    params = _kernel_params()
    assert name in params, f"{name} is no longer a kernel argument"
    assert params[name] is None, f"{name} must stay a runtime argument"
