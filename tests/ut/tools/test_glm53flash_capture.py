# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the worker's captured-logit layout contract."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def capture_method():
    # Load the actual method without importing the NPU-only model runtime.
    path = Path(__file__).resolve().parents[3] / "tools/ci/glm53flash_collect.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashWorker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "end_flash_forced_capture")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("batch,verify_width", [(3, 1), (3, 4), (4, 1), (4, 4)])
def test_capture_selects_first_verify_row_per_request(tmp_path, batch, verify_width):
    rows = np.arange(batch * verify_width, dtype=np.float32)[:, None]
    logits = np.broadcast_to(rows, (batch * verify_width, 154880))
    original = object()
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(model=SimpleNamespace()),
        _original_compute_logits=original,
        _captured_logits=[logits] * 8,
    )
    path = tmp_path / "logits.npy"
    result = capture_method()(worker, str(path), batch)
    assert result["shape"] == [8, batch, 154880]
    actual = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(actual[0, :, 0], np.arange(batch) * verify_width)
    assert worker.model_runner.model.compute_logits is original
    assert worker._captured_logits == []


@pytest.mark.parametrize("shapes", [[1, 3, 3, 3, 3, 3, 3, 3, 2], [3] * 7, [4] * 8])
def test_capture_rejects_unaligned_or_staggered_schedule(tmp_path, shapes):
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(model=SimpleNamespace()),
        _original_compute_logits=object(),
        _captured_logits=[np.broadcast_to(np.float32(0), (rows, 154880)) for rows in shapes],
    )
    path = tmp_path / "invalid.npy"
    with pytest.raises(AssertionError):
        capture_method()(worker, str(path), 3)
    assert not path.exists()


def test_capture_rejects_nonfinite_logits(tmp_path):
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(model=SimpleNamespace()),
        _original_compute_logits=object(),
        _captured_logits=[np.broadcast_to(np.float32(np.nan), (3, 154880))] * 8,
    )
    path = tmp_path / "invalid.npy"
    with pytest.raises(AssertionError):
        capture_method()(worker, str(path), 3)
    assert not path.exists()
