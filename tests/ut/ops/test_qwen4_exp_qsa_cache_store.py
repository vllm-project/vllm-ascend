# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import pytest

_CHILD_EXIT_NO_DEVICE = 4

_CHILD_SNIPPET = f"""
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit({_CHILD_EXIT_NO_DEVICE})
if not torch.npu.is_available():
    raise SystemExit({_CHILD_EXIT_NO_DEVICE})

from vllm_ascend.ops.triton.qwen4_exp.qsa import qsa_store_cache_rows

blocks = 5
page_size = 8
width = 128
block_gap = 37
block_stride = page_size * width + block_gap
storage_size = (blocks - 1) * block_stride + page_size * width

backing = torch.full(
    (storage_size,), -7, dtype=torch.bfloat16, device="npu"
)
cache = torch.as_strided(
    backing,
    size=(blocks, page_size, 1, width),
    stride=(block_stride, width, width, 1),
)
capacity = blocks * page_size
slot_values = [0, page_size - 1, page_size, capacity - 1, -1, capacity, 2**35]
slots = torch.tensor(slot_values, dtype=torch.int64, device="npu")
rows = (
    torch.arange(len(slot_values) * width, dtype=torch.int32)
    .remainder(97)
    .reshape(len(slot_values), 1, width)
    .to(dtype=torch.bfloat16, device="npu")
)

before = backing.cpu().clone()
rows_cpu = rows.cpu()
qsa_store_cache_rows(cache, slots, rows)
torch.npu.synchronize()
actual = backing.cpu()

expected = before.clone()
for row, slot in enumerate(slot_values):
    if 0 <= slot < capacity:
        block, token = divmod(slot, page_size)
        start = block * block_stride + token * width
        expected[start : start + width] = rows_cpu[row, 0]

assert cache.stride() == (block_stride, width, width, 1)
assert torch.equal(actual, expected)
print("CHILD-OK")
"""


def test_qsa_store_invalid_slots_stay_in_bounds_for_strided_cache():
    """Masked invalid slots must still use an in-bounds physical address."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires a real Ascend NPU")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in (repo_root, env.get("PYTHONPATH")) if path)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SNIPPET],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("QSA cache-store kernel timed out on invalid slots")

    if proc.returncode == _CHILD_EXIT_NO_DEVICE:
        pytest.skip("no usable Ascend NPU in a clean subprocess")
    assert proc.returncode == 0 and "CHILD-OK" in proc.stdout, (
        "QSA cache-store kernel crashed or corrupted its backing storage:\n"
        f"exit={proc.returncode}\n{(proc.stderr or proc.stdout)[-4000:]}"
    )
