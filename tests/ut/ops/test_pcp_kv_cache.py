# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from vllm_ascend.ops.triton.pcp_kv_cache import _get_pcp_kv_cache_rows


@pytest.mark.parametrize(
    "num_tokens,block_cols,element_size,num_caches,ub_kib,expected",
    [
        (32, 1024, 1, 1, 192, 1),  # Keep small batches spread across cores.
        (128, 1024, 1, 1, 192, 2),
        (4096, 256, 2, 1, 192, 8),  # Retain the maximum when the tile fits.
        (4096, 1024, 1, 1, 192, 4),  # C8 includes address and cast temporaries.
        (4096, 512, 2, 2, 192, 4),  # Budget both latent and RoPE cache tiles.
        (4096, 1024, 1, 1, 128, 2),  # Smaller devices reduce row batching.
        (4096, 2048, 1, 1, 192, 2),  # Wider rows reduce row batching.
        (4096, 1024, 4, 1, 192, 2),  # Account for larger element sizes.
        (4096, 1024, 1, 1, 160, 2),  # Round down when the budget fits three rows.
        (4096, 8192, 1, 1, 192, 1),  # Preserve the baseline for very wide rows.
    ],
)
def test_pcp_kv_row_budget(num_tokens, block_cols, element_size, num_caches, ub_kib, expected):
    with patch("vllm_ascend.ops.triton.pcp_kv_cache.get_ub_size_bytes", return_value=ub_kib * 1024):
        assert _get_pcp_kv_cache_rows(num_tokens, 48, block_cols, element_size, num_caches) == expected


# Exercises the batched branch of _copy_pcp_kv_cache_kernel with invalid slots.
# Runs as a subprocess because an unclamped kernel raises a device error at
# synchronize, which poisons the test process. The token count keeps
# _get_pcp_kv_cache_rows above one row on supported devices. The child also
# probes the device itself: the parent's torch.npu view can be mocked by the
# CPU UT conftest, or report a driver whose devices are not visible to this
# process, so the child reports "cannot exercise the kernel" through dedicated
# exit codes that the parent maps to skips.
_CHILD_EXIT_SINGLE_ROW = 3
_CHILD_EXIT_NO_DEVICE = 4
_BATCH_TOKENS = 240
_FAULT_INJECTION_SLOT = -(2**35)

_CHILD_TEMPLATE = """
import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit(__EXIT_NO_DEVICE__)
try:
    device_count = torch.npu.device_count()
except Exception:
    raise SystemExit(__EXIT_NO_DEVICE__)
if device_count == 0:
    raise SystemExit(__EXIT_NO_DEVICE__)
from vllm_ascend.ops.triton.pcp_kv_cache import _get_pcp_kv_cache_rows, copy_pcp_kv_cache
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

shape = (20, 32, 1, 656)
num_rows = shape[0] * shape[1]
raw = (torch.arange(num_rows * shape[3], dtype=torch.int32) % 256 - 128).to(torch.int8).reshape(shape)
src = raw.to("npu")
invalid_slots = (-1, __FAULT_SLOT__)
indices = [(i * 7) % (num_rows - 2) for i in range(__TOKENS__)]
for i in range(4, __TOKENS__, 8):
    indices[i] = invalid_slots[(i // 8) % 2]
slots = torch.tensor(indices, dtype=torch.int64, device="npu")
init_device_properties_triton()
rows = _get_pcp_kv_cache_rows(slots.numel(), get_vectorcore_num(), 1024, 1, 1)
if rows < 2:
    raise SystemExit(__EXIT_SINGLE_ROW__)
packed = copy_pcp_kv_cache((src,), slots)
torch.npu.synchronize()
out = packed.cpu()
for row, slot in enumerate(indices):
    want = (
        torch.zeros(shape[3], dtype=torch.int8)
        if slot < 0
        else raw[slot // shape[1], slot % shape[1], 0]
    )
    assert torch.equal(out[row], want), f"row {row} slot {slot} mismatch"
print("CHILD-OK")
"""

_CHILD_SNIPPET = (
    _CHILD_TEMPLATE.replace("__FAULT_SLOT__", str(_FAULT_INJECTION_SLOT))
    .replace("__TOKENS__", str(_BATCH_TOKENS))
    .replace("__EXIT_SINGLE_ROW__", str(_CHILD_EXIT_SINGLE_ROW))
    .replace("__EXIT_NO_DEVICE__", str(_CHILD_EXIT_NO_DEVICE))
)


def test_batched_invalid_slots_stay_in_bounds():
    """Invalid slots must not read outside the cache in the batched branch.

    The Ascend compiler lowers the kernel's masked indirect loads into
    unconditional copies followed by a select, so the load mask only zeroes
    the result and never guards the access; invalid slots must be clamped
    before address computation. The -1 sentinel alone under-reads by one row
    and faults only when the preceding page is unmapped, so it cannot fail a
    test deterministically. Mixing in a very large negative slot makes any
    unclamped address land far outside every mapping and fault at synchronize,
    while a clamped kernel just zero-fills those rows.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires a real Ascend NPU")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH")) if p)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SNIPPET],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("kernel subprocess timed out: invalid slots wedged the device")
    if proc.returncode == _CHILD_EXIT_SINGLE_ROW:
        pytest.skip("device tiles the kernel into single rows; batched branch not exercised")
    if proc.returncode == _CHILD_EXIT_NO_DEVICE:
        pytest.skip("no usable Ascend NPU in a clean subprocess; requires real hardware")
    assert proc.returncode == 0 and "CHILD-OK" in proc.stdout, (
        "invalid slots crashed or corrupted the batched copy:\n"
        f"exit={proc.returncode}\n{(proc.stderr or proc.stdout)[-2000:]}"
    )
