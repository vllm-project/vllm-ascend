import gc
import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def scatter_nd_update_golden(var, indices, update):
    """CPU reference of npu_scatter_nd_update_sk.

    var/indices/update are CPU tensors. var is [a, b], indices is [n, 1],
    update is [n, b]. Indices are unique (or updates for duplicated indices
    are identical), so the row-by-row assignment is deterministic.
    """
    out = var.clone()
    for i in range(indices.shape[0]):
        out[int(indices[i, 0].item())] = update[i]
    return out


@pytest.mark.parametrize(
    "a",
    [16, 77],
)
@pytest.mark.parametrize(
    "b",
    [8, 128],
)
@pytest.mark.parametrize(
    "var_dtype, idx_dtype",
    [
        (torch.float16, torch.int32),
        (torch.bfloat16, torch.int64),
        (torch.float32, torch.int32),
        (torch.int8, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_scatter_nd_update_sk(a: int, b: int, var_dtype, idx_dtype, contiguous: bool):
    n = max(1, a // 4)

    # unique indices in [0, a)
    idx = np.random.choice(a, size=n, replace=False)
    indices_cpu = torch.from_numpy(idx.astype(np.int64)).view(-1, 1).to(idx_dtype)

    if var_dtype == torch.int8:
        update_cpu = torch.randint(-32, 32, (n, b), dtype=torch.int32).to(torch.int8)
    else:
        update_cpu = torch.randn(n, b, dtype=torch.float32).to(var_dtype)

    # var is a non-contiguous view (rows of width b in a buffer of width 2*b)
    # when contiguous=False, matching the real KV-cache layout.
    row_stride = b if contiguous else 2 * b
    var_cpu = torch.zeros(a, row_stride, dtype=var_dtype)[:, :b]

    var_npu = var_cpu.clone().npu()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(var_npu, indices_cpu.npu(), update_cpu.npu())

    golden = scatter_nd_update_golden(var_cpu, indices_cpu, update_cpu)
    # pure data movement, bit-exact comparison
    assert torch.equal(var_npu.cpu(), golden)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "var_dtype, idx_dtype",
    [
        (torch.float16, torch.int32),
        (torch.bfloat16, torch.int64),
        (torch.int8, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
)
def test_scatter_nd_update_sk_duplicate_indices(var_dtype, idx_dtype, contiguous: bool):
    """Duplicated indices take the sort path; the result is deterministic when
    all updates for the same index are identical."""
    a, b, n = 32, 64, 8
    row = 7  # duplicated target row
    indices_cpu = torch.full((n, 1), row, dtype=idx_dtype)

    if var_dtype == torch.int8:
        update_cpu = torch.randint(-32, 32, (1, b), dtype=torch.int32).to(torch.int8).repeat(n, 1)
    else:
        update_cpu = torch.randn(1, b, dtype=torch.float32).to(var_dtype).repeat(n, 1)

    row_stride = b if contiguous else 2 * b
    var_cpu = torch.zeros(a, row_stride, dtype=var_dtype)[:, :b]

    var_npu = var_cpu.clone().npu()
    torch.ops._C_ascend.npu_scatter_nd_update_sk(var_npu, indices_cpu.npu(), update_cpu.npu())

    golden = scatter_nd_update_golden(var_cpu, indices_cpu, update_cpu)
    assert torch.equal(var_npu.cpu(), golden)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
