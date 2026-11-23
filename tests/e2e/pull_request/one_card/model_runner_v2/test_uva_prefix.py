# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu import buffer_utils

from vllm_ascend.patch.worker import patch_uva


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint64])
@pytest.mark.parametrize("source", ["numpy", "tensor", "list"])
def test_uva_prefix_pool_reuse(monkeypatch, dtype, source):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    pool = buffer_utils.UvaBufferPool(4, dtype, max_concurrency=2)
    for n in [16, 3, 8, 2, 393216, 7]:
        values = np.arange(n, dtype=np.int64)
        if dtype == torch.uint64:
            values = values.astype(np.uint64) + np.uint64(2**63)
        cpu = torch.from_numpy(values).to(dtype)
        data = cpu if source == "tensor" else cpu.numpy()
        if source == "list":
            data = data.tolist()
        output = pool.copy_to_uva(data)
        # Queue a reader before the next slot use, preserving pool ordering.
        saved = output.clone()
        torch.npu.synchronize()
        assert saved.cpu().tolist() == cpu.tolist()
        buf = pool._uva_bufs[pool._curr]
        assert buf._cpu.is_pinned()
        assert not buf._modified_indices
        assert buf._dirty_prefix_end == 0


def test_uva_prefix_and_sparse_npu(monkeypatch):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    buf = patch_uva.UvaBufferWrapper((16, 3), torch.int64)
    buf.np[12] = 8
    buf.np[:5] = 2
    buf.cpu[9:15:2] = 7
    output = buf.uva().clone()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), buf._cpu)


@pytest.mark.parametrize("source", ["numpy", "tensor"])
@pytest.mark.parametrize("column", [slice(None), 0])
def test_uva_multidimensional_prefix_and_contiguous_range(monkeypatch, source, column):
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    buf = patch_uva.UvaBufferWrapper((16, 3), torch.int64)
    view = buf.np if source == "numpy" else buf.cpu
    view[:5, column] = 2
    view[9:13, :] = 7
    output = buf.uva().clone()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), buf._cpu)
    assert not buf._modified_indices
    assert buf._dirty_prefix_end == 0
