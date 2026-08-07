# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.patch.worker import patch_mamba_ssd
from vllm_ascend.patch.worker.patch_mamba_ssd import _resolve_accelerator_index


def test_resolve_accelerator_index_with_device_index():
    assert _resolve_accelerator_index(torch.device("cpu", 3)) == 3


def test_resolve_accelerator_index_with_int():
    assert _resolve_accelerator_index(2) == 2


def test_chunk_scan_does_not_pass_removed_seqlen_kernel_argument(monkeypatch):
    captured_kwargs = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(**kwargs):
                captured_kwargs.update(kwargs)

            return launch

    monkeypatch.setattr(patch_mamba_ssd._chunk_scan, "_chunk_scan_fwd_kernel", FakeKernel())

    tensor = torch.zeros((1, 1, 1), dtype=torch.float32)
    patch_mamba_ssd._chunk_scan_fwd(
        cb=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        x=tensor,
        dt=tensor,
        dA_cumsum=tensor,
        C=tensor,
        states=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        cu_chunk_seqlens=torch.tensor([0, 1]),
        out=tensor,
        seq_idx=torch.zeros(1, dtype=torch.int32),
    )

    assert "seqlen" not in captured_kwargs
