# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.patch.worker.patch_mamba_ssd import _resolve_accelerator_index


def test_resolve_accelerator_index_with_device_index():
    assert _resolve_accelerator_index(torch.device("cpu", 3)) == 3


def test_resolve_accelerator_index_with_int():
    assert _resolve_accelerator_index(2) == 2
