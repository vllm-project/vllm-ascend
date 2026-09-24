# SPDX-License-Identifier: Apache-2.0
"""NPU accelerator views of registered pinned CPU storage."""

import torch


def can_get_npu_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> bool:
    """Whether CANN can map this tensor's pinned storage without registering it."""
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    return torch.ops._C_ascend.can_get_npu_view_from_cpu_tensor(cpu_tensor)


def get_npu_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Return an NPU-typed alias without copying or taking storage ownership."""
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    return torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu_tensor)
