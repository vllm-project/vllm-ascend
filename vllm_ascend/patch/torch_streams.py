# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal CUDA stream API compatibility for NPU-only vLLM processes.

Shared by platform startup and standalone weight-transfer trainers.
No graph, IPC, allocation or generic torch.Event API is patched.
Remove these aliases once supported vLLM pairings use device-neutral stream
creation/contexts and no longer need CUDA stream compatibility.
"""

import torch


def patch_cuda_streams() -> None:
    """Install the same three aliases at every NPU process entry point.

    Reapplying is intentional: standalone trainers need not initialize the
    platform patches, and existing wrappers may have changed the aliases.
    Call before using vLLM streams; mixed CUDA/NPU processes are unsupported.
    """
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.stream = torch.npu.stream
    torch.cuda.current_stream = torch.npu.current_stream
