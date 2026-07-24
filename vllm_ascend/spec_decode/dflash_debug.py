#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Rate-limited tensor diagnostics for the 310P DFlash correctness path.

The trace intentionally remains in the runtime code.  DFlash acceptance
problems are often caused by a numerically valid but semantically wrong tensor
at a boundary (target auxiliary states, context KV, SplitFuse metadata, draft
logits, or rejection verification).  Shape-only logs are insufficient for
those failures, so the first calls also copy a bounded tensor sample to CPU.

The copy synchronizes the NPU.  To keep the permanent diagnostics affordable,
each owner/key pair logs only the first few calls and then power-of-two calls.
No environment variable or mutable module-global switch is required.
"""

from typing import Any

import torch
from vllm.logger import logger

_INITIAL_TRACE_CALLS = 4
_TENSOR_HEAD_VALUES = 16
_TENSOR_SAMPLE_VALUES = 1024


def _next_trace_call(owner: Any, key: str) -> tuple[int, bool]:
    """Return the per-owner call number and whether this call should be logged."""
    attr = f"_dflash_trace_{key.replace('.', '_')}_calls"
    call = int(getattr(owner, attr, 0)) + 1
    setattr(owner, attr, call)
    should_log = call <= _INITIAL_TRACE_CALLS or (call & (call - 1) == 0)
    return call, should_log


def tensor_summary(tensor: Any) -> str:
    """Build a bounded value-and-statistics summary without dumping full tensors."""
    if tensor is None:
        return "None"
    if isinstance(tensor, (list, tuple)):
        values = ", ".join(f"{index}:{tensor_summary(value)}" for index, value in enumerate(tensor[:4]))
        suffix = ", ..." if len(tensor) > 4 else ""
        return f"{type(tensor).__name__}[{values}{suffix}]"
    if not torch.is_tensor(tensor):
        return repr(tensor)

    detached = tensor.detach()
    metadata = (
        f"shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"device={detached.device} stride={tuple(detached.stride())} "
        f"contiguous={detached.is_contiguous()}"
    )
    if detached.numel() == 0:
        return f"{metadata} empty"

    # One bounded device-to-host synchronization per tensor.  Statistics are
    # computed on the CPU sample so logging does not issue several NPU syncs.
    sample = detached.reshape(-1)[:_TENSOR_SAMPLE_VALUES].to(device="cpu", dtype=torch.float32)
    head = sample[:_TENSOR_HEAD_VALUES].tolist()
    finite = torch.isfinite(sample)
    finite_count = int(finite.sum())
    if finite_count:
        finite_sample = sample[finite]
        stats = (
            f"sample_n={sample.numel()} finite={finite_count} "
            f"min={float(finite_sample.min()):.7g} "
            f"max={float(finite_sample.max()):.7g} "
            f"mean={float(finite_sample.mean()):.7g} "
            f"abs_mean={float(finite_sample.abs().mean()):.7g}"
        )
    else:
        stats = f"sample_n={sample.numel()} finite=0"
    return f"{metadata} {stats} head={head}"


def trace_dflash_tensors(
    owner: Any,
    key: str,
    *,
    message: str = "",
    **tensors: Any,
) -> bool:
    """Log bounded DFlash diagnostics and return whether a trace was emitted."""
    call, should_log = _next_trace_call(owner, key)
    if not should_log:
        return False

    summaries = [f"{name}=({tensor_summary(value)})" for name, value in tensors.items()]
    logger.info(
        "[dflash/trace][%s][call=%d] %s%s",
        key,
        call,
        message,
        (" | " if message and summaries else "") + " | ".join(summaries),
    )
    return True
