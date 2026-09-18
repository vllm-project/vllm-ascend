#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Task-bus collectives for runtime_config / runtime_guard.

Idle-friendly pattern (PP==1 broadcast **wave-head**):

1. **One** ``all_reduce(MAX, [due_0, due_1, …])`` — ask which lanes have work
2. For each due lane → its own ``broadcast_object`` (config and dump stay separate)

When nothing is due, ranks pay only the cheap due vector all_reduce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def _group_rank(group: Any, src: int = 0) -> int:
    try:
        return int(group.rank_in_group)
    except Exception:
        if bool(getattr(group, "is_first_rank", False)):
            return int(src)
        return int(src) + 1


def _cpu_gate(group: Any) -> Any | None:
    gate = getattr(group, "cpu_group", None)
    if gate is None:
        gate = getattr(group, "device_group", None)
    return gate


def sync_due_bits(group: Any, due_locals: Sequence[bool]) -> list[bool]:
    """Agree on a due bit-vector via ``all_reduce(MAX)``.

    All ranks in ``group`` must call with the same vector length. When ``group``
    is missing or world_size<=1, returns the local bits unchanged.
    """
    bits = [bool(x) for x in due_locals]
    if not bits:
        return []
    if group is None or int(getattr(group, "world_size", 1) or 1) <= 1:
        return bits

    import torch

    gate = _cpu_gate(group)
    if gate is None:
        return bits

    # HCCL device_group only accepts NPU tensors; Gloo cpu_group accepts CPU.
    device = "npu" if gate == getattr(group, "device_group", None) else "cpu"
    due_t = torch.tensor(
        [1.0 if b else 0.0 for b in bits],
        dtype=torch.float32,
        device=device,
    )
    torch.distributed.all_reduce(
        due_t,
        op=torch.distributed.ReduceOp.MAX,
        group=gate,
    )
    return [float(due_t[i].item()) >= 0.5 for i in range(len(bits))]


def broadcast_when_due(
    group: Any,
    *,
    due: bool,
    payload: Any = None,
    build_payload: Callable[[], Any] | None = None,
    src: int = 0,
) -> Any | None:
    """If ``due``, ``broadcast_object`` from ``src``; else return ``None``.

    All ranks must pass the same agreed ``due`` (from :func:`sync_due_bits`).
    """
    if not due:
        return None

    if group is None or int(getattr(group, "world_size", 1) or 1) <= 1:
        if build_payload is not None:
            return build_payload()
        return payload

    rank = _group_rank(group, src)
    if rank == int(src):
        src_obj = build_payload() if build_payload is not None else payload
    else:
        src_obj = None
    # Do not swallow broadcast failures: peers that already entered the
    # collective would hang if this rank returned early.
    return group.broadcast_object(src_obj, src=src)


def sync_task_bus(
    group: Any,
    *,
    due_local: bool,
    payload: Any = None,
    build_payload: Callable[[], Any] | None = None,
    src: int = 0,
) -> Any | None:
    """Single-lane bus: one due bit → optional one broadcast.

    Used when only dump (or only config) participates — e.g. PP>1 file mode
    dump drain on the last-PP TP group.
    """
    due = sync_due_bits(group, [due_local])[0]
    return broadcast_when_due(
        group,
        due=due,
        payload=payload,
        build_payload=build_payload,
        src=src,
    )
