#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

"""Distributed helpers for runtime_config / runtime_guard (multi-DP safe).

Two concerns, one module (both are transport-layer, no config knowledge):

- **Role / sync-group selection**: which process may write the JSON, which
  process group (if any) carries config+dump broadcasts.
- **Task-bus collectives**: idle-friendly due-bit agreement plus
  source-owned ``broadcast_object`` lanes.

Task-bus pattern (last-PP TP broadcast **wave-head**):

1. **One** source-owned ``broadcast([wave_idx, due_0, due_1, …])`` from TP0 —
   ask which lanes have work (``sync_due_bits_from_src``; the symmetric
   ``all_reduce`` variant stays for the single-lane path)
2. For each due lane → its own ``broadcast_object`` (config and dump stay
   separate)

When nothing is due, ranks pay only the cheap due-vector collective.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

# Paths that already have a non-worker background reloader in this process.
_bg_reload_paths: set[str] = set()

# One-shot logs for file-poll fallbacks.
_rg_file_poll_fallback_logged = False
_rg_non_last_pp_file_logged = False

# Wave counters ride in a float32 payload slot; compare modulo 2**24 so
# the comparison stays exact across long-lived serving sessions.
_WAVE_IDX_MOD = 1 << 24


# ---------------------------------------------------------------------------
# Role / sync-group selection
# ---------------------------------------------------------------------------


def _world_group_or_none():
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group()
    except Exception:
        return None


def _dp_world_size_or_one() -> int:
    try:
        from vllm.distributed.parallel_state import get_dp_group

        return int(get_dp_group().world_size)
    except Exception:
        return 1


def _inner_dp_world_or_none():
    try:
        from vllm.distributed.parallel_state import get_inner_dp_world_group

        return get_inner_dp_world_group()
    except Exception:
        return None


def _is_last_pp_rank() -> bool:
    """True when this worker is on the last pipeline stage."""
    try:
        from vllm.distributed.parallel_state import get_pp_group

        return bool(get_pp_group().is_last_rank)
    except Exception:
        return False


def _log_non_last_pp_file_once() -> None:
    global _rg_non_last_pp_file_logged
    if _rg_non_last_pp_file_logged:
        return
    _rg_non_last_pp_file_logged = True
    from vllm_ascend.logger import init_logger_ascend

    init_logger_ascend(__name__).info(
        "[runtime_config] non-last-PP rank: config sync uses local file poll "
        "(collectives for config+dump stay on last-PP TP only)"
    )


def _log_file_poll_fallback_once(*, path: str, role: str) -> None:
    """One-shot reason why this process polls JSON instead of last-PP TP bus."""
    global _rg_file_poll_fallback_logged
    if _rg_file_poll_fallback_logged:
        return
    _rg_file_poll_fallback_logged = True
    from vllm_ascend.logger import init_logger_ascend

    logger = init_logger_ascend(__name__)
    if not _is_last_pp_rank():
        _log_non_last_pp_file_once()
        return
    if _dp_world_size_or_one() > 1:
        logger.info(
            "[runtime_config] multi-DP / tp_size<=1: no last-PP TP broadcast "
            "group → local file poll (no cross-DP sync); place a readable "
            "runtime_config_path on each EngineCore (per-node copy ok). path=%s %s",
            path,
            role,
        )
    else:
        logger.info(
            "[runtime_config] config hot-reload uses local file poll (no last-PP TP broadcast group). path=%s %s",
            path,
            role,
        )


def _runtime_config_sync_group_or_none():
    """Process group for config+dump broadcast, or None → local file poll.

    Communication domain is **last-PP × all TP** only (same ranks that dump KV):

    - last PP + ``tp_size>1``: ``get_tp_group()`` (TP0 due-broadcast + dump job bus)
    - non-last PP / no PP group / ``tp_size<=1``: ``None`` → each process polls JSON

    Never returns a cross-DP or cross-PP group.
    """
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        if not get_pp_group().is_last_rank:
            return None
        tp = get_tp_group()
    except Exception:
        return None
    if tp is None or int(getattr(tp, "world_size", 1) or 1) <= 1:
        return None
    return tp


def _is_distributed_worker_process() -> bool:
    """True when this process is (or is becoming) a distributed Worker.

    Used to keep the non-worker file-poll reloader off Workers. Prefer env
    markers (``RANK`` / ``LOCAL_RANK`` / ``VLLM_DP_RANK``) and a live world
    group — AscendConfig may run before ``RANK`` is set, so the background
    loop must re-check and exit if the process later becomes a Worker.
    """
    if os.environ.get("RANK") is not None:
        return True
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("VLLM_DP_RANK") is not None:
        return True
    return _world_group_or_none() is not None


def _process_role_tag() -> str:
    """Identify which process applied config (worker broadcast vs API file-poll)."""
    world = _world_group_or_none()
    if world is not None:
        return f"role=worker world_rank={world.rank}/{world.world_size}"
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        return f"role=worker RANK={rank_env} (world not ready)"
    if _is_distributed_worker_process():
        return "role=worker (pre-world)"
    return "role=non-worker"


def _is_json_writer() -> bool:
    """True if this process may write the runtime_config JSON (one leader per EngineCore).

    Order:
    1. ``inner_dp_world`` first rank (per-DP monitor when the group exists)
    2. Multi-DP without that group: TP0 ∧ PP0 (one writer on each DP replica)
    3. Full world first rank / ``RANK==0`` / single-process
    """
    inner = _inner_dp_world_or_none()
    if inner is not None and inner.world_size > 1:
        return bool(inner.is_first_rank)

    if _dp_world_size_or_one() > 1:
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            return bool(get_tp_group().is_first_rank and get_pp_group().is_first_rank)
        except Exception:
            pass

    world = _world_group_or_none()
    if world is not None and world.world_size > 1:
        return bool(world.is_first_rank)
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass
    return True


# ---------------------------------------------------------------------------
# Task-bus collectives
# ---------------------------------------------------------------------------


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


def sync_due_bits_from_src(
    group: Any,
    due_locals: Sequence[bool],
    *,
    wave_idx: int | None = None,
) -> list[bool]:
    """Agree on a due bit-vector via one ``broadcast`` from rank 0 of ``group``.

    Same per-wave merged-bus contract as :func:`sync_due_bits`, but bits are
    computed once by ``rank_in_group == 0`` and pushed to peers (dump jobs and
    periodic config reload are source-owned). Empirically this removes the
    wave-tail drain tax seen with a symmetric ``all_reduce`` under TP CPU
    phase drift (C2 hot path); peers must still enter the collective.

    ``wave_idx`` rides in the payload and is asserted on receivers so a rank
    that skipped a wave fails fast instead of consuming a crossed payload.
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
    payload = torch.zeros(len(bits) + 1, dtype=torch.float32, device=device)
    if _group_rank(group, 0) == 0:
        payload[0] = float(int(wave_idx) % _WAVE_IDX_MOD) if wave_idx is not None else -1.0
        for i, b in enumerate(bits):
            payload[i + 1] = 1.0 if b else 0.0
    torch.distributed.broadcast(
        payload,
        src=torch.distributed.get_process_group_ranks(gate)[0],
        group=gate,
    )
    out = [float(payload[i + 1].item()) >= 0.5 for i in range(len(bits))]
    src_idx = int(payload[0].item())
    if wave_idx is not None and int(wave_idx) >= 0 and src_idx != int(wave_idx) % _WAVE_IDX_MOD:
        raise RuntimeError(
            "[runtime_guard] merged-bus wave misalignment: source wave_idx="
            f"{src_idx} but local wave_idx={int(wave_idx)} "
            f"(rank_in_group={_group_rank(group, 0)}); a rank skipped a wave "
            "and the collectives are crossed - aborting to avoid silent corruption"
        )
    return out


def broadcast_when_due(
    group: Any,
    *,
    due: bool,
    payload: Any = None,
    build_payload: Callable[[], Any] | None = None,
    src: int = 0,
) -> Any | None:
    """If ``due``, ``broadcast_object`` from ``src``; else return ``None``.

    All ranks must pass the same agreed ``due`` (from
    :func:`sync_due_bits_from_src` on the merged bus, or :func:`sync_due_bits`
    on the single-lane path).
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
