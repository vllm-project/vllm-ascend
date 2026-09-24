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

"""Distributed role / sync-group helpers for runtime_config (multi-DP safe)."""

from __future__ import annotations

import os

# sync_mode values
SYNC_BROADCAST = "broadcast"
SYNC_FILE = "file"

# Paths that already have a non-worker background reloader in this process.
_bg_reload_paths: set[str] = set()

# One-shot logs for fallback / forced-mode messages.
_rg_multi_dp_file_fallback_logged = False
_rg_pp_force_file_logged = False


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


def _pp_world_size_or_one() -> int:
    try:
        from vllm.distributed.parallel_state import get_pp_group

        return int(get_pp_group().world_size)
    except Exception:
        return 1


def _inner_dp_world_or_none():
    try:
        from vllm.distributed.parallel_state import get_inner_dp_world_group

        return get_inner_dp_world_group()
    except Exception:
        return None


def _pp_forces_file_sync() -> bool:
    """True when PP>1: cross-PP config broadcast is unsafe → force file poll."""
    return _pp_world_size_or_one() > 1


def _log_pp_force_file_once() -> None:
    global _rg_pp_force_file_logged
    if _rg_pp_force_file_logged:
        return
    _rg_pp_force_file_logged = True
    from vllm_ascend.logger import init_logger_ascend

    init_logger_ascend(__name__).info(
        "[runtime_config] PP>1: forcing sync_mode=file "
        "(cross-PP broadcast deadlocks when stages are idle/dummy); "
        "dump uses last-PP TP all_reduce only"
    )


def _runtime_config_sync_group_or_none():
    """Process group for runtime_config broadcast, or None → local file poll.

    Never return the full multi-DP ``get_world_group()`` when ``dp_size>1``:
    after a request, one EngineCore may still ``execute_dummy_batch`` while the
    peer has gone idle — a cross-DP collective deadlocks.

    - PP>1: always ``None`` (sync_mode forced to file; no cross-PP collective)
    - dp==1: ``get_world_group()``
    - dp>1 + ``inner_dp_world``: that per-DP group (leader monitors JSON)
    - dp>1 without ``inner_dp_world``: ``None`` → file poll per rank
    """
    world = _world_group_or_none()
    if world is None:
        return None
    # PP>1: only the last PP rank samples; earlier ranks may be idle or on a
    # dummy batch in the same step, so a broadcast spanning PP ranks deadlocks.
    if _pp_forces_file_sync():
        return None
    if _dp_world_size_or_one() <= 1:
        return world
    return _inner_dp_world_or_none()


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
