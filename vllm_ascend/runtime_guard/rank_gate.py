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

"""Rank gating for detection and incident actions."""

from __future__ import annotations

from typing import Any

from vllm.distributed.parallel_state import get_pp_group, get_tp_group


def anomaly_check_rank_skip_reason(runner: Any) -> str | None:
    """None if this rank may run detectors; otherwise a short skip reason.

    Detection is last-PP TP0 only. Async scheduling's ``unique_reply_rank``
    forwards only the output-rank (TP0) return into ``enqueue_output`` /
    ``get_output``, so other TP ranks never materialize sampled ids on the
    executor path. Forcing ``get_output()`` on those ranks to spread detectors
    stalls the next ``execute_model`` TP collective.
    """
    if runner is None:
        return "no runner"
    try:
        if not get_pp_group().is_last_rank:
            return "not last PP rank"
    except Exception:
        return "PP group unavailable"
    try:
        tp_size = int(get_tp_group().world_size)
    except Exception:
        tp_size = int(getattr(runner, "tp_size", 1) or 1)
    if tp_size > 1 and runner_tp_rank(runner) != 0:
        return "not TP0"
    return None


def should_run_anomaly_check_on_rank(runner: Any) -> bool:
    return anomaly_check_rank_skip_reason(runner) is None


def runner_tp_rank(runner: Any) -> int:
    """TP rank within this worker's TP group.

    Process-group truth first: v1 model runners do NOT carry a ``tp_rank``
    attribute (vllm GPUModelRunner has none), so attribute fallbacks silently
    resolve 0 on every worker. ``get_tp_group().rank_in_group`` is set up by
    the time any hook fires.
    """
    try:
        return int(get_tp_group().rank_in_group)
    except Exception:
        return int(getattr(runner, "tp_rank", 0) or 0)


def runner_tp_world_size(runner: Any) -> int:
    """TP group size (``>= 1``) for this worker."""
    try:
        size = int(get_tp_group().world_size)
    except Exception:
        size = int(getattr(runner, "tp_size", 1) or 1)
    return max(1, size)


def runner_dp_rank(runner: Any) -> int:
    """Global data-parallel replica id for dump paths / ``rank_tag``.

    Prefer the runner / config / env global DP rank. Do **not** trust
    ``get_dp_group().rank_in_group`` first: with external multi-DP each replica
    is often its own world (``world_size==1``), so that value is always ``0`` and
    every replica would write ``dp0_…`` and collide.
    """
    try:
        val = getattr(runner, "dp_rank", None)
        if val is not None:
            return int(val)
    except (TypeError, ValueError):
        pass
    try:
        vllm_config = getattr(runner, "vllm_config", None)
        pc = getattr(runner, "parallel_config", None) or getattr(vllm_config, "parallel_config", None)
        if pc is not None:
            return int(getattr(pc, "data_parallel_rank", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        pass
    try:
        import os

        for key in ("VLLM_DP_RANK", "DP_RANK"):
            raw = os.environ.get(key)
            if raw is not None and str(raw).strip() != "":
                return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        from vllm.distributed.parallel_state import get_dp_group

        return int(get_dp_group().rank_in_group)
    except Exception:
        return 0


def runner_pp_rank(runner: Any) -> int | str:
    try:
        return int(get_pp_group().rank_in_group)
    except Exception:
        return "?"


def runner_cp_rank(runner: Any) -> int:
    """Combined CP rank (``pcp_rank * dcp_size + dcp_rank``) when CP is on."""
    try:
        dcp = int(getattr(runner, "dcp_rank", 0) or 0)
    except Exception:
        dcp = 0
    try:
        pcp = int(getattr(runner, "pcp_rank", 0) or 0)
    except Exception:
        pcp = 0
    try:
        dcp_size = int(getattr(runner, "dcp_size", 1) or 1)
    except Exception:
        dcp_size = 1
    if dcp_size < 1:
        dcp_size = 1
    return int(pcp) * int(dcp_size) + int(dcp)


def dump_rank_tag(runner: Any) -> str:
    """Directory / report tag for this worker's KV shard (dp/tp/pp/cp)."""
    return (
        f"dp{runner_dp_rank(runner)}_"
        f"tp{runner_tp_rank(runner)}_"
        f"pp{runner_pp_rank(runner)}_"
        f"cp{runner_cp_rank(runner)}"
    )


def should_dump_kv_on_rank(runner: Any) -> bool:
    """Dump last-PP KV shards on every TP rank of that stage (not other PP)."""
    del runner
    try:
        return bool(get_pp_group().is_last_rank)
    except Exception:
        return False


def is_action_leader_rank(runner: Any) -> bool:
    """Rank that writes reports and arms dump (last-PP TP0)."""
    try:
        if not get_pp_group().is_last_rank:
            return False
    except Exception:
        return False
    try:
        if get_tp_group().world_size > 1:
            return runner_tp_rank(runner) == 0
    except Exception:
        pass
    return True
