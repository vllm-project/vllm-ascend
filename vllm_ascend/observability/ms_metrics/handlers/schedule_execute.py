# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbol-hook metrics for the 调度与执行 directory.

Each handler wraps a function that already exists. Nothing in the scheduler
or model runner is edited. Per-step durations stay at detail so the default
basic level does not sample them.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_BASIC,
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "schedule_execute"

_SAMPLE_SPEC = MetricSpec(
    "executor:sample_tokens:duration_ms",
    MetricType.HISTOGRAM,
    ("runner",),
    level=METRIC_LEVEL_DETAIL,
)
_GRAPH_REPLAY_SPEC = MetricSpec(
    "graph:replay:duration",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
# The whole ``_update_seq_lens_cpu`` call, including the D2H event wait.
# The wait and the seq_lens assignment are not separate functions.
_SEQ_LENS_SPEC = MetricSpec(
    "executor:seq_lens_update:duration_ms",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
_PREEMPT_SPEC = MetricSpec(
    "scheduler:preempt_total",
    MetricType.COUNTER,
    ("reason",),
    level=METRIC_LEVEL_BASIC,
)
_SPECS = (_SAMPLE_SPEC, _GRAPH_REPLAY_SPEC, _SEQ_LENS_SPEC, _PREEMPT_SPEC)


def _record(spec: MetricSpec, value: float, labels: dict[str, str]) -> None:
    record_observation(_recorders, _KEY, _SPECS, spec, value, labels, "schedule_execute")


def sample_tokens_handler(original_func, runner, *args, **kwargs):
    """Wall clock of ModelRunner V2 ``sample_tokens``, including grammar and rejection."""
    result, elapsed_ms = invoke_original(original_func, (runner, *args), kwargs)
    _record(_SAMPLE_SPEC, elapsed_ms, {"runner": "v2"})
    return result


def graph_replay_handler(original_func, graph_manager, *args, **kwargs):
    """Wall clock of V2 ``ModelAclGraphManager.run_fullgraph``."""
    result, elapsed_ms = invoke_original(original_func, (graph_manager, *args), kwargs)
    _record(_GRAPH_REPLAY_SPEC, elapsed_ms, {})
    return result


def seq_lens_update_handler(original_func, runner, *args, **kwargs):
    """Wall clock of ``_update_seq_lens_cpu``, including the D2H synchronize."""
    result, elapsed_ms = invoke_original(original_func, (runner, *args), kwargs)
    _record(_SEQ_LENS_SPEC, elapsed_ms, {})
    return result


def schedule_preempt_handler(original_func, scheduler, *args, **kwargs):
    """Count requests reported preempted by ``schedule()``.

    ``schedule`` does not receive a reason. Capacity preemption and prefix-cache
    reset share this output, so the series is ``reason=scheduled``.
    """
    result = original_func(scheduler, *args, **kwargs)
    preempted = getattr(result, "preempted_req_ids", None) or ()
    if preempted:
        _record(_PREEMPT_SPEC, float(len(preempted)), {"reason": "scheduled"})
    return result


def recompute_preempt_handler(original_func, scheduler, *args, **kwargs):
    """Count decode requests sent back to prefill after KV offload fails."""
    result = original_func(scheduler, *args, **kwargs)
    _record(_PREEMPT_SPEC, 1.0, {"reason": "recompute"})
    return result


def lb_pause_preempt_handler(original_func, scheduler, *args, **kwargs):
    """Count DyntraLB pauses. This path keeps KV and does not call ``_preempt_request``."""
    result = original_func(scheduler, *args, **kwargs)
    _record(_PREEMPT_SPEC, 1.0, {"reason": "lb_pause"})
    return result
