# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACL graph replay metrics for the V2 model runner.

The V2 path never goes through ``ACLGraphWrapper``. Graph replay dispatches
per step: FULL mode replays through ``ModelAclGraphManager.run_fullgraph``
and PIECEWISE mode through ``ModelAclGraphManager.run_pw_graph``; both are
hooked so ``PIECEWISE``-only deployments (e.g. encoder-decoder fallback)
and ``FULL_AND_PIECEWISE`` prefill steps are covered too. Capture-side
metrics are listed as deleted in the ModelRunnerV2 DFX sheet, so only
replay is instrumented.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
    metric_enabled,
    record_observation,
)

_ACL_GRAPH_KEY = "acl_graph"

_REPLAY_SPEC = MetricSpec(
    "graph:replay:duration_ms",
    MetricType.HISTOGRAM,
    ("mode",),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_REPLAY_SPEC,)

_recorders: dict[str, Any] = {}


def _time_replay(mode: str, original_func, graph_manager, *args, **kwargs):
    """Time one graph replay and record it under the ``mode`` label."""
    # Level gate first: skip the timing reads on the per-step replay path when
    # the detail level is not enabled.
    if not metric_enabled(_REPLAY_SPEC):
        return original_func(graph_manager, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (graph_manager, *args), kwargs)
    record_observation(
        _recorders,
        _ACL_GRAPH_KEY,
        _SPECS,
        _REPLAY_SPEC,
        elapsed_ms,
        {"mode": mode},
        "acl_graph",
    )
    return result


def graph_replay_handler(original_func, graph_manager, *args, **kwargs):
    """Wall clock of V2 ``ModelAclGraphManager.run_fullgraph``."""
    return _time_replay("full", original_func, graph_manager, *args, **kwargs)


def graph_pw_replay_handler(original_func, graph_manager, *args, **kwargs):
    """Wall clock of V2 ``ModelAclGraphManager.run_pw_graph``."""
    return _time_replay("piecewise", original_func, graph_manager, *args, **kwargs)
