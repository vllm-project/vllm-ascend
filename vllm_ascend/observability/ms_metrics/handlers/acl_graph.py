# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACL graph replay metric for the V2 model runner.

The V2 path never goes through ``ACLGraphWrapper``; graph replay is
``ModelAclGraphManager.run_fullgraph``. Capture-side metrics are listed as
deleted in the ModelRunnerV2 DFX sheet, so only replay is instrumented.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
    record_observation,
)

_ACL_GRAPH_KEY = "acl_graph"

_REPLAY_SPEC = MetricSpec(
    "graph:replay:duration",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_REPLAY_SPEC,)

_recorders: dict[str, Any] = {}


def graph_replay_handler(original_func, graph_manager, *args, **kwargs):
    """Wall clock of V2 ``ModelAclGraphManager.run_fullgraph``."""
    result, elapsed_ms = invoke_original(original_func, (graph_manager, *args), kwargs)
    record_observation(
        _recorders,
        _ACL_GRAPH_KEY,
        _SPECS,
        _REPLAY_SPEC,
        elapsed_ms,
        {},
        "acl_graph",
    )
    return result
