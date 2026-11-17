# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-replay (routed-experts capture) state for RL rollouts.

``rl:routed_experts_state{state}`` is a 0/1 gauge with three states:

- ``enabled``: ``aux_output_config.enable_return_routed_experts`` is on.
- ``ready``: the auxiliary-output data plane is resolved after KV cache init.
  Every rank reports this stage; only the TP owner rank builds the buffer, so
  the value is rank-invariant and no ``rank`` label is needed.
- ``capturing``: the capture hook has run at least once in this process.

The states are reported from the code that owns each transition instead of a
symbol hook: ``capture`` is a per-MoE-layer hot path, and the latch keeps it to
one recording per process.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_BASIC,
    MetricSpec,
    MetricType,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "rl_router_replay"

_ROUTED_EXPERTS_STATE_SPEC = MetricSpec(
    "rl:routed_experts_state",
    MetricType.GAUGE,
    ("state",),
    level=METRIC_LEVEL_BASIC,
)
_SPECS = (_ROUTED_EXPERTS_STATE_SPEC,)


def record_routed_experts_state(state: str, value: int) -> None:
    """Report one ``rl:routed_experts_state`` line; ``state`` is a 0/1 gauge."""
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _ROUTED_EXPERTS_STATE_SPEC,
        value,
        {"state": state},
        "rl_router_replay",
    )
