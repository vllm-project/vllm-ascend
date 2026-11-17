# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbol-hook metrics for sequence parallelism and MoE communication.

``sp_padding_mask`` runs once per model forward (before the layer loop), so one
sample per step per model is enough to describe the padding overhead. Both
``num_tokens`` and the pad size are host-side ints, so the handler adds no
device sync.

``set_additional_forward_context`` is the upstream platform hook that runs once
per forward context. It selects the MoE communication method, which is where
ModelRunner V2 exposes the per-step choice.
"""

import logging
from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    metric_enabled,
    record_observation,
    safe_recording,
)

logger = logging.getLogger(__name__)

_recorders: dict[str, Any] = {}
_KEY = "parallel"
# Log-once guard: a missing TP group is a persistent state, so warn once per
# process instead of on every forward step.
_tp_size_error_logged = False

_SP_PADDING_SPEC = MetricSpec(
    "parallel:sp_padding_ratio",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
_MOE_COMM_SELECTION_SPEC = MetricSpec(
    "parallel:moe_comm_selection_total",
    MetricType.COUNTER,
    ("comm",),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_SP_PADDING_SPEC, _MOE_COMM_SELECTION_SPEC)


def _tp_size() -> int:
    """Return the TP world size, or 1 when it is unavailable.

    Failures are expected outside a TP-initialized process (e.g. kernel or
    precision tests calling ``sp_padding_mask`` directly), so the metric is
    skipped instead of propagating into the model forward.
    """
    global _tp_size_error_logged
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_world_size()
    except Exception:
        if not _tp_size_error_logged:
            _tp_size_error_logged = True
            logger.warning(
                "parallel metrics: tensor model parallel size unavailable; "
                "skipping parallel:sp_padding_ratio (warned once per process)",
                exc_info=True,
            )
        return 1


def sp_padding_handler(original_func, is_padding, hidden_states, *args, **kwargs):
    """Record the ratio of rows padded away by sequence-parallel sharding."""
    result = original_func(is_padding, hidden_states, *args, **kwargs)
    # Level gate first: skip the host-side token math and the TP size lookup
    # when the detail level is not enabled.
    if not metric_enabled(_SP_PADDING_SPEC):
        return result
    with safe_recording("parallel"):
        num_tokens = hidden_states.shape[0]
        tp_size = _tp_size()
        if tp_size > 1 and num_tokens > 0:
            sp_pad = (-num_tokens) % tp_size
            record_observation(
                _recorders,
                _KEY,
                _SPECS,
                _SP_PADDING_SPEC,
                float(sp_pad) / float(num_tokens),
                {},
                "parallel",
            )
    return result


def moe_comm_selection_handler(original_func, *args, **kwargs):
    """Count the MoE communication method chosen for each forward context.

    Non-MoE models return ``None`` and are skipped instead of being recorded as
    a ``comm=none`` series.
    """
    result = original_func(*args, **kwargs)
    comm_type = result.get("moe_comm_type") if isinstance(result, dict) else None
    if comm_type is not None:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _MOE_COMM_SELECTION_SPEC,
            1.0,
            {"comm": getattr(comm_type, "name", str(comm_type))},
            "parallel",
        )
    return result
