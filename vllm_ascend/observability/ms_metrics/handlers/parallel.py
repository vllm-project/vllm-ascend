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

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "parallel"

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
    """Lazy import keeps the metric core importable without torch or vLLM."""
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )
    except ImportError:
        return 1
    return get_tensor_model_parallel_world_size()


def sp_padding_handler(original_func, is_padding, hidden_states, *args, **kwargs):
    """Record the ratio of rows padded away by sequence-parallel sharding."""
    num_tokens = hidden_states.shape[0]
    result = original_func(is_padding, hidden_states, *args, **kwargs)
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
