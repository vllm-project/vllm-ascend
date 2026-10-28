# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers, one module per feature.

Shared building blocks live in :mod:`.common`; feature modules only depend on
them. Re-export new feature handlers here so the YAML reference
``vllm_ascend.observability.ms_metrics.handlers:function_name`` stays stable.
"""

from vllm_ascend.observability.ms_metrics.handlers.eplb import (
    eplb_do_update_hotness_handler,
    eplb_expert_weight_transfer_handler,
)
from vllm_ascend.observability.ms_metrics.handlers.rl_lifecycle import (
    sleep_handler,
    update_weights_handler,
    wake_handler,
)
from vllm_ascend.observability.ms_metrics.handlers.schedule_execute import (
    graph_replay_handler,
    lb_pause_preempt_handler,
    recompute_preempt_handler,
    sample_tokens_handler,
    schedule_preempt_handler,
    seq_lens_update_handler,
)

__all__ = [
    "eplb_do_update_hotness_handler",
    "eplb_expert_weight_transfer_handler",
    "graph_replay_handler",
    "lb_pause_preempt_handler",
    "recompute_preempt_handler",
    "sample_tokens_handler",
    "schedule_preempt_handler",
    "seq_lens_update_handler",
    "sleep_handler",
    "update_weights_handler",
    "wake_handler",
]
