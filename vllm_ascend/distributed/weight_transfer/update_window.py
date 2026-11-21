# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Mark the Engram weight-update window around a receive.

``update_weights`` is the call that hands a worker the update's payload, so the
Engram loader can install the rows it receives instead of re-reading the checkpoint.
The Ascend engines that run the receiving side mix this in.

The window is deliberately opened here and not in ``start_weight_update`` /
``finish_weight_update``: those two are pinned to the upstream implementation by
``test_hccl_engine.py::test_inherits_upstream_orchestration`` so that the Ascend
engines cannot drift from upstream's orchestration. The final layer replay runs
inside ``finish_weight_update``, outside this window; the loader still installs the
payload there because ``_host_storage_was_replaced`` sees the rebuilt parameter.

The flag lives in this module rather than under ``models/deepseek_v41/engram``: every
Ascend engine imports it, and reaching into the Engram package would pull the
DeepSeek-V4.1 models - and with them the model registry - into the weight update of an
unrelated model. The Engram loader imports the flag from here.
"""

from typing import Any

_DEPTH = 0


def begin_engram_weight_update() -> None:
    """Open one weight-update transaction (nests)."""
    global _DEPTH
    _DEPTH += 1


def end_engram_weight_update() -> None:
    """Close one weight-update transaction; a stray call is a no-op.

    The engines always pair this with ``begin`` in a ``finally``, but a failed
    transaction must never leave the window open: the next startup-style load would
    then install the checkpoint's payload as if it were an update.
    """
    global _DEPTH
    _DEPTH = max(0, _DEPTH - 1)


def engram_weight_update_active() -> bool:
    """Whether a weight update owns the model right now."""
    return _DEPTH > 0


class EngramWeightUpdateWindow:
    """Mark the window in which a receive owns the Engram rows."""

    def update_weights(self, *args: Any, **kwargs: Any) -> Any:
        begin_engram_weight_update()
        try:
            return super().update_weights(*args, **kwargs)  # type: ignore[misc]
        finally:
            end_engram_weight_update()
