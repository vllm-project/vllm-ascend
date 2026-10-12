# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import wraps
from importlib import import_module


def load_cann_gdn_ops() -> tuple[Callable | None, Callable | None]:
    """Resolve optional GDN operators during attention initialization."""
    try:
        ops = import_module("cann_ops_transformer.ops")
    except (ImportError, OSError, RuntimeError):
        return None, None
    return getattr(ops, "fused_gdn_gating", None), getattr(ops, "chunk_gated_delta_rule_compute_wy", None)


def bind_cann_gdn_ops_on_init(init: Callable) -> Callable:
    """Bind GDN ops on the class actually instantiated by the model."""
    if getattr(init, "_binds_cann_gdn_ops", False):
        return init

    @wraps(init)
    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        # A wrapped parent constructor may already have bound the operators.
        if not hasattr(self, "fused_gdn_gating_op"):
            self.fused_gdn_gating_op, self.chunk_gated_delta_rule_compute_wy_op = load_cann_gdn_ops()

    initialize._binds_cann_gdn_ops = True
    return initialize
