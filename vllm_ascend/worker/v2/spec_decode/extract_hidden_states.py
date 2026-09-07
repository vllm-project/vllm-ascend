# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU glue for upstream extract_hidden_states (MRV2).

The speculator itself is upstream ``ExtractHiddenStatesSpeculator``. This
module only pins the target backbone to eager Python so aux hidden states
appended across decoder layers survive NPU Dynamo graph-breaks at TP
collectives, and rejects out-of-range ``eagle_aux_hidden_state_layer_ids``.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from vllm.logger import logger


def _eagle3_aux_holder(model: nn.Module) -> nn.Module:
    """Unwrap multimodal wrappers the same way ``SupportsEagle3`` does."""
    parent_ref: nn.Module = model
    if hasattr(parent_ref, "get_language_model"):
        parent_ref = parent_ref.get_language_model()
    elif hasattr(parent_ref, "language_model"):
        language_model = parent_ref.language_model
        parent_ref = language_model() if callable(language_model) else language_model
    return getattr(parent_ref, "model", parent_ref)


def configure_extract_hidden_states_target(
    model: nn.Module,
    speculator: Any,
) -> None:
    """Disable compile on the aux holder and fail fast on bad layer ids."""
    holder = _eagle3_aux_holder(model)
    if hasattr(holder, "do_not_compile"):
        # @support_torch_compile graph-breaks at TP all-reduce; NPU Dynamo can
        # drop early Python-list aux appends (Expected N, got M on Worker_TPn).
        holder.do_not_compile = True

    if speculator is None:
        return

    aux_layers = tuple(getattr(holder, "aux_hidden_state_layers", ()) or ())
    layers = getattr(holder, "layers", None)
    if not aux_layers:
        raise ValueError(
            "extract_hidden_states requires auxiliary hidden-state layers on the "
            "target model, but none were registered."
        )
    if len(aux_layers) != speculator.num_hidden_states:
        raise ValueError(
            "extract_hidden_states expected "
            f"{speculator.num_hidden_states} auxiliary hidden states from "
            "eagle_aux_hidden_state_layer_ids, but the target model registered "
            f"{aux_layers}."
        )
    if layers is not None:
        num_layers = len(layers)
        oob = [idx for idx in aux_layers if idx < 0 or idx > num_layers]
        if oob:
            default = (2, num_layers // 2, max(num_layers - 3, 0))
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids "
                f"{aux_layers} includes out-of-range ids {oob} for a "
                f"{num_layers}-layer model (valid range is 0..{num_layers}). "
                "The documented example [2, 18, 34] is for 36-layer Qwen3-8B. "
                f"Suggested EAGLE3 default: {list(default)}."
            )

    logger.info("extract_hidden_states auxiliary layers: %s", aux_layers)
