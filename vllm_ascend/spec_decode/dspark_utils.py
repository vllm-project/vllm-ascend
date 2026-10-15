# SPDX-License-Identifier: Apache-2.0
"""Shared DSpark target auxiliary-layer configuration, independent of PD roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_dspark_aux_layer_ids(vllm_config: VllmConfig) -> tuple[int, ...]:
    """Resolve ordered target capture boundaries without loading a draft model.

    DSpark runners use the upstream checkpoint resolver, including its model-
    specific layer-to-capture-boundary conversion. Target-only P runners use
    explicit connector metadata, which already contains capture boundaries.
    Topology, execution mode and prefix-cache restrictions belong to the
    context backend, not to this shared configuration utility.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None and speculative.method == "dspark":
        layer_ids = get_eagle3_aux_layers_from_config(speculative)
    else:
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        extra = getattr(transfer, "kv_connector_extra_config", None) or {}
        layer_ids = extra.get("dspark_aux_hidden_state_layer_ids")
    if layer_ids is None:
        return ()
    num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    if (
        not isinstance(layer_ids, (list, tuple))
        or not layer_ids
        or any(type(index) is not int or not 0 <= index <= num_layers for index in layer_ids)
        or list(layer_ids) != sorted(set(layer_ids))
    ):
        raise ValueError("DSpark auxiliary IDs must be ordered unique target-layer boundaries within the model.")
    return tuple(layer_ids)
