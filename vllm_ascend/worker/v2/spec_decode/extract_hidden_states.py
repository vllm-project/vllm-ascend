# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend MRV2 helpers for extract_hidden_states.

Upstream ``ExtractHiddenStatesSpeculator.propose`` requires the target forward
to return one auxiliary hidden-state tensor per configured
``eagle_aux_hidden_state_layer_ids`` entry. On Ascend, two things commonly
yield ``Expected N auxiliary hidden states, got M``:

* torch.compile graph-breaks at TP collectives, so a Python list of aux
  tensors appended across decoder layers can lose early-layer entries.
* Example layer ids such as ``[2, 18, 34]`` are Qwen3-8B (36-layer) specific
  and silently skip out-of-range ids on shallower models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.logger import logger
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesSpeculator,
)


def _eagle3_aux_holder(model: nn.Module) -> nn.Module:
    """Unwrap multimodal wrappers the same way ``SupportsEagle3`` does."""
    parent_ref: nn.Module = model
    if hasattr(parent_ref, "get_language_model"):
        parent_ref = parent_ref.get_language_model()
    elif hasattr(parent_ref, "language_model"):
        language_model = parent_ref.language_model
        parent_ref = language_model() if callable(language_model) else language_model
    return getattr(parent_ref, "model", parent_ref)


def _disable_torch_compile_for_aux_collection(model: nn.Module) -> None:
    """Keep target aux-hidden-state collection in eager Python.

    ``@support_torch_compile`` backends insert graph breaks at TP all-reduce /
    reduce-scatter. Auxiliary states are collected into a Python list across
    those breaks; NPU Dynamo can drop early-layer appends so a 3-id config
    arrives at propose() with only 2 tensors (``Worker_TPn`` traces).
    """
    holder = _eagle3_aux_holder(model)
    if hasattr(holder, "do_not_compile"):
        holder.do_not_compile = True


def validate_extract_hidden_states_aux_layers(model: nn.Module, speculator: Any) -> None:
    """Fail fast when configured aux ids cannot be collected by the target."""
    holder = _eagle3_aux_holder(model)
    aux_layers = tuple(getattr(holder, "aux_hidden_state_layers", ()) or ())
    layers = getattr(holder, "layers", None)
    if not aux_layers:
        raise ValueError(
            "extract_hidden_states requires auxiliary hidden-state layers on the "
            "target model, but none were registered. The target must support "
            "the EAGLE3 aux-hidden-state interface."
        )
    if len(aux_layers) != speculator.num_hidden_states:
        raise ValueError(
            "extract_hidden_states expected "
            f"{speculator.num_hidden_states} auxiliary hidden states from "
            "eagle_aux_hidden_state_layer_ids, but the target model registered "
            f"{aux_layers}."
        )
    if layers is None:
        return
    num_layers = len(layers)
    # Decoder-layer *inputs* use 0..num_layers-1; last-layer *output* uses
    # num_layers (see vLLM extract_hidden_states docs and deepseek_v2).
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


def configure_extract_hidden_states_target(
    model: nn.Module,
    speculator: Any,
) -> None:
    """Pin aux collection after the upstream load_model aux-layer setup."""
    _disable_torch_compile_for_aux_collection(model)
    if speculator is None:
        return
    validate_extract_hidden_states_aux_layers(model, speculator)
    holder = _eagle3_aux_holder(model)
    logger.info(
        "extract_hidden_states auxiliary layers: %s",
        getattr(holder, "aux_hidden_state_layers", ()),
    )


class AscendExtractHiddenStatesSpeculator(ExtractHiddenStatesSpeculator):
    """Upstream speculator with an Ascend-specific mismatch diagnostic."""

    def propose(  # type: ignore[override]
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if (
            not skip_attn_for_dummy_run
            and aux_hidden_states is not None
            and len(aux_hidden_states) != self.num_hidden_states
        ):
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            raise ValueError(
                f"Expected {self.num_hidden_states} auxiliary hidden states "
                f"from eagle_aux_hidden_state_layer_ids={layer_ids}, "
                f"got {len(aux_hidden_states)}. Each id must exist on the "
                "target model (valid range is 0..num_hidden_layers, where "
                "num_hidden_layers is the last-layer output). "
                "The documented [2, 18, 34] example is for 36-layer Qwen3-8B; "
                "use (2, num_layers//2, num_layers-3) for other depths."
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
