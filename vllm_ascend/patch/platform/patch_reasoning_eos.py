# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import json
from typing import Any

from pydantic import TypeAdapter, ValidationError
from vllm.sampling_params import SamplingParams

from vllm_ascend.sample.reasoning_config import AscendReasoningConfig

_ORIGINAL_UPDATE_ATTR = "_ascend_original_update_from_generation_config"
_MODEL_EOS_KEY = "vllm_ascend_model_eos_token_ids"


def _patch_sampling_params_model_eos() -> None:
    if hasattr(SamplingParams, "model_eos_token_ids"):
        return
    if hasattr(SamplingParams, _ORIGINAL_UPDATE_ATTR):
        return

    original_update = SamplingParams.update_from_generation_config
    setattr(SamplingParams, _ORIGINAL_UPDATE_ATTR, original_update)

    def update_from_generation_config(
        self: SamplingParams,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        if self.extra_args is None:
            self.extra_args = {}
        model_eos_token_ids = set(self.extra_args.get(_MODEL_EOS_KEY, ()))
        if eos_token_id is not None:
            model_eos_token_ids.add(eos_token_id)
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            if isinstance(eos_ids, int):
                model_eos_token_ids.add(eos_ids)
            else:
                model_eos_token_ids.update(eos_ids)

        original_update(self, generation_config, eos_token_id)
        self.extra_args[_MODEL_EOS_KEY] = tuple(sorted(model_eos_token_ids))

    def model_eos_token_ids(self: SamplingParams) -> set[int]:
        if self.extra_args is None:
            return set()
        return set(self.extra_args.get(_MODEL_EOS_KEY, ()))

    SamplingParams.update_from_generation_config = update_from_generation_config
    SamplingParams.model_eos_token_ids = property(model_eos_token_ids)  # type: ignore[attr-defined]


def update_reasoning_config_cli(parser: object) -> None:
    actions = getattr(parser, "_option_string_actions", {})
    action = actions.get("--reasoning-config")
    if action is not None:
        action.type = _parse_reasoning_config


def _parse_reasoning_config(value: str) -> AscendReasoningConfig | None:
    if value.lower() == "none":
        return None
    try:
        return TypeAdapter(AscendReasoningConfig).validate_json(value)
    except (ValidationError, json.JSONDecodeError) as error:
        raise argparse.ArgumentTypeError(repr(error)) from error


_patch_sampling_params_model_eos()
