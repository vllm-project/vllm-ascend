#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resolve persisted batch-consistency parameters for SMLA-family pytest cases."""

import ast
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .adapter import CaseAdapter
from .schema import get_q_lengths


POLICIES = ("auto", "on", "off")
ENABLED_KEY = "batch_consistency"
LEVEL_KEY = "npu_deterministic_level"
SEED_KEY = "batch_consistency_seed"
ORDER_KEY = "batch_consistency_order"
BATCH_SPLIT_KEY = "batch_consistency_batch_split"
MODE_BATCH_KEY = "batch_consistency_mode_batch"
TOKEN_SPLIT_KEY = "batch_consistency_token_split"
SHAPE_CHANGE_KEY = "batch_consistency_shape_change"

PARAMETER_KEYS = (
    SEED_KEY,
    ORDER_KEY,
    BATCH_SPLIT_KEY,
    MODE_BATCH_KEY,
    TOKEN_SPLIT_KEY,
    SHAPE_CHANGE_KEY,
)
PERSISTED_KEYS = (ENABLED_KEY, LEVEL_KEY) + PARAMETER_KEYS


def normalize_policy(value: Any) -> str:
    policy = str(value or "auto").strip().lower()
    if policy not in POLICIES:
        raise ValueError(
            f"CONFIG_ERROR: batch-consistency policy must be one of {POLICIES}, got {value!r}"
        )
    return policy


def parse_optional_bool(value: Any, name: str) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "on", "yes"):
            return True
        if normalized in ("false", "0", "off", "no"):
            return False
    raise ValueError(f"CONFIG_ERROR: {name} must be a boolean, got {value!r}")


def parse_optional_int(value: Any, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"CONFIG_ERROR: {name} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"CONFIG_ERROR: {name} must be an integer, got {value!r}"
        ) from error


def parse_optional_int_list(value: Any, name: str) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"CONFIG_ERROR: {name} must be an integer list, got {value!r}"
            ) from error
    if not isinstance(parsed, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        raise ValueError(f"CONFIG_ERROR: {name} must be an integer list, got {value!r}")
    return [int(item) for item in parsed]


def _has_parameter_value(params: Mapping[str, Any]) -> bool:
    return any(params.get(key) not in (None, "", []) for key in PARAMETER_KEYS)


def consistency_enabled(params: Mapping[str, Any], policy: str) -> bool:
    policy = normalize_policy(policy)
    if policy == "on":
        return True
    if policy == "off":
        return False
    configured = parse_optional_bool(params.get(ENABLED_KEY), ENABLED_KEY)
    return configured if configured is not None else _has_parameter_value(params)


def clear_consistency_params(params: Dict[str, Any]) -> None:
    for key in PERSISTED_KEYS:
        params.pop(key, None)


def prepare_consistency_params(params: Dict[str, Any], policy: str = "auto") -> bool:
    """Decide which consistency fields enter generated testcase data."""
    enabled = consistency_enabled(params, policy)
    if enabled:
        params[ENABLED_KEY] = True
    else:
        clear_consistency_params(params)
    return enabled


def _random_partition(total: int, rng: random.Random) -> Optional[List[int]]:
    if total < 2:
        return None
    part_count = rng.randint(2, min(total, 4))
    cuts = sorted(rng.sample(range(1, total), part_count - 1))
    boundaries = [0] + cuts + [total]
    return [boundaries[index + 1] - boundaries[index] for index in range(part_count)]


def _random_order(batch_size: int, rng: random.Random) -> Optional[List[int]]:
    if batch_size < 2:
        return None
    original = list(range(batch_size))
    order = original.copy()
    while order == original:
        rng.shuffle(order)
    return order


def _validate_positive_partition(
    values: Sequence[int], total: int, name: str
) -> List[int]:
    result = list(values)
    if len(result) < 2 or any(value <= 0 for value in result) or sum(result) != total:
        raise ValueError(
            f"CONFIG_ERROR: {name} must contain at least two positive integers "
            f"summing to {total}, got {result}"
        )
    return result


def _resolve_mode_batch(
    configured: Optional[int], valid_lengths: Sequence[int], rng: random.Random
) -> int:
    candidates = [index for index, length in enumerate(valid_lengths) if length >= 2]
    if not candidates:
        candidates = [
            index for index, length in enumerate(valid_lengths) if length >= 1
        ]
    if not candidates:
        raise ValueError(
            "CONFIG_ERROR: batch consistency requires at least one valid Q token"
        )
    if configured is None:
        return rng.choice(candidates)
    if configured < 0 or configured >= len(valid_lengths):
        raise ValueError(
            f"CONFIG_ERROR: {MODE_BATCH_KEY} must be within [0, {len(valid_lengths)}), "
            f"got {configured}"
        )
    return configured


def resolve_consistency_config(
    input_data: Dict[str, Any], policy: str = "auto", persist: bool = False
) -> Optional[Dict[str, Any]]:
    """Resolve explicit and generated values against the actual saved case shape."""
    params = input_data["params"]
    if not consistency_enabled(params, policy):
        return None

    configured_level = parse_optional_int(params.get(LEVEL_KEY), LEVEL_KEY)
    if configured_level not in (None, 3):
        raise ValueError(f"CONFIG_ERROR: {LEVEL_KEY} must be 3 when enabled")

    configured_seed = parse_optional_int(params.get(SEED_KEY), SEED_KEY)
    seed = (
        configured_seed
        if configured_seed is not None
        else random.SystemRandom().randrange(2**31)
    )
    if seed < 0:
        raise ValueError(f"CONFIG_ERROR: {SEED_KEY} must be non-negative")
    rng = random.Random(seed)

    batch_size = CaseAdapter(input_data).get_batch_size()
    valid_lengths = get_q_lengths(input_data, valid_only=True)

    order = parse_optional_int_list(params.get(ORDER_KEY), ORDER_KEY)
    if order is None:
        order = _random_order(batch_size, rng)
    elif sorted(order) != list(range(batch_size)) or order == list(range(batch_size)):
        raise ValueError(
            f"CONFIG_ERROR: {ORDER_KEY} must be a non-identity permutation of [0, {batch_size})"
        )

    batch_split = parse_optional_int_list(params.get(BATCH_SPLIT_KEY), BATCH_SPLIT_KEY)
    if batch_split is None:
        batch_split = _random_partition(batch_size, rng)
    else:
        batch_split = _validate_positive_partition(
            batch_split, batch_size, BATCH_SPLIT_KEY
        )

    configured_batch = parse_optional_int(params.get(MODE_BATCH_KEY), MODE_BATCH_KEY)
    mode_batch = _resolve_mode_batch(configured_batch, valid_lengths, rng)
    token_count = valid_lengths[mode_batch]

    token_split = parse_optional_int_list(params.get(TOKEN_SPLIT_KEY), TOKEN_SPLIT_KEY)
    if token_split is None:
        token_split = _random_partition(token_count, rng)
    else:
        token_split = _validate_positive_partition(
            token_split, token_count, TOKEN_SPLIT_KEY
        )

    shape_change = parse_optional_int_list(
        params.get(SHAPE_CHANGE_KEY), SHAPE_CHANGE_KEY
    )
    if shape_change is None:
        if token_count >= 2:
            shape_change = [rng.randint(1, token_count - 1), 0]
        else:
            shape_change = [1, 1]
    if len(shape_change) != 2:
        raise ValueError(
            f"CONFIG_ERROR: {SHAPE_CHANGE_KEY} must be [common_tokens, derived_extra_tokens]"
        )
    common_tokens, derived_extra_tokens = shape_change
    if (
        common_tokens < 1
        or common_tokens > token_count
        or derived_extra_tokens < 0
        or common_tokens + derived_extra_tokens == token_count
    ):
        raise ValueError(
            f"CONFIG_ERROR: {SHAPE_CHANGE_KEY} must select a valid common prefix and "
            f"change the Q length {token_count}, got {shape_change}"
        )

    config = {
        ENABLED_KEY: True,
        LEVEL_KEY: 3,
        SEED_KEY: seed,
        ORDER_KEY: order,
        BATCH_SPLIT_KEY: batch_split,
        MODE_BATCH_KEY: mode_batch,
        TOKEN_SPLIT_KEY: token_split,
        SHAPE_CHANGE_KEY: shape_change,
    }
    if persist:
        params.update(config)
    return config
