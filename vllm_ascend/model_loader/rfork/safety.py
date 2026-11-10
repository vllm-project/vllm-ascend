# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""RFork loader guards for configurations with mutable model storage."""

from typing import Any


def mutable_weights_bypass_reason(vllm_config: Any, model_config: Any) -> str | None:
    """Return why RFork must not register weights that can later be replaced."""
    if any(
        bool(getattr(config, "enable_sleep_mode", False))
        for config in (model_config, getattr(vllm_config, "model_config", None))
    ):
        return "sleep mode"
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        return "online weight transfer (weight_transfer_config)"
    return None
