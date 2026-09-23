# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy


def create_eplb_policy(config: EplbConfig) -> AbstractEplbPolicy:
    """Create the configured Ascend EPLB policy."""
    return StairEplbPolicy(config.stair_config)
