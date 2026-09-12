#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os
import shlex

LOG_FULL_ENV_VAR = "VLLM_ASCEND_LOG_FULL_ENV"

SENSITIVE_ENV_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY")


def is_full_env_logging_enabled() -> bool:
    return os.getenv(LOG_FULL_ENV_VAR, "0") == "1"


def mask_env_value(key: str, value: str) -> str:
    if any(token in key.upper() for token in SENSITIVE_ENV_TOKENS):
        return "***"
    return value


def log_full_environment(env: dict[str, str], logger: logging.Logger, *, prefix: str = "") -> None:
    if not is_full_env_logging_enabled():
        return
    logger.info("%sFull environment (%d vars):", prefix, len(env))
    for key in sorted(env):
        logger.info("%sENV %s=%s", prefix, key, mask_env_value(key, str(env[key])))


def format_env_prefix(env: dict[str, str]) -> list[str]:
    return [f"{key}={shlex.quote(mask_env_value(key, str(value)))}" for key, value in sorted(env.items())]
