#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Compatibility shim — prefer ``vllm_ascend.observability.runtime_config``.

Public re-exports only. Submodules must be imported from
``vllm_ascend.observability.runtime_config``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "vllm_ascend.runtime_config has moved to vllm_ascend.observability.runtime_config",
    DeprecationWarning,
    stacklevel=2,
)

from vllm_ascend.observability.runtime_config import (
    ADDITIONAL_CONFIG_STRIP_KEYS,
    RuntimeConfig,
    RuntimeConfigBootstrap,
    build_runtime_config_from_additional,
    resolve_runtime_config_path,
    resolve_runtime_report_dir,
)

__all__ = [
    "ADDITIONAL_CONFIG_STRIP_KEYS",
    "RuntimeConfig",
    "RuntimeConfigBootstrap",
    "build_runtime_config_from_additional",
    "resolve_runtime_config_path",
    "resolve_runtime_report_dir",
]
