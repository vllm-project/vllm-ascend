# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
"""Static vLLM/vllm-ascend interface contract analysis engine."""

from .analysis_plans import (
    ANALYSIS_PLAN_VERSION,
    MAIN2MAIN_SCENARIO,
    VLLM_INTERFACE_SCENARIO,
    AnalysisPlan,
    resolve_analysis_plan,
)
from .call_contracts import (
    DirectCallDetector,
    ReturnContract,
    ReturnUse,
)
from .generator import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    InterfaceBoundaryGenerator,
    RepositoryIndex,
)

__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "ANALYSIS_PLAN_VERSION",
    "MAIN2MAIN_SCENARIO",
    "VLLM_INTERFACE_SCENARIO",
    "AnalysisPlan",
    "DirectCallDetector",
    "InterfaceBoundaryGenerator",
    "RepositoryIndex",
    "ReturnContract",
    "ReturnUse",
    "resolve_analysis_plan",
]
