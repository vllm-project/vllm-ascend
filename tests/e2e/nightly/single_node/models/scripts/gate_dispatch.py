#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""Dispatcher for the reduced-checkpoint single-node gates.

The generic single-node runner only calls :func:`dispatch_gate`; the
model-specific gate wiring lives here instead of being embedded in
`test_single_node()`.
"""

from __future__ import annotations

from tests.e2e.nightly.single_node.models.scripts.single_node_config import (
    SingleNodeConfig,
)


def dispatch_gate(config: SingleNodeConfig) -> bool:
    """Run the gate declared by `config`; return whether one was dispatched."""
    if "glm5x_logits_gate" in config.extra_config:
        from tools.glm_reduced.run_logits_gate import run_nightly as run_logits_gate

        run_logits_gate(config)
        return True
    if "glm5x_perf_gate" in config.extra_config:
        from tools.glm_reduced.run_perf_gate import run_nightly as run_perf_gate

        run_perf_gate(config)
        return True
    if "reduced_model_gate" in config.extra_config:
        from tests.e2e.conftest import RemoteOpenAIServer
        from tools.glm_reduced.nightly import run_nightly

        run_nightly(config, RemoteOpenAIServer)
        return True
    return False
