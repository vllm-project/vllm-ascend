#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Inject ``VppContinuationOutput`` into ``vllm.v1.outputs``.

vLLM 0.22.1 upstream does NOT define ``VppContinuationOutput`` — it is part
of the VPP feature. vllm-ascend installs the class here as a monkey-patch
on ``vllm.v1.outputs`` so that:

  * the worker (``vllm_ascend.worker.model_runner_v1``) and the engine core
    (``vllm.v1.engine.core``) reference the **same class object**;
  * ``pickle`` round-trips between worker and engine core resolve to a
    class the engine core can ``isinstance``-check.

The injection runs at import time. Any later
``from vllm.v1.outputs import VppContinuationOutput`` (in either process)
gets this exact class.
"""

from dataclasses import dataclass

import vllm.v1.outputs as _vllm_outputs
from vllm.v1.outputs import KVConnectorOutput


@dataclass
class VppContinuationOutput:
    """Indicates VPP execution yielded and should be resumed."""

    next_vp_stage: int = 0
    kv_connector_output: KVConnectorOutput | None = None


# Install on the vllm module so all callers (worker & engine core) see the
# same class object. Idempotent — if a future vLLM upstream adds this class
# already, we just keep theirs.
if not hasattr(_vllm_outputs, "VppContinuationOutput"):
    _vllm_outputs.VppContinuationOutput = VppContinuationOutput
else:
    # Upstream already provides it; keep their definition but rebind our
    # local symbol so direct imports still work.
    VppContinuationOutput = _vllm_outputs.VppContinuationOutput


# Make ``pickle`` locate the class under ``vllm.v1.outputs`` (matching the
# symlink above) rather than ``vllm_ascend.patch.platform.patch_outputs``.
# Both worker and engine-core processes execute this module at startup,
# so the symbol is always present in the receiving process.
VppContinuationOutput.__module__ = "vllm.v1.outputs"


__all__ = ["VppContinuationOutput"]
