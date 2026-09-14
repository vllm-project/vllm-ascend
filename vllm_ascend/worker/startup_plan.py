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
# WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""StartPlan helpers for Ascend.

Upstream fingerprints include ``torch.version.cuda``, which is empty on NPU.
Fold ``torch_npu`` into the key so a CANN / torch_npu bump cannot reuse a
stale KV-cache memory plan.
"""

from __future__ import annotations

import hashlib
import json

import torch_npu
from vllm.v1.worker import startup_plan as startup_plan_mod
from vllm.v1.worker.startup_plan import maybe_apply_startup_plan, maybe_save_startup_plan


def _wrap_compute_plan_fingerprint(original):
    def compute_plan_fingerprint(vllm_config, rank: int, world_size: int) -> str:
        base = original(vllm_config, rank, world_size)
        extra = json.dumps({"torch_npu": torch_npu.__version__}, sort_keys=True)
        return hashlib.sha256(f"{base}:{extra}".encode()).hexdigest()[:16]

    compute_plan_fingerprint._vllm_ascend_npu_factors = True  # type: ignore[attr-defined]
    return compute_plan_fingerprint


def _install_npu_fingerprint() -> None:
    current = startup_plan_mod.compute_plan_fingerprint
    if getattr(current, "_vllm_ascend_npu_factors", False):
        return
    startup_plan_mod.compute_plan_fingerprint = _wrap_compute_plan_fingerprint(current)


_install_npu_fingerprint()

compute_plan_fingerprint = startup_plan_mod.compute_plan_fingerprint

__all__ = [
    "compute_plan_fingerprint",
    "maybe_apply_startup_plan",
    "maybe_save_startup_plan",
]
