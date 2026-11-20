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
"""Backport vLLM PP + MTP runtime support.

The local Eagle/MTP drafter returns the draft tokens that belong to the model
output being processed. With PP batch_queue, EngineCore schedules a newer batch
before consuming the older output, so updating ``request.spec_token_ids`` from
``post_step`` observes live Request state from the newer schedule step.
"""

from __future__ import annotations

import copy
from functools import wraps

from vllm.logger import logger

_PATCHED = False


def _is_pd_prefill_node(vllm_config) -> bool:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return False

    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role == "kv_producer":
        return True

    is_kv_producer = getattr(kv_transfer_config, "is_kv_producer", False)
    is_kv_consumer = getattr(kv_transfer_config, "is_kv_consumer", False)
    return is_kv_producer and not is_kv_consumer


def _use_pp_mtp_runtime_patch(vllm_config, use_pp: bool) -> bool:
    if not _use_pp_ipc_runtime_patch(vllm_config, use_pp):
        return False
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return speculative_config is not None


def _use_pp_ipc_runtime_patch(vllm_config, use_pp: bool) -> bool:
    # The PP IPC runtime patch only applied to Model Runner V1, which was removed.
    return False


def _patch_model_runner_output() -> None:
    from vllm.v1 import outputs as outputs_mod

    model_runner_output_cls = outputs_mod.ModelRunnerOutput
    fields = getattr(model_runner_output_cls, "__dataclass_fields__", {})
    if "spec_token_ids" not in fields:
        model_runner_output_cls.spec_token_ids = None
        original_init = model_runner_output_cls.__init__
        if getattr(original_init, "_vllm_ascend_pp_mtp_patched", False):
            return

        @wraps(original_init)
        def _patched_init(self, *args, spec_token_ids=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.spec_token_ids = spec_token_ids

        _patched_init._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
        model_runner_output_cls.__init__ = _patched_init

    empty_output = outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
    if not hasattr(empty_output, "spec_token_ids"):
        empty_output.spec_token_ids = None


def _patch_engine_core() -> None:
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.post_step, "_vllm_ascend_pp_mtp_patched", False):
        return

    original_post_step = EngineCore.post_step

    @wraps(original_post_step)
    def _patched_post_step(self, model_executed: bool) -> None:
        scheduler = getattr(self, "scheduler", None)
        use_pp_mtp_runtime_patch = _use_pp_mtp_runtime_patch(
            getattr(scheduler, "vllm_config", None),
            getattr(scheduler, "use_pp", False),
        )
        if (
            use_pp_mtp_runtime_patch
            and getattr(self, "batch_queue", None) is not None
            and not getattr(self, "async_scheduling", False)
            and getattr(self, "use_spec_decode", False)
            and model_executed
        ):
            return
        return original_post_step(self, model_executed)

    _patched_post_step._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    EngineCore.post_step = _patched_post_step


def _patch_model_config_validation() -> None:
    from typing import get_args

    from vllm.config.model import ModelConfig
    from vllm.config.speculative import MTPModelTypes

    original_verify = ModelConfig.verify_with_parallel_config
    if getattr(original_verify, "_vllm_ascend_pp_mtp_patched", False):
        return

    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
        model_type = getattr(hf_config, "model_type", None)
        architectures = getattr(self, "architectures", ())
        is_eagle_drafter = (model_type == "eagle" or model_type == "speculators") and any(
            arch.startswith("Eagle") or arch.endswith("Eagle3") for arch in architectures
        )
        is_mtp_drafter = model_type in mtp_model_types
        is_dspark_drafter = any(
            arch in {"DSparkDraftModel", "K3DSparkModel", "Qwen3DSparkModel"} for arch in architectures
        )
        if (
            getattr(self, "runner", None) == "draft"
            and (is_eagle_drafter or is_mtp_drafter or is_dspark_drafter)
            and getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            # Local speculative drafters are loaded on the last PP stage
            # rather than partitioned across all PP stages. Keep normal target
            # validation intact, but validate these draft models as PP=1.
            logger.warning(
                "Validating local speculative drafter with pipeline_parallel_size=1 "
                "because it is loaded locally on the last pipeline stage."
            )
            patched_config = copy.copy(parallel_config)
            patched_config.pipeline_parallel_size = 1
            return original_verify(self, patched_config)
        return original_verify(self, parallel_config)

    _patched_verify_with_parallel_config._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    ModelConfig.verify_with_parallel_config = _patched_verify_with_parallel_config


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_model_runner_output()
    # The patch for engine core has been moved to patch_engine_core.py.
    _patch_model_config_validation()


_apply_patch()
