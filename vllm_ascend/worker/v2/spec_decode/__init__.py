# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.
#
import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config import CompilationMode, VllmConfig

from vllm_ascend.ascend_config import validate_additional_config_bool


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.
    Use AscendEagleSpeculator when eagle is used.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    if speculative_config.method == "extract_hidden_states":
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if speculative_config.use_dspark():
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )

        return AscendDSparkSpeculator(vllm_config, device)
    if speculative_config.use_dflash():
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return AscendDFlash2Speculator(vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")


def _build_draft_additional_config(vllm_config: VllmConfig) -> dict[str, Any] | None:
    """Return a CPP-disabled copy when a PP target creates a PP=1 draft."""
    if vllm_config.parallel_config.pipeline_parallel_size <= 1:
        return None

    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return None

    scheduler_config = additional_config.get("scheduler_config")
    if scheduler_config is None:
        scheduler_config = {}
    elif not isinstance(scheduler_config, dict):
        return None

    # Match SchedulerConfig.from_additional_config: the nested form wins over
    # the deprecated top-level form when both are present.
    if "profiling_chunk_config" in scheduler_config:
        profiling_chunk_config = scheduler_config["profiling_chunk_config"]
        config_path = "additional_config.scheduler_config.profiling_chunk_config"
        nested = True
    elif "profiling_chunk_config" in additional_config:
        profiling_chunk_config = additional_config["profiling_chunk_config"]
        config_path = "additional_config.profiling_chunk_config"
        nested = False
    else:
        return None

    if not isinstance(profiling_chunk_config, dict):
        return None

    enabled = validate_additional_config_bool(
        profiling_chunk_config.get("enabled", False),
        f"{config_path}.enabled",
    )
    if not enabled:
        return None

    draft_profiling_chunk_config = profiling_chunk_config.copy()
    draft_profiling_chunk_config["enabled"] = False

    draft_additional_config = additional_config.copy()
    if nested:
        draft_scheduler_config = scheduler_config.copy()
        draft_scheduler_config["profiling_chunk_config"] = draft_profiling_chunk_config
        draft_additional_config["scheduler_config"] = draft_scheduler_config
    else:
        draft_additional_config["profiling_chunk_config"] = draft_profiling_chunk_config
    return draft_additional_config


@contextmanager
def disable_profiling_chunk_for_draft(vllm_config: VllmConfig) -> Iterator[None]:
    """Temporarily expose CPP-disabled input while constructing a PP=1 draft.

    The draft still goes through the normal ``VllmConfig.replace`` validation.
    Rebinding the target's ``additional_config`` only for that call lets the
    resulting draft retain the copied input while the target is restored.
    """
    draft_additional_config = _build_draft_additional_config(vllm_config)
    if draft_additional_config is None:
        yield
        return

    target_additional_config = vllm_config.additional_config
    vllm_config.additional_config = draft_additional_config
    try:
        yield
    finally:
        vllm_config.additional_config = target_additional_config
