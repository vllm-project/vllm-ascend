# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the vLLM 0.24 MiniMax M3 vision tower without its platform entrypoint.

Importing ``vllm.models.minimax_m3.common.vision_tower`` normally executes
``vllm.models.minimax_m3.__init__`` first. In vLLM 0.24 that entrypoint eagerly
imports the NVIDIA model on every non-ROCm platform, which requires CUDA custom
ops that are absent from an NPU/empty vLLM build.

This compatibility bridge executes the installed vLLM source module directly
under a stable vllm-ascend module name. It reuses the vLLM implementation
without copying it and can be removed once the upstream package entrypoint no
longer imports a hardware model while loading ``common`` modules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import vllm

_MODULE_NAME = "vllm_ascend.models._vllm_024_minimax_m3_vision_tower"
if vllm.__file__ is None:
    raise ImportError("Unable to locate the installed vLLM package.")
_SOURCE_PATH = (
    Path(vllm.__file__).resolve().parent
    / "models"
    / "minimax_m3"
    / "common"
    / "vision_tower.py"
)


def _load_vision_tower_module() -> ModuleType:
    loaded_module = sys.modules.get(_MODULE_NAME)
    if loaded_module is not None:
        return loaded_module

    if not _SOURCE_PATH.is_file():
        raise ImportError(
            "The vLLM MiniMax M3 vision tower source was not found at "
            f"{_SOURCE_PATH}. This vllm-ascend adapter requires vLLM 0.24."
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {_SOURCE_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


_vision_tower_module = _load_vision_tower_module()
MiniMaxVLVisionModel = _vision_tower_module.MiniMaxVLVisionModel

__all__ = ["MiniMaxVLVisionModel"]
