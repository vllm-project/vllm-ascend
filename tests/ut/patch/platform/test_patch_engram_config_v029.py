# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dynamic_engram_config_is_excluded_from_vllm_replace(monkeypatch):
    class VllmConfig:
        def __post_init__(self):
            pass

    fake_config: Any = ModuleType("vllm.config")
    fake_utils: Any = ModuleType("vllm.config.utils")
    fake_utils.is_init_field = lambda cls, name: True
    fake_config.VllmConfig = VllmConfig
    fake_config.utils = fake_utils
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.utils", fake_utils)

    root = Path(__file__).resolve().parents[4]
    _load_module(
        "engram_config_v029_patch_test",
        root / "vllm_ascend/patch/platform/patch_engram_config_v029.py",
    )

    assert fake_utils.is_init_field(VllmConfig, "engram_config") is False
    assert fake_utils.is_init_field(VllmConfig, "model_config") is True
    assert fake_utils.is_init_field(object, "engram_config") is True
