# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

RFORK_ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"


def _load_module(monkeypatch, module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RFORK_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub(monkeypatch, module_name: str, **attributes):
    module = ModuleType(module_name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module
