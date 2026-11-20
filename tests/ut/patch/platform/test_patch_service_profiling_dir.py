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

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import ProfilerConfig

from vllm_ascend import envs

_MODULE_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_service_profiling_dir.py"
_spec = importlib.util.spec_from_file_location("_ascend_profiler_dir_patch", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_apply_profiler_dir = _module._apply_profiler_dir
_maybe_enable_profiling = _module._maybe_enable_profiling


def test_apply_profiler_dir_is_noop_when_env_is_empty():
    profiler_config = ProfilerConfig()

    with patch.object(envs, "VLLM_ASCEND_PROFILER_DIR", ""):
        assert _apply_profiler_dir(profiler_config) is False

    assert profiler_config.profiler is None
    assert profiler_config.torch_profiler_dir == ""
    assert profiler_config.ignore_frontend is False


def test_apply_profiler_dir_enables_torch_profiler(tmp_path):
    profiler_config = ProfilerConfig()

    with patch.object(envs, "VLLM_ASCEND_PROFILER_DIR", f"  {tmp_path}  "):
        assert _apply_profiler_dir(profiler_config) is True

    assert profiler_config.profiler == "torch"
    assert profiler_config.torch_profiler_dir == str(tmp_path)
    assert profiler_config.ignore_frontend is True


def test_apply_profiler_dir_keeps_explicit_profiler_config(tmp_path):
    profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/explicit/traces")

    with patch.object(envs, "VLLM_ASCEND_PROFILER_DIR", str(tmp_path)):
        assert _apply_profiler_dir(profiler_config) is False

    assert profiler_config.torch_profiler_dir == "/explicit/traces"
    assert profiler_config.ignore_frontend is False


def test_apply_profiler_dir_keeps_explicit_output_dir(tmp_path):
    profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/explicit/traces")

    with patch.object(envs, "VLLM_ASCEND_PROFILER_DIR", str(tmp_path)):
        assert _apply_profiler_dir(profiler_config) is False

    assert profiler_config.torch_profiler_dir == "/explicit/traces"


def test_maybe_enable_profiling_creates_missing_config(tmp_path):
    args = SimpleNamespace(profiler_config=None)

    with patch.object(envs, "VLLM_ASCEND_PROFILER_DIR", str(tmp_path)):
        _maybe_enable_profiling(args)

    assert isinstance(args.profiler_config, ProfilerConfig)
    assert args.profiler_config.profiler == "torch"
    assert args.profiler_config.torch_profiler_dir == str(tmp_path)
