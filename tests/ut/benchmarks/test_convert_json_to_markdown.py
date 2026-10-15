#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "benchmarks" / "scripts" / "convert_json_to_markdown.py"
MODULE_NAME = "vllm_ascend_benchmarks_convert_json_to_markdown"


@pytest.fixture(scope="module")
def convert_md():
    """Load the benchmark result conversion script as a module.

    ``tabulate`` is only declared in ``benchmarks/requirements-bench.txt`` and
    is not part of the vllm-ascend dependencies, so stub it to keep the module
    importable in the CPU UT environment.
    """
    sys.modules.pop(MODULE_NAME, None)
    with patch.dict(sys.modules, {"tabulate": MagicMock()}):
        spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def test_classify_result_file_by_name(convert_md):
    assert convert_md.classify_result_file(Path("serving_llama_qps_10.json")) == "serving"
    assert convert_md.classify_result_file(Path("latency_llama.json")) == "latency"
    assert convert_md.classify_result_file(Path("throughput_llama.json")) == "throughput"


def test_classify_result_file_unknown_category(convert_md):
    assert convert_md.classify_result_file(Path("misc.json")) is None


def test_classify_result_file_ignores_parent_directory(convert_md):
    """A results folder containing a keyword must not drive the classification.

    Regression test: the serving branch used to match the full path, so every
    file inside e.g. ``serving_results/`` was reported as a serving result and
    latency / throughput results were silently dropped.
    """
    serving_dir = Path("/data/serving_results")
    assert convert_md.classify_result_file(serving_dir / "latency_llama.json") == "latency"
    assert convert_md.classify_result_file(serving_dir / "throughput_llama.json") == "throughput"

    latency_dir = Path("/data/latency_results")
    assert convert_md.classify_result_file(latency_dir / "serving_llama_qps_10.json") == "serving"
