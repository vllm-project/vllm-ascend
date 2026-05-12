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

import ast
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILING_CONFIG = PROJECT_ROOT / "vllm_ascend" / "profiling_config.py"


def _load_service_profiling_symbols():
    tree = ast.parse(PROFILING_CONFIG.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SERVICE_PROFILING_SYMBOLS_YAML":
                    return yaml.safe_load(ast.literal_eval(node.value))
    raise AssertionError("SERVICE_PROFILING_SYMBOLS_YAML is not defined")


def test_local_scheduler_symbols_reference_existing_classes_and_methods():
    symbols = _load_service_profiling_symbols()
    scheduler_symbols = [
        item["symbol"] for item in symbols if item["symbol"].startswith("vllm_ascend.core.")
    ]

    assert "vllm_ascend.core.scheduler:AscendScheduler.schedule" not in scheduler_symbols
    assert {
        "vllm_ascend.core.scheduler_dynamic_batch:SchedulerDynamicBatch.schedule",
        "vllm_ascend.core.scheduler_profiling_chunk:ProfilingChunkScheduler.schedule",
        "vllm_ascend.core.recompute_scheduler:RecomputeScheduler.schedule",
    }.issubset(set(scheduler_symbols))

    for symbol in scheduler_symbols:
        module_path, qualname = symbol.split(":", 1)
        class_name, method_name = qualname.split(".", 1)
        source_path = PROJECT_ROOT / (module_path.replace(".", "/") + ".py")

        assert source_path.exists(), f"Missing module for profiling symbol: {symbol}"
        source = source_path.read_text(encoding="utf-8")
        assert re.search(rf"^class\s+{re.escape(class_name)}\b", source, re.MULTILINE), symbol
        assert re.search(rf"^\s+def\s+{re.escape(method_name)}\b", source, re.MULTILINE), symbol
