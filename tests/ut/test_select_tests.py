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
# This file is a part of the vllm-ascend project.

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path


def _load_select_tests():
    script = Path(__file__).parents[2] / ".github/workflows/scripts/select_tests.py"
    spec = importlib.util.spec_from_file_location("select_tests", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pytest_node_exists_handles_functions_classes_and_parameters(tmp_path):
    module = _load_select_tests()
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "async def test_async():\n    pass\n\nclass TestGroup:\n    def test_case(self):\n        pass\n",
        encoding="utf-8",
    )

    assert module._pytest_node_exists(f"{test_file}::test_async[param]")
    assert module._pytest_node_exists(f"{test_file}::TestGroup")
    assert module._pytest_node_exists(f"{test_file}::TestGroup::test_case[value]")
    assert not module._pytest_node_exists(f"{test_file}::test_removed")
    assert not module._pytest_node_exists(f"{test_file}::TestGroup::test_removed")


def test_stale_nodeid_falls_back_to_file_target(tmp_path, monkeypatch, capsys):
    module = _load_select_tests()
    monkeypatch.chdir(tmp_path)
    test_file = Path("tests/e2e/pull_request/one_card/test_sample.py")
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_current():\n    pass\n", encoding="utf-8")
    groups: defaultdict[str, list[str]] = defaultdict(list)
    monkeypatch.setattr(module, "_route_e2e_file", lambda _target: "a2-1")

    module._route_explicit_test_target(f"{test_file.as_posix()}::test_removed", groups)

    assert list(groups.values()) == [[test_file.as_posix()]]
    assert "falling back to file-level target" in capsys.readouterr().err
