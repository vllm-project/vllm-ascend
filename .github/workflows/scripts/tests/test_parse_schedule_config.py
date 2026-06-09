#!/usr/bin/env python3
#
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
"""Unit tests for parse_schedule_config.py.

Run with:
    pytest .github/workflows/scripts/tests/test_parse_schedule_config.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse_schedule_config import (  # noqa: E402
    AccuracyFramework,
    BaseFramework,
    ModelFramework,
    OpsFramework,
    PeriodicCase,
    _build_frameworks,
    _dedupe_cases,
    _derive_name,
    _detect_chip,
    _detect_multi_node_type,
    _find_framework,
    _matches_filter,
    _normalize_path,
    main,
)

# ---------------------------------------------------------------------------
# Minimal runner_map for tests (avoids needing runner_label.json on disk)
# ---------------------------------------------------------------------------
RUNNER_MAP: dict[tuple[str, int], str] = {
    ("a3", 1): "linux-aarch64-a3-1",
    ("a3", 2): "linux-aarch64-a3-2",
    ("a3", 4): "linux-aarch64-a3-4",
    ("a3", 8): "linux-aarch64-a3-8",
    ("a3", 16): "linux-aarch64-a3-16",
    ("a2", 1): "linux-aarch64-a2b3-1",
    ("a2", 2): "linux-aarch64-a2b3-2",
    ("a2", 4): "linux-aarch64-a2b3-4",
    ("a2", 8): "linux-aarch64-a2b3-8",
    ("310p", 1): "linux-aarch64-310p-1",
    ("310p", 2): "linux-aarch64-310p-2",
    ("310p", 4): "linux-aarch64-310p-4",
}


def parse(path: str) -> PeriodicCase:
    cases = parse_entry(path)
    assert len(cases) == 1
    return cases[0]


def parse_entry(path: str) -> list[PeriodicCase]:
    frameworks = _build_frameworks(RUNNER_MAP)
    framework = _find_framework(_normalize_path(path), frameworks)
    return framework.expand(path)


def matrix_item(case: PeriodicCase, output_name: str) -> dict:
    framework = _find_framework(case.path, _build_frameworks(RUNNER_MAP))
    grouped = framework.group([case])
    items = grouped[output_name]
    assert len(items) == 1
    return items[0]


# Repo root, so directory tests resolve relative paths regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _list_yaml(rel_dir: str) -> list[Path]:
    return sorted((_REPO_ROOT / rel_dir).rglob("*.yaml"))


def _read_github_output(path: Path) -> dict[str, object]:
    outputs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        outputs[key] = json.loads(value)
    return outputs


class TestCaseNames:
    def test_derive_name_preserves_filename_stem(self):
        assert _derive_name("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml") == (
            "DeepSeek-V3.2-W8A8"
        )
        assert _derive_name("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml") == (
            "GLM5_1-W8A8-EP-external_dp"
        )

    def test_dedupe_records_duplicate_names(self):
        first = parse("tests/e2e/schedule/model/Qwen/one_card/Duplicate.yaml")
        duplicate = parse("tests/e2e/schedule/model/GLM/two_card/Duplicate.yaml")
        unique = parse("tests/e2e/schedule/model/Qwen/one_card/Unique.yaml")

        deduped, duplicates = _dedupe_cases([first, duplicate, unique])

        assert deduped == [first, unique]
        assert duplicates == {"Duplicate": [first, duplicate]}

    def test_main_prints_duplicate_warning_in_summary(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / "schedule_config.yaml"
        config.write_text(
            "\n".join(
                [
                    "periodic_tests:",
                    "  - name: manual",
                    "    files:",
                    "      - tests/e2e/schedule/model/Qwen/one_card/Duplicate.yaml",
                    "      - tests/e2e/schedule/model/GLM/two_card/Duplicate.yaml",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        runner_label = tmp_path / "runner_label.json"
        runner_label.write_text(
            """
{
  "runner-a3-1": {"chip": "a3", "npu_num": 1},
  "runner-a3-2": {"chip": "a3", "npu_num": 2}
}
""",
            encoding="utf-8",
        )
        output_path = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "parse_schedule_config.py",
                "--config",
                str(config),
                "--runner-label",
                str(runner_label),
                "--event-name",
                "workflow_dispatch",
            ],
        )

        main()

        err = capsys.readouterr().err
        assert "WARNING: duplicate test case names detected; kept the first occurrence:" in err
        assert "kept: tests/e2e/schedule/model/Qwen/one_card/Duplicate.yaml" in err
        assert "duplicate: tests/e2e/schedule/model/GLM/two_card/Duplicate.yaml" in err


class TestMainOutputCompatibility:
    def test_main_writes_all_output_fields(self, tmp_path, monkeypatch):
        config = tmp_path / "schedule_config.yaml"
        config.write_text(
            "\n".join(
                [
                    "periodic_tests:",
                    "  - name: manual",
                    "    files:",
                    "      - tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml",
                    "      - tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml",
                    "      - tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml",
                    "      - tests/e2e/schedule/ops/one_card/test_fused_moe.py",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        runner_label = tmp_path / "runner_label.json"
        runner_label.write_text(
            """
{
  "runner-a2-1": {"chip": "a2", "npu_num": 1},
  "runner-a2-4": {"chip": "a2", "npu_num": 4},
  "runner-a3-2": {"chip": "a3", "npu_num": 2}
}
""",
            encoding="utf-8",
        )
        output_path = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "parse_schedule_config.py",
                "--config",
                str(config),
                "--runner-label",
                str(runner_label),
                "--event-name",
                "workflow_dispatch",
            ],
        )

        main()

        outputs = _read_github_output(output_path)
        assert set(outputs) == {
            "single_node_matrix",
            "multi_node_matrix",
            "accuracy_matrix",
            "ops_matrix",
            "image_build_targets",
            "selected_cases_summary",
        }
        assert len(outputs["single_node_matrix"]) == 1
        assert len(outputs["multi_node_matrix"]) == 1
        assert len(outputs["accuracy_matrix"]) == 1
        assert len(outputs["ops_matrix"]) == 1
        assert outputs["image_build_targets"] == ["a2", "a3"]
        assert outputs["multi_node_matrix"][0]["multi_node_type"] == "external_dp"
        assert outputs["multi_node_matrix"][0]["size"] == 2
        assert "runner" not in outputs["multi_node_matrix"][0]


@pytest.fixture
def tmp_accuracy_mixed_dir():
    """Create a throwaway mixed-chip accuracy dir under an unregistered resource
    (eight_card), then remove it. Used to exercise per-chip directory grouping."""
    base = _REPO_ROOT / "tests/e2e/schedule/accuracy/eight_card"
    dummy = "model_name: dummy/model\nmodel_type: vllm\n"
    for chip in ("a2", "a3"):
        d = base / chip
        d.mkdir(parents=True, exist_ok=True)
        (d / "_dummy.yaml").write_text(dummy, encoding="utf-8")
    try:
        yield base
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. model one_node -> single_node
# ---------------------------------------------------------------------------
class TestModelOneNode:
    def test_framework_and_route(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        assert case.framework == "model"
        assert case.route == "single_node"

    def test_resource(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        assert case.resource_type == "node"
        assert case.resource_num == 1
        assert case.resource_dir == "one_node"

    def test_chip_default_a3(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        assert case.chip == "a3"

    def test_runner(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        assert case.runner == "linux-aarch64-a3-16"

    def test_framework_match(self):
        path = "tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml"
        framework = _find_framework(path, _build_frameworks(RUNNER_MAP))
        assert isinstance(framework, ModelFramework)

    def test_matrix_item(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        item = matrix_item(case, "single_node_matrix")
        assert item["config_path"] == "tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml"
        assert item["chip"] == "a3"
        assert item["runner"] == "linux-aarch64-a3-16"


# ---------------------------------------------------------------------------
# 2. model four_card A2 -> single_node + a2
# ---------------------------------------------------------------------------
class TestModelFourCardA2:
    def test_framework_and_route(self):
        case = parse("tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml")
        assert case.framework == "model"
        assert case.route == "single_node"

    def test_chip_a2(self):
        case = parse("tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml")
        assert case.chip == "a2"

    def test_runner(self):
        case = parse("tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml")
        assert case.runner == "linux-aarch64-a2b3-4"

    def test_resource(self):
        case = parse("tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml")
        assert case.resource_type == "card"
        assert case.resource_num == 4

    def test_a22b_not_matched_as_a2(self):
        # A22B in filename should NOT be treated as A2 chip
        case = parse("tests/e2e/schedule/model/Qwen/two_node/Qwen3-235B-A22B.yaml")
        assert case.chip == "a3"


# ---------------------------------------------------------------------------
# 3. model two_node external_dp -> multi_node + external_dp
# ---------------------------------------------------------------------------
class TestModelTwoNodeExternalDp:
    def test_framework_and_route(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.framework == "model"
        assert case.route == "multi_node"

    def test_multi_node_type_external(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.multi_node_type == "external_dp"

    def test_size(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.size == 2

    def test_framework_match(self):
        path = "tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml"
        framework = _find_framework(path, _build_frameworks(RUNNER_MAP))
        assert isinstance(framework, ModelFramework)

    def test_matrix_item(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        item = matrix_item(case, "multi_node_matrix")
        assert item["multi_node_type"] == "external_dp"
        assert item["size"] == 2
        assert "runner" not in item


# ---------------------------------------------------------------------------
# 4. model two_node without external_dp -> multi_node + internal_dp
# ---------------------------------------------------------------------------
class TestModelTwoNodeInternal:
    def test_multi_node_type_internal(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/two_node/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml")
        assert case.multi_node_type == "internal_dp"

    def test_external_without_dp_is_internal(self):
        # Filename contains "external" but NOT "external_dp" -> internal_dp
        type_ = _detect_multi_node_type("tests/e2e/schedule/model/GLM/two_node/GLM5-W8A8-external.yaml")
        assert type_ == "internal_dp"

    def test_external_dp_in_stem_only(self):
        # "external_dp" must be in the file stem, not a subdirectory
        type_ = _detect_multi_node_type("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert type_ == "external_dp"


# ---------------------------------------------------------------------------
# 5. accuracy direct config YAML -> accuracy route with config_paths
# ---------------------------------------------------------------------------
_ACC_FILE = "tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml"


class TestAccuracyDirectConfig:
    def test_framework_and_route(self):
        case = parse(_ACC_FILE)
        assert case.framework == "accuracy"
        assert case.route == "accuracy"

    def test_chip_from_dir(self):
        case = parse(_ACC_FILE)
        assert case.chip == "a2"

    def test_runner(self):
        case = parse(_ACC_FILE)
        assert case.runner == "linux-aarch64-a2b3-1"

    def test_framework_match(self):
        framework = _find_framework(_ACC_FILE, _build_frameworks(RUNNER_MAP))
        assert isinstance(framework, AccuracyFramework)

    def test_config_paths_set_no_model_list(self):
        case = parse(_ACC_FILE)
        assert case.case_path == [_ACC_FILE]
        item = matrix_item(case, "accuracy_matrix")
        assert item["config_paths"] == [_ACC_FILE]
        assert "model_list" not in item

    def test_py_accuracy_rejected(self):
        # Accuracy entries are YAML configs only; a .py path is no longer routable.
        with pytest.raises(ValueError, match="must be YAML"):
            parse("tests/e2e/schedule/accuracy/four_card/test_acc_example.py")


# ---------------------------------------------------------------------------
# 5b. Accuracy directory expansion (grouped by chip into config_paths)
# ---------------------------------------------------------------------------
_ACC_A2_DIR = "tests/e2e/schedule/accuracy/one_card/a2/"


class TestAccuracyDirectoryExpansion:
    def test_single_chip_dir_yields_one_grouped_case(self):
        cases = parse_entry(_ACC_A2_DIR)
        assert len(cases) == 1
        case = cases[0]
        assert case.framework == "accuracy"
        assert case.chip == "a2"
        assert case.runner == "linux-aarch64-a2b3-1"
        # all real a2 configs are bundled into one job
        assert len(case.case_path) == len(_list_yaml(_ACC_A2_DIR))
        assert all(p.endswith(".yaml") for p in case.case_path)

    def test_matrix_item_has_config_paths_no_model_list(self):
        case = parse_entry(_ACC_A2_DIR)[0]
        item = matrix_item(case, "accuracy_matrix")
        assert "model_list" not in item
        assert isinstance(item["config_paths"], list) and item["config_paths"]

    def test_mixed_chip_dir_groups_per_chip(self, tmp_accuracy_mixed_dir):
        # eight_card/ holds a2/ and a3/ subdirs with dummy dict configs
        cases = parse_entry("tests/e2e/schedule/accuracy/eight_card/")
        chips = sorted(c.chip for c in cases)
        assert chips == ["a2", "a3"]
        for c in cases:
            assert c.framework == "accuracy"
            assert all(f"/{c.chip}/" in p for p in c.case_path)


# ---------------------------------------------------------------------------
# 6. ops one_card -> ops route
# ---------------------------------------------------------------------------
class TestOpsOneCard:
    def test_framework_and_route(self):
        case = parse("tests/e2e/schedule/ops/one_card/test_fused_moe.py")
        assert case.framework == "ops"
        assert case.route == "ops"

    def test_tests_field_set(self):
        case = parse("tests/e2e/schedule/ops/one_card/test_fused_moe.py")
        assert case.case_path == "tests/e2e/schedule/ops/one_card/test_fused_moe.py"

    def test_framework_match(self):
        path = "tests/e2e/schedule/ops/one_card/test_fused_moe.py"
        framework = _find_framework(path, _build_frameworks(RUNNER_MAP))
        assert isinstance(framework, OpsFramework)

    def test_ops_a2_from_filename(self):
        case = parse("tests/e2e/schedule/ops/four_card/test_matmul_allreduce_add_rmsnorm_a2.py")
        assert case.chip == "a2"
        assert case.runner == "linux-aarch64-a2b3-4"

    def test_ops_one_node(self):
        case = parse("tests/e2e/schedule/ops/one_node/test_dispatch_ffn_combine.py")
        assert case.framework == "ops"
        assert case.route == "ops"
        assert case.resource_type == "node"
        assert case.resource_num == 1


# ---------------------------------------------------------------------------
# 7. tests/e2e/accuracy path -> error
# ---------------------------------------------------------------------------
class TestOldAccuracyPathError:
    def test_old_accuracy_path_fails(self):
        with pytest.raises(ValueError, match="tests/e2e/accuracy"):
            parse("tests/e2e/accuracy/one_card/accuracy-group-1-a2.yaml")

    def test_non_schedule_path_fails(self):
        with pytest.raises(ValueError, match="No framework matched"):
            parse("tests/e2e/pull_request/one_card/test_foo.yaml")


# ---------------------------------------------------------------------------
# 8. Path outside tests/e2e/schedule -> error
# ---------------------------------------------------------------------------
class TestInvalidPathError:
    def test_absolute_like_path_fails(self):
        with pytest.raises(ValueError):
            parse("some/other/path/one_card/test.yaml")

    def test_missing_resource_dir_fails(self):
        with pytest.raises(ValueError, match="No resource directory"):
            parse("tests/e2e/schedule/model/DeepSeek/DeepSeek-V3.yaml")

    def test_directory_missing_resource_dir_fails(self):
        with pytest.raises(ValueError, match="No resource directory"):
            parse_entry("tests/e2e/schedule/model/DeepSeek/")

    def test_accuracy_node_resource_fails(self):
        with pytest.raises(ValueError, match="card resources"):
            parse("tests/e2e/schedule/accuracy/one_node/accuracy.yaml")


# ---------------------------------------------------------------------------
# 9b. model/ directory layer (framework is the 4th path segment)
# ---------------------------------------------------------------------------
class TestModelDirectoryLayer:
    def test_user_example_routes(self):
        case = parse("tests/e2e/schedule/model/Kimi/one_node/Kimi-K2.5.yaml")
        assert case.framework == "model"
        assert case.route == "single_node"
        assert case.resource_dir == "one_node"

    def test_resource_segment_under_model_routes(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/two_node/DeepSeek-V3.1-BF16.yaml")
        assert case.route == "multi_node"
        assert case.resource_dir == "two_node"

    def test_bare_family_without_model_layer_fails(self):
        # Old layout (no model/ layer) is no longer accepted.
        with pytest.raises(ValueError, match="No framework matched"):
            parse("tests/e2e/schedule/Kimi/one_node/Kimi-K2.5.yaml")

    def test_model_without_family_routes(self):
        case = parse("tests/e2e/schedule/model/one_node/Kimi-K2.5.yaml")
        assert case.framework == "model"
        assert case.route == "single_node"
        assert case.resource_dir == "one_node"


# ---------------------------------------------------------------------------
# 9. test_filter matching
# ---------------------------------------------------------------------------
class TestFilterMatching:
    def _case(self, path: str) -> PeriodicCase:
        return parse(path)

    def test_filter_all(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "all")

    def test_filter_by_family_segment(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "Qwen")

    def test_filter_by_filename(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "Qwen3-235B-A22B-W8A8.yaml")

    def test_filter_by_stem(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "Qwen3-235B-A22B-W8A8")

    def test_filter_by_full_path(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")

    def test_filter_no_match(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert not _matches_filter(case, "DeepSeek")

    def test_filter_comma_separated_list_matches_any_item(self):
        case = self._case("tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml")
        assert _matches_filter(case, "DeepSeek,Qwen")
        assert not _matches_filter(case, "DeepSeek,GLM")

    def test_filter_by_resource_segment(self):
        case = self._case("tests/e2e/schedule/ops/one_node/test_dispatch_ffn_combine.py")
        assert _matches_filter(case, "ops")

    def test_filter_accuracy_family(self):
        case = self._case("tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml")
        assert _matches_filter(case, "accuracy")

    def test_filter_external_dp_filename(self):
        case = self._case("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert _matches_filter(case, "GLM5_1-W8A8-EP-external_dp.yaml")
        assert _matches_filter(case, "GLM5_1-W8A8-EP-external_dp")
        assert _matches_filter(case, "GLM")


# ---------------------------------------------------------------------------
# Chip detection edge cases
# ---------------------------------------------------------------------------
class TestChipDetection:
    def test_a2_suffix(self):
        assert _detect_chip("tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml") == "a2"

    def test_a2_lowercase(self):
        assert _detect_chip("tests/e2e/schedule/model/Qwen/two_card/Qwen3.5-27B-w8a8-A2.yaml") == "a2"

    def test_a22b_not_a2(self):
        assert _detect_chip("tests/e2e/schedule/model/Qwen/two_node/Qwen3-235B-A22B.yaml") == "a3"

    def test_default_a3(self):
        assert _detect_chip("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml") == "a3"

    def test_explicit_a3_marker(self):
        assert _detect_chip("tests/e2e/schedule/model/Qwen/one_node/Qwen3.5-122B-A10B-W8A8-A3.yaml") == "a3"

    def test_310_token(self):
        assert _detect_chip("tests/e2e/schedule/ops/one_card/test_causal_conv1d_310.py") == "310p"

    def test_v310_token(self):
        assert _detect_chip("tests/e2e/schedule/ops/one_card/test_recurrent_gated_delta_rule_v310.py") == "310p"

    def test_310p_dir_token(self):
        assert _detect_chip("tests/e2e/schedule/ops/310p/one_card/test_foo.py") == "310p"

    def test_310_not_matched_inside_model_name(self):
        # "310B" inside a model name must not trigger 310p (followed by a letter).
        assert _detect_chip("tests/e2e/schedule/model/Qwen/one_card/Qwen3-310B.yaml") == "a3"


# ---------------------------------------------------------------------------
# 310p ops case routes with the 310p runner
# ---------------------------------------------------------------------------
class TestOps310p:
    def test_v310_ops_case(self):
        case = parse("tests/e2e/schedule/ops/one_card/test_recurrent_gated_delta_rule_v310.py")
        assert case.framework == "ops"
        assert case.route == "ops"
        assert case.chip == "310p"
        assert case.runner == "linux-aarch64-310p-1"

    def test_310p_matrix_item(self):
        case = parse("tests/e2e/schedule/ops/one_card/test_recurrent_gated_delta_rule_v310.py")
        item = matrix_item(case, "ops_matrix")
        assert item["chip"] == "310p"
        assert item["runner"] == "linux-aarch64-310p-1"


# ---------------------------------------------------------------------------
# Framework registry: every raw path matches exactly one framework
# ---------------------------------------------------------------------------
class TestFrameworkRegistry:
    CASES = [
        "tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml",
        "tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml",
        "tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml",
        "tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-A3-dual-nodes.yaml",
        "tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml",
        "tests/e2e/schedule/ops/one_card/test_fused_moe.py",
        "tests/e2e/schedule/ops/one_node/",
    ]

    def test_each_entry_matches_exactly_one_framework(self):
        frameworks = _build_frameworks(RUNNER_MAP)
        for path in self.CASES:
            matched = [fw for fw in frameworks if fw.match(path)]
            assert len(matched) == 1, f"Path {path!r} matched {len(matched)} frameworks: {[fw.name for fw in matched]}"
            for case in matched[0].expand(path):
                assert case.framework == matched[0].name

    def test_find_framework_rejects_unknown_path(self):
        with pytest.raises(ValueError, match="No framework matched"):
            _find_framework("tests/e2e/schedule/benchmark/one_card/foo.yaml", _build_frameworks(RUNNER_MAP))

    def test_find_framework_rejects_multiple_matches(self):
        class ShadowModelFramework(BaseFramework):
            name = "shadow_model"
            output_names = ("ops_matrix",)

            def match(self, path: str) -> bool:
                return path.startswith("tests/e2e/schedule/model/")

        frameworks = [ModelFramework(RUNNER_MAP), ShadowModelFramework(RUNNER_MAP)]
        with pytest.raises(ValueError, match="Multiple frameworks matched"):
            _find_framework("tests/e2e/schedule/model/Qwen/one_card/foo.yaml", frameworks)


# ---------------------------------------------------------------------------
# Directory entry expansion (per-chip groups for ops; per-file for model/accuracy)
# ---------------------------------------------------------------------------
class TestDirectoryExpansion:
    def test_ops_dir_expands_to_per_chip_groups(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/")
        assert all(c.framework == "ops" for c in cases)
        chips = {c.chip for c in cases}
        # one_card holds both default-a3 ops and 310p ops variants
        assert "a3" in chips
        assert "310p" in chips
        # one case per chip (no duplicate chip groups)
        assert len(chips) == len(cases)

    def test_310p_group_contains_v310_file(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/")
        p310 = next(c for c in cases if c.chip == "310p")
        assert any("test_recurrent_gated_delta_rule_v310.py" in path for path in p310.case_path)
        assert p310.runner == "linux-aarch64-310p-1"

    def test_a3_group_runner_and_multifile(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/")
        a3 = next(c for c in cases if c.chip == "a3")
        assert a3.runner == "linux-aarch64-a3-2"
        # a3 group bundles many files as a space-separated pytest target list
        assert len(a3.case_path) > 1

    def test_trailing_slash_and_bare_dir_equivalent(self):
        with_slash = parse_entry("tests/e2e/schedule/ops/one_card/")
        no_slash = parse_entry("tests/e2e/schedule/ops/one_card")
        assert {c.name for c in with_slash} == {c.name for c in no_slash}

    def test_single_file_entry_returns_one_case(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/test_fused_moe.py")
        assert len(cases) == 1
        assert cases[0].case_path == "tests/e2e/schedule/ops/one_card/test_fused_moe.py"
