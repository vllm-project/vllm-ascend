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

import sys
from pathlib import Path

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from parse_schedule_config import (  # noqa: E402
    ROUTERS,
    AccuracyRouter,
    ModelMultiNodeRouter,
    ModelSingleNodeRouter,
    OpsRouter,
    PeriodicCase,
    _detect_chip,
    _detect_multi_node_type,
    _matches_filter,
    _parse_entry,
    _parse_to_case,
    _validate_accuracy_config,
)

# ---------------------------------------------------------------------------
# Minimal runner_map for tests (avoids needing runner_label.json on disk)
# ---------------------------------------------------------------------------
RUNNER_MAP: dict[tuple[str, int], str] = {
    ("a3", 0): "linux-aarch64-a3-0",
    ("a3", 1): "linux-aarch64-a3-1",
    ("a3", 2): "linux-aarch64-a3-2",
    ("a3", 4): "linux-aarch64-a3-4",
    ("a3", 8): "linux-aarch64-a3-8",
    ("a3", 16): "linux-aarch64-a3-16",
    ("a2", 0): "linux-amd64-cpu-8-hk",
    ("a2", 1): "linux-aarch64-a2b3-1",
    ("a2", 2): "linux-aarch64-a2b3-2",
    ("a2", 4): "linux-aarch64-a2b3-4",
    ("a2", 8): "linux-aarch64-a2b3-8",
    ("310p", 1): "linux-aarch64-310p-1",
    ("310p", 2): "linux-aarch64-310p-2",
    ("310p", 4): "linux-aarch64-310p-4",
}


def parse(path: str) -> PeriodicCase:
    return _parse_to_case(path, RUNNER_MAP)


def parse_entry(path: str) -> list[PeriodicCase]:
    return _parse_entry(path, RUNNER_MAP)


# Repo root, so directory tests resolve relative paths regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _list_yaml(rel_dir: str) -> list[Path]:
    return sorted((_REPO_ROOT / rel_dir).rglob("*.yaml"))


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
        assert case.family == "DeepSeek"

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

    def test_router_match(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        assert ModelSingleNodeRouter().match(case)
        assert not ModelMultiNodeRouter().match(case)

    def test_matrix_item(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml")
        item = ModelSingleNodeRouter().to_matrix_item(case)
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
# 3. model two_node external_dp -> multi_node + external
# ---------------------------------------------------------------------------
class TestModelTwoNodeExternalDp:
    def test_framework_and_route(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.framework == "model"
        assert case.route == "multi_node"

    def test_multi_node_type_external(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.multi_node_type == "external"

    def test_size(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert case.size == 2

    def test_router_match(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert ModelMultiNodeRouter().match(case)

    def test_matrix_item(self):
        case = parse("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        item = ModelMultiNodeRouter().to_matrix_item(case)
        assert item["multi_node_type"] == "external"
        assert item["size"] == 2


# ---------------------------------------------------------------------------
# 4. model two_node without external_dp -> multi_node + internal
# ---------------------------------------------------------------------------
class TestModelTwoNodeInternal:
    def test_multi_node_type_internal(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/two_node/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml")
        assert case.multi_node_type == "internal"

    def test_external_without_dp_is_internal(self):
        # Filename contains "external" but NOT "external_dp" -> internal
        type_ = _detect_multi_node_type("tests/e2e/schedule/model/GLM/two_node/GLM5-W8A8-external.yaml")
        assert type_ == "internal"

    def test_external_dp_in_stem_only(self):
        # "external_dp" must be in the file stem, not a subdirectory
        type_ = _detect_multi_node_type("tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml")
        assert type_ == "external"


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

    def test_router_match(self):
        case = parse(_ACC_FILE)
        assert AccuracyRouter().match(case)
        assert not ModelSingleNodeRouter().match(case)

    def test_config_paths_set_no_model_list(self):
        case = parse(_ACC_FILE)
        assert case.config_paths == [_ACC_FILE]
        assert case.config_path is None
        assert case.tests is None
        item = AccuracyRouter().to_matrix_item(case)
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
        assert len(case.config_paths) == len(_list_yaml(_ACC_A2_DIR))
        assert all(p.endswith(".yaml") for p in case.config_paths)

    def test_matrix_item_has_config_paths_no_model_list(self):
        case = parse_entry(_ACC_A2_DIR)[0]
        item = AccuracyRouter().to_matrix_item(case)
        assert "model_list" not in item
        assert isinstance(item["config_paths"], list) and item["config_paths"]

    def test_mixed_chip_dir_groups_per_chip(self, tmp_accuracy_mixed_dir):
        # eight_card/ holds a2/ and a3/ subdirs with dummy dict configs
        cases = parse_entry("tests/e2e/schedule/accuracy/eight_card/")
        chips = sorted(c.chip for c in cases)
        assert chips == ["a2", "a3"]
        for c in cases:
            assert c.framework == "accuracy"
            assert all(f"/{c.chip}/" in p for p in c.config_paths)


# ---------------------------------------------------------------------------
# 5c. Old model-list group YAML must be rejected (validation)
# ---------------------------------------------------------------------------
class TestAccuracyConfigValidation:
    def test_list_group_yaml_rejected(self, tmp_path):
        p = tmp_path / "accuracy-group-1-a2.yaml"
        p.write_text("- Qwen3-8B\n- Qwen2-Audio-7B-Instruct\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML dict"):
            _validate_accuracy_config(str(p))

    def test_dict_config_accepted(self, tmp_path):
        p = tmp_path / "Qwen3-8B.yaml"
        p.write_text("model_name: Qwen/Qwen3-8B\nmodel_type: vllm\n", encoding="utf-8")
        _validate_accuracy_config(str(p))  # no raise

    def test_missing_file_skipped(self, tmp_path):
        _validate_accuracy_config(str(tmp_path / "does-not-exist.yaml"))  # no raise


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
        assert case.tests == "tests/e2e/schedule/ops/one_card/test_fused_moe.py"

    def test_router_match(self):
        case = parse("tests/e2e/schedule/ops/one_card/test_fused_moe.py")
        assert OpsRouter().match(case)
        assert not ModelSingleNodeRouter().match(case)

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
# 7. Numeric resource directory -> error
# ---------------------------------------------------------------------------
class TestNumericResourceError:
    def test_1_card_fails(self):
        with pytest.raises(ValueError, match="Numeric resource directory"):
            parse("tests/e2e/schedule/model/DeepSeek/1_card/DeepSeek-V3.yaml")

    def test_2_node_fails(self):
        with pytest.raises(ValueError, match="Numeric resource directory"):
            parse("tests/e2e/schedule/model/Qwen/2_node/Qwen3.yaml")

    def test_4_card_fails(self):
        with pytest.raises(ValueError, match="Numeric resource directory"):
            parse("tests/e2e/schedule/ops/4_card/test_foo.py")

    def test_8_card_fails(self):
        with pytest.raises(ValueError, match="Numeric resource directory"):
            parse("tests/e2e/schedule/model/Qwen/8_card/Qwen3.yaml")


# ---------------------------------------------------------------------------
# 8. tests/e2e/accuracy path -> error
# ---------------------------------------------------------------------------
class TestOldAccuracyPathError:
    def test_old_accuracy_path_fails(self):
        with pytest.raises(ValueError, match="tests/e2e/accuracy"):
            parse("tests/e2e/accuracy/one_card/accuracy-group-1-a2.yaml")

    def test_non_schedule_path_fails(self):
        with pytest.raises(ValueError, match="tests/e2e/schedule"):
            parse("tests/e2e/pull_request/one_card/test_foo.yaml")


# ---------------------------------------------------------------------------
# 9. Path outside tests/e2e/schedule -> error
# ---------------------------------------------------------------------------
class TestInvalidPathError:
    def test_absolute_like_path_fails(self):
        with pytest.raises(ValueError):
            parse("some/other/path/one_card/test.yaml")

    def test_missing_resource_dir_fails(self):
        with pytest.raises(ValueError, match="No resource directory"):
            parse("tests/e2e/schedule/model/DeepSeek/DeepSeek-V3.yaml")

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
        assert case.family == "Kimi"
        assert case.route == "single_node"
        assert case.resource_dir == "one_node"

    def test_family_is_segment_after_model(self):
        case = parse("tests/e2e/schedule/model/DeepSeek/two_node/DeepSeek-V3.1-BF16.yaml")
        assert case.family == "DeepSeek"
        assert case.route == "multi_node"

    def test_bare_family_without_model_layer_fails(self):
        # Old layout (no model/ layer) is no longer accepted.
        with pytest.raises(ValueError, match="Unknown framework directory"):
            parse("tests/e2e/schedule/Kimi/one_node/Kimi-K2.5.yaml")

    def test_model_without_family_fails(self):
        with pytest.raises(ValueError, match="must include a family"):
            parse("tests/e2e/schedule/model/one_node/Kimi-K2.5.yaml")


# ---------------------------------------------------------------------------
# 10. test_filter matching
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
        item = OpsRouter().to_matrix_item(case)
        assert item["chip"] == "310p"
        assert item["runner"] == "linux-aarch64-310p-1"


# ---------------------------------------------------------------------------
# Router registry: every case matches exactly one router
# ---------------------------------------------------------------------------
class TestRouterRegistry:
    CASES = [
        "tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.2-W8A8.yaml",
        "tests/e2e/schedule/model/Qwen/four_card/Qwen3-32B-Int8-A2.yaml",
        "tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml",
        "tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-A3-dual-nodes.yaml",
        "tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml",
        "tests/e2e/schedule/ops/one_card/test_fused_moe.py",
        "tests/e2e/schedule/ops/one_node/",
    ]

    def test_each_case_matches_exactly_one_router(self):
        for path in self.CASES:
            for case in parse_entry(path):
                matched = [r for r in ROUTERS if r.match(case)]
                assert len(matched) == 1, (
                    f"Path {path!r} (case {case.name!r}) matched {len(matched)} routers: {[r.name for r in matched]}"
                )


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
        assert "test_recurrent_gated_delta_rule_v310.py" in p310.tests
        assert p310.runner == "linux-aarch64-310p-1"

    def test_a3_group_runner_and_multifile(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/")
        a3 = next(c for c in cases if c.chip == "a3")
        assert a3.runner == "linux-aarch64-a3-1"
        # a3 group bundles many files as a space-separated pytest target list
        assert len(a3.tests.split()) > 1

    def test_trailing_slash_and_bare_dir_equivalent(self):
        with_slash = parse_entry("tests/e2e/schedule/ops/one_card/")
        no_slash = parse_entry("tests/e2e/schedule/ops/one_card")
        assert {c.name for c in with_slash} == {c.name for c in no_slash}

    def test_single_file_entry_returns_one_case(self):
        cases = parse_entry("tests/e2e/schedule/ops/one_card/test_fused_moe.py")
        assert len(cases) == 1
        assert cases[0].tests == "tests/e2e/schedule/ops/one_card/test_fused_moe.py"
