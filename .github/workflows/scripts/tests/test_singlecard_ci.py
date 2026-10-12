# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and process isolation regressions; no accelerator imports."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import collect_merged_new_tests
import pytest
import run_singlecard_test
import test_selector
import yaml

ROOT = Path(__file__).resolve().parents[4]
OPS = "tests/e2e/nightly/single_node/ops/singlecard_ops"
SCRIPT = ROOT / ".github/workflows/scripts/select_tests.py"


def select(tmp_path, *args, cwd=ROOT):
    output = tmp_path / "output"
    output.write_text("")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        check=True,
    )
    fields = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert "Warning: E2E" not in result.stderr
    return json.loads(fields["test_groups"])


def assignments(groups):
    return {(target.replace("\\", "/"), group["npu_type"]) for group in groups for target in group["tests"].split()}


def test_complete_singlecard_inventory(tmp_path):
    groups = select(tmp_path, "--curated", "singlecard-ops")
    actual = assignments(groups)
    files = {path.relative_to(ROOT).as_posix() for path in (ROOT / OPS).rglob("test_*.py")}
    assert {path for path, _ in actual} == files
    assert all(group["num_npus"] == 1 for group in groups)
    assert {chip for _, chip in actual} == {"a2", "a3", "a5", "310p"}
    assert (f"{OPS}/test_attn_res_fwd.py", "a3") in actual
    assert (f"{OPS}/test_gmm_dequant_situ_quant_msd.py", "a3") in actual
    assert (f"{OPS}/test_causal_conv1d_310.py", "310p") in actual
    assert (f"{OPS}/test_recurrent_gated_delta_rule_310.py", "310p") in actual
    assert (f"{OPS}/test_qfa_c8_paged_layout.py", "a5") in actual
    assert (f"{OPS}/test_cann_qli_binding.py", "a2") not in actual
    assert {(f"{OPS}/triton/test_rope.py", chip) for chip in ("a2", "a5")} <= actual
    for name in ("kda_gate_cumsum_vllm", "kda_layout_swap12", "dequant_situ_quant"):
        assert {(f"{OPS}/ascendc/a2a3/test_{name}.py", chip) for chip in ("a2", "a3")} <= actual
    assert (f"{OPS}/ascendc/a5/test_situ_mx_quant.py", "a5") in actual
    assert (f"{OPS}/ascendc/a5/test_situ_mx_quant.py", "a2") not in actual
    assert {(f"{OPS}/triton/a2a3/test_copy_conv_state.py", chip) for chip in ("a2", "a3")} <= actual
    assert (f"{OPS}/triton/a5/test_copy_conv_state.py", "a5") in actual


def test_new_file_is_discovered_recursively(tmp_path):
    source = tmp_path / OPS / "triton/new_subdir/test_new_kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_new_kernel(): pass\n")
    groups = select(tmp_path, "--curated", "singlecard-ops", cwd=tmp_path)
    assert assignments(groups) == {(source.relative_to(tmp_path).as_posix(), "a2")}


@pytest.mark.parametrize(
    "directory, chips",
    [
        ("ascendc/a2a3", {"a2", "a3"}),
        ("triton/a2a3", {"a2", "a3"}),
        ("ascendc/a5", {"a5"}),
        ("triton/a5", {"a5"}),
        ("ascendc/310p", {"310p"}),
    ],
)
def test_architecture_directories_route_new_files(tmp_path, directory, chips):
    source = tmp_path / OPS / directory / "test_new_operator.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_new_operator(): pass\n")
    groups = select(tmp_path, "--curated", "singlecard-ops", cwd=tmp_path)
    assert assignments(groups) == {(source.relative_to(tmp_path).as_posix(), chip) for chip in chips}


def test_all_tests_contains_singlecard_and_existing_pr_tests(tmp_path):
    groups = select(tmp_path, "--all-tests", "--skip-cpu-ut")
    actual = assignments(groups)
    singlecard = assignments(select(tmp_path, "--curated", "singlecard-ops"))
    assert singlecard <= actual
    assert ("tests/e2e/pull_request/one_card/test_msa_index_score.py", "a2") in actual
    assert not any(path.startswith("tests/ut/") for path, _ in actual)


def test_nodeid_scope_and_override_deduplication(tmp_path):
    target = f"{OPS}/triton/test_rope.py::test_rotary_embedding_triton_kernel_fp8"
    groups = select(tmp_path, "--explicit-e2e-tests", target)
    assert assignments(groups) == {(target, "a2"), (target, "a5")}
    groups = select(tmp_path, "--explicit-e2e-tests", target, "--runner-label-override", "linux-aarch64-a5-4")
    assert assignments(groups) == {(target, "a5")}
    assert sum(len(group["tests"].split()) for group in groups) == 1
    assert groups[0]["num_npus"] == 1


def test_skip_is_not_resurrected_on_another_device(tmp_path):
    config_path = SCRIPT.with_name("test_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    target = f"{OPS}/triton/test_rope.py"
    config["skip_tests"].append(target)
    custom_config = tmp_path / "test_config.yaml"
    custom_config.write_text(yaml.safe_dump(config))
    groups = select(tmp_path, "--config", str(custom_config), "--explicit-e2e-tests", target)
    assert groups == []


def test_changed_and_merged_tests_are_recommended(tmp_path):
    target = f"{OPS}/triton/test_new_kernel.py"
    diff = tmp_path / "pr.diff"
    diff.write_text(f"--- /dev/null\n+++ b/{target}\n@@ -0,0 +1 @@\n+def test_new(): pass\n")
    assert test_selector._get_test_files_from_pr_diff(str(diff)) == [target]
    assert collect_merged_new_tests._is_test_file(target)
    assert not test_selector.is_selectable_test(f"{OPS}/benchmark_kernel.py")


@pytest.mark.parametrize("mask, expected", [(None, "0"), ("3", "3"), ("3,5", "3"), (" 2 , 7 ", "2")])
def test_visible_device_assignment_is_narrowed(mask, expected):
    assert run_singlecard_test.single_device_mask(mask) == expected


@pytest.mark.parametrize("mask", ["", "-1", "0,", "all", "0;echo bad"])
def test_invalid_assignment_fails_closed(mask):
    with pytest.raises(ValueError):
        run_singlecard_test.single_device_mask(mask)


@pytest.mark.parametrize("count", [0, 2])
def test_launcher_rejects_wrong_device_count(monkeypatch, count):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3,5")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(device_count=lambda: count)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    with pytest.raises(RuntimeError, match="exactly one visible NPU"):
        run_singlecard_test.main()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3"
