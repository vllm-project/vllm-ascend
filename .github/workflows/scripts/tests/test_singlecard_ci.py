# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and process isolation regressions; no accelerator imports."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import collect_merged_new_tests
import pytest
import run_singlecard_test
import select_tests
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
    assert (f"{OPS}/ascendc/a2a3/test_attn_res_fwd.py", "a3") in actual
    assert (f"{OPS}/ascendc/a2a3/test_attn_res_fwd.py", "a2") not in actual
    assert (f"{OPS}/ascendc/a2a3/test_gmm_dequant_situ_quant_msd.py", "a3") in actual
    for name in ("copy_sfa_offload_metadata", "lim_mtp_index_reuse", "sparse_kv_offload"):
        target = f"{OPS}/ascendc/a2a3/test_{name}.py"
        assert (target, "a3") in actual
        assert (target, "a2") not in actual
    assert (f"{OPS}/test_causal_conv1d_310.py", "310p") in actual
    assert (f"{OPS}/test_recurrent_gated_delta_rule_310.py", "310p") in actual
    assert (f"{OPS}/ascendc/a5/test_qfa_c8_paged_layout.py", "a5") in actual
    assert {(f"{OPS}/triton/{arch}/test_rope.py", chip) for arch, chip in (("a2a3", "a2"), ("a5", "a5"))} <= actual
    for name in ("kda_gate_cumsum_vllm", "kda_layout_swap12", "dequant_situ_quant"):
        assert {(f"{OPS}/ascendc/a2a3/test_{name}.py", chip) for chip in ("a2", "a3")} <= actual
    assert (f"{OPS}/ascendc/a5/test_situ_mx_quant.py", "a5") in actual
    assert (f"{OPS}/ascendc/a5/test_situ_mx_quant.py", "a2") not in actual
    assert {(f"{OPS}/triton/a2a3/test_copy_conv_state.py", chip) for chip in ("a2", "a3")} <= actual
    assert (f"{OPS}/triton/a5/test_copy_conv_state.py", "a5") in actual


def test_all_operator_tests_have_backend_and_architecture():
    legacy = {"test_causal_conv1d_310.py", "test_recurrent_gated_delta_rule_310.py"}
    root = ROOT / OPS
    for source in root.rglob("test_*.py"):
        relative = source.relative_to(root)
        if relative.as_posix() in legacy:
            continue
        assert len(relative.parts) == 3, relative
        assert relative.parts[0] in {"ascendc", "triton"}, relative
        assert relative.parts[1] in {"a2a3", "a5"}, relative
    assert not list(root.rglob(".gitkeep"))


def test_a5_cases_are_separated_from_a2a3():
    def names(relative):
        tree = ast.parse((ROOT / OPS / relative).read_text())
        return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}

    fp8 = "test_rotary_embedding_triton_kernel_fp8"
    assert fp8 not in names("triton/a2a3/test_rope.py")
    assert fp8 in names("triton/a5/test_rope.py")
    assert not any(name.startswith("test_a5_") for name in names("triton/a2a3/test_dcp_a2a.py"))
    assert any(name.startswith("test_a5_") for name in names("triton/a5/test_dcp_a2a.py"))
    assert "test_a5_cache_destination_layout" not in names("ascendc/a2a3/test_scatter_nd_update_sk.py")
    assert "test_a5_cache_destination_layout" in names("ascendc/a5/test_scatter_nd_update_sk.py")


def test_external_operator_only_files_are_removed():
    removed = {
        "test_cann_qli_binding.py",
        "test_gating_top_k_softmax.py",
        "test_pa_kv_cache_ops.py",
        "test_chunk_kda_aclnn.py",
        "test_kimi_k3_chunk_kda_tail_npu.py",
        "test_kimi_kda_ascendc_npu.py",
        "test_kimi_kda_recurrent_ascendc_npu.py",
    }
    assert not {source.name for source in (ROOT / OPS).rglob("test_*.py")} & removed


def test_new_file_is_discovered_recursively(tmp_path):
    source = tmp_path / OPS / "triton/a2a3/new_subdir/test_new_kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_new_kernel(): pass\n")
    groups = select(tmp_path, "--curated", "singlecard-ops", cwd=tmp_path)
    assert assignments(groups) == {(source.relative_to(tmp_path).as_posix(), chip) for chip in ("a2", "a3")}


@pytest.mark.parametrize(
    "directory, chips",
    [
        ("ascendc/a2a3", {"a2", "a3"}),
        ("triton/a2a3", {"a2", "a3"}),
        ("ascendc/a5", {"a5"}),
        ("triton/a5", {"a5"}),
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
    target = f"{OPS}/triton/a2a3/test_rope.py::test_rotary_embedding_triton_kernel"
    groups = select(tmp_path, "--explicit-e2e-tests", target)
    assert assignments(groups) == {(target, "a2"), (target, "a3")}
    groups = select(tmp_path, "--explicit-e2e-tests", target, "--runner-label-override", "linux-aarch64-a5-4")
    assert assignments(groups) == {(target, "a5")}
    assert sum(len(group["tests"].split()) for group in groups) == 1
    assert groups[0]["num_npus"] == 1


@pytest.mark.parametrize("mode", ["explicit", "recommended"])
@pytest.mark.parametrize("override", [False, True])
def test_file_and_nodeid_run_once_per_device(tmp_path, mode, override):
    target = f"{OPS}/ascendc/a2a3/test_kda_layout_swap12.py"
    targets = [target + "::test_kda_layout_swap12_output_metadata", target, target]
    if mode == "explicit":
        args = ["--explicit-e2e-tests", *targets]
    else:
        test_list = tmp_path / "recommended.txt"
        test_list.write_text("\n".join(targets))
        args = ["--test-list-file", str(test_list)]
    if override:
        args.extend(["--runner-label-override", "linux-aarch64-a5-4"])
    groups = select(tmp_path, *args)
    chips = {"a5"} if override else {"a2", "a3"}
    assert assignments(groups) == {(target, chip) for chip in chips}
    assert sum(len(group["tests"].split()) for group in groups) == len(chips)


def test_dedup_preserves_other_partition_nodeids():
    target = "tests/e2e/example/test_operator.py"
    nodeid = target + "::test_case"
    groups = {"a2": [nodeid, target, target], "a3": [nodeid, nodeid]}
    select_tests._dedup_groups(groups)
    assert groups == {"a2": [target], "a3": [nodeid]}


def test_legacy_routes_and_310p_override(tmp_path):
    root = "tests/e2e/pull_request/one_card"
    expected = {(f"{root}/test_operator.py", "a2"), (f"{root}/test_operator_310p.py", "310p")}
    for target, _ in expected:
        source = tmp_path / target
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def test_case(): pass\n")
    groups = select(tmp_path, "--explicit-e2e-tests", root, cwd=tmp_path)
    assert assignments(groups) == expected


def test_specific_route_replaces_directory_devices(tmp_path):
    target = f"{OPS}/ascendc/a2a3/test_kda_layout_swap12.py"
    config = yaml.safe_load(SCRIPT.with_name("test_config.yaml").read_text())
    config["runner_mapping"][target.replace(".", r"\.") + "$"] = {"default": ["ops-a5-1"]}
    custom_config = tmp_path / "test_config.yaml"
    custom_config.write_text(yaml.safe_dump(config))
    groups = select(tmp_path, "--config", str(custom_config), "--explicit-e2e-tests", target)
    assert assignments(groups) == {(target, "a5")}


@pytest.mark.parametrize("value", [[], ["ops-a2-1", 3], None])
def test_invalid_partition_lists_are_rejected(monkeypatch, value):
    monkeypatch.setattr(select_tests, "_RUNNER_MAPPING", [])
    with pytest.raises(ValueError, match="must name a partition or a non-empty list"):
        select_tests._load_runner_mapping({"runner_mapping": {"tests/": {"default": value}}})


def test_unknown_partition_in_list_is_rejected(monkeypatch):
    monkeypatch.setattr(select_tests, "_RUNNER_MAPPING", [])
    config = yaml.safe_load(SCRIPT.with_name("test_config.yaml").read_text())
    config["runner_mapping"]["tests/example/"] = {"default": ["ops-a2-1", "missing-partition"]}
    select_tests._load_runner_mapping(config)
    with pytest.raises(ValueError, match="undefined logical partition.*missing-partition"):
        select_tests._validate_runner_config(select_tests._load_runners(), select_tests._load_partition_config(config))


@pytest.mark.parametrize("suffix", ["", "::test_rotary_embedding_triton_kernel_fp8"])
def test_skipped_target_is_removed_from_all_devices(tmp_path, suffix):
    config_path = SCRIPT.with_name("test_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    target = f"{OPS}/triton/a5/test_rope.py{suffix}"
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
