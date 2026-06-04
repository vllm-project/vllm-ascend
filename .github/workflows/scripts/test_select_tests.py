from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections import defaultdict

import pytest
import regex as re
import yaml

sys.modules.setdefault("regex", re)
select_tests = importlib.import_module("select_tests")


def test_match_modules_include_exclude_and_required_modules():
    config = [
        {"name": "required", "optional": False},
        {"name": "file", "source_file_dependencies": ["src/a.py"]},
        {"name": "dir", "source_file_dependencies": ["src/pkg/"]},
        {
            "name": "dir_except_file",
            "source_file_dependencies": ["src/pkg"],
            "exclude_source_file_dependencies": ["src/pkg/__init__.py"],
        },
        {
            "name": "dir_except_subdir",
            "source_file_dependencies": ["src/pkg"],
            "exclude_source_file_dependencies": ["src/pkg/skip"],
        },
    ]

    assert select_tests._match_modules([], config) == []
    assert select_tests._match_modules(["src/a.py"], config) == ["required", "file"]
    assert select_tests._match_modules(["src/pkg/b.py"], config) == [
        "required",
        "dir",
        "dir_except_file",
        "dir_except_subdir",
    ]
    assert select_tests._match_modules(["src/pkg/__init__.py"], config) == [
        "required",
        "dir",
        "dir_except_subdir",
    ]
    assert select_tests._match_modules(["src/pkg/skip/b.py"], config) == [
        "required",
        "dir",
        "dir_except_file",
    ]
    assert select_tests._match_modules(["src/pkg_extra/b.py"], config) == ["required"]


def test_resolve_config_inheritance_merges_base_modules():
    config = [
        {
            "name": "base_a",
            "source_file_dependencies": ["src/base_a.py"],
            "exclude_source_file_dependencies": ["src/base_a_skip.py"],
            "tests": ["tests/ut/base_a"],
            "skip_tests": ["tests/ut/base_a/test_skip.py"],
        },
        {
            "name": "base_b",
            "source_file_dependencies": ["src/base_b.py"],
            "tests": ["tests/ut/base_b"],
        },
        {
            "name": "child",
            "base": ["base_a", "base_b"],
            "source_file_dependencies": ["src/child.py", "src/base_a.py"],
            "tests": ["tests/ut/child", "tests/ut/base_a"],
        },
    ]

    resolved = select_tests._resolve_config_inheritance(config)
    child = {module["name"]: module for module in resolved}["child"]
    assert child["source_file_dependencies"] == ["src/base_a.py", "src/base_b.py", "src/child.py"]
    assert child["exclude_source_file_dependencies"] == ["src/base_a_skip.py"]
    assert child["tests"] == ["tests/ut/base_a", "tests/ut/base_b", "tests/ut/child"]
    assert child["skip_tests"] == ["tests/ut/base_a/test_skip.py"]

    with pytest.raises(ValueError, match="Unknown base module"):
        select_tests._resolve_config_inheritance([{"name": "child", "base": "missing"}])
    with pytest.raises(ValueError, match="Circular test config inheritance"):
        select_tests._resolve_config_inheritance(
            [
                {"name": "a", "base": "b"},
                {"name": "b", "base": "a"},
            ]
        )


def test_collect_paths_and_basic_path_helpers():
    config = [
        {"name": "a", "tests": ["tests/ut/a/", "tests/ut/a/test_x.py", "tests/e2e/x"]},
        {"name": "b", "tests": ["tests/ut/b/test_y.py", "tests/ut/b/test_y.py"]},
    ]

    assert select_tests._matches_path_dependency("pkg/a.py", "pkg")
    assert select_tests._matches_path_dependency("pkg/a.py", "pkg/")
    assert not select_tests._matches_path_dependency("pkg_extra/a.py", "pkg")
    assert select_tests._collect_test_dirs(["a", "b"], config) == [
        "tests/e2e/x",
        "tests/ut/a",
        "tests/ut/b/test_y.py",
    ]
    assert select_tests._is_ut_path("tests/ut/a.py")
    assert select_tests._is_e2e_path("tests/e2e/a.py")


def test_route_helpers_and_e2e_type_filters():
    assert select_tests._route_ut_dir("tests/ut/mod/a2_2/test_x.py") == (2, select_tests.NpuType.A2)
    assert select_tests._route_ut_dir("tests/ut/mod/a2/test_x.py") == (1, select_tests.NpuType.A2)
    assert select_tests._route_ut_dir("tests/ut/mod/a3_4/test_x.py") == (4, select_tests.NpuType.A3)
    assert select_tests._route_ut_dir("tests/ut/mod/a3_2/test_x.py") == (2, select_tests.NpuType.A3)
    assert select_tests._route_ut_dir("tests/ut/mod/310p/test_x.py") == (1, select_tests.NpuType._310P)
    assert select_tests._route_ut_dir("tests/ut/_310p/test_x.py") == select_tests._DEFAULT_KEY
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/full/four_cards/") == (4, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/full/four_card/") == (4, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/full/two_cards/") == (2, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/full/two_card/") == (2, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/full/one_card/") == (1, select_tests.NpuType.A2)
    assert select_tests._route_e2e_dir("tests/e2e/other/") is None
    assert select_tests._route_e2e_file("tests/e2e/pull_request/full/four_cards/test_x_310p.py") == (
        4,
        select_tests.NpuType._310P,
    )
    assert select_tests._route_e2e_file("tests/e2e/pull_request/full/one_card/test_x_310p.py") == (
        1,
        select_tests.NpuType._310P,
    )
    assert select_tests._matches_e2e_type("tests/e2e/pull_request/light/one_card/test_x.py", "light")
    assert not select_tests._matches_e2e_type("tests/e2e/pull_request/light/one_card/test_x.py", "full")
    assert select_tests._matches_e2e_type("tests/e2e/310p/singlecard/test_x.py", "full")


def test_scan_ut_test_dir(tmp_path):
    groups = defaultdict(list)
    missing = tmp_path / "missing"
    select_tests._scan_ut_test_dir(str(missing), groups)
    assert groups[select_tests._DEFAULT_KEY] == [str(missing)]

    test_file = tmp_path / "tests" / "ut" / "mod" / "test_file.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    groups = defaultdict(list)
    select_tests._scan_ut_test_dir(str(test_file), groups)
    assert groups[select_tests._DEFAULT_KEY] == [str(test_file)]

    mixed = tmp_path / "tests" / "ut" / "mixed"
    nested = mixed / "nested"
    a3 = mixed / "a3_2"
    nested.mkdir(parents=True)
    a3.mkdir(parents=True)
    (mixed / "__init__.py").write_text("")
    (mixed / "helper.py").write_text("")
    (mixed / "test_root.py").write_text("")
    (nested / "test_nested.py").write_text("")
    (a3 / "test_a3.py").write_text("")
    groups = defaultdict(list)
    select_tests._scan_ut_test_dir(str(mixed), groups)
    assert sorted(groups[select_tests._DEFAULT_KEY]) == sorted(
        [str(mixed / "test_root.py"), str(nested / "test_nested.py")]
    )
    assert groups[(2, select_tests.NpuType.A3)] == [str(a3 / "test_a3.py")]

    groups = defaultdict(list)
    select_tests._scan_ut_test_dir(str(a3), groups)
    assert groups[(2, select_tests.NpuType.A3)] == [str(a3 / "test_a3.py")]


def test_scan_e2e_test_dir(tmp_path, capsys):
    groups = defaultdict(list)
    select_tests._scan_e2e_test_dir(str(tmp_path / "missing"), groups)
    assert groups == {}

    unrouted = tmp_path / "tests" / "e2e" / "test_unrouted.py"
    unrouted.parent.mkdir(parents=True)
    unrouted.write_text("")
    select_tests._scan_e2e_test_dir(str(unrouted), groups)
    assert "does not match any runner pattern" in capsys.readouterr().err

    one_card = tmp_path / "tests" / "e2e" / "pull_request" / "full" / "one_card"
    one_card.mkdir(parents=True)
    test_one = one_card / "test_one.py"
    test_310p = one_card / "test_one_310p.py"
    helper = one_card / "helper.py"
    test_one.write_text("")
    test_310p.write_text("")
    helper.write_text("")
    select_tests._scan_e2e_test_dir(str(one_card), groups)
    assert str(test_one) in groups[(1, select_tests.NpuType.A2)]
    assert str(test_310p) in groups[(1, select_tests.NpuType._310P)]
    assert str(helper) not in groups[(1, select_tests.NpuType.A2)]

    parent = tmp_path / "tests" / "e2e" / "pull_request" / "light"
    two_card = parent / "two_card"
    two_card.mkdir(parents=True)
    test_two = two_card / "test_two.py"
    test_two.write_text("")
    select_tests._scan_e2e_test_dir(str(parent), groups)
    assert str(test_two) in groups[(2, select_tests.NpuType.A3)]


def test_dedup_runner_resolution_and_output(tmp_path, monkeypatch, capsys):
    groups = defaultdict(list)
    groups[select_tests._DEFAULT_KEY].extend(["b", "a", "b"])
    select_tests._dedup_groups(groups)
    assert groups[select_tests._DEFAULT_KEY] == ["b", "a"]

    runner_file = tmp_path / "runner_label.json"
    runner_file.write_text(
        json.dumps(
            {
                "cpu-runner": {"chip": "cpu", "npu_num": 0, "image_tag": "cpu-img"},
                "a2-runner": {"chip": "a2", "npu_num": 1, "image_tag": "a2-img"},
            }
        )
    )
    monkeypatch.setattr(select_tests, "_RUNNER_LABEL_PATH", runner_file)
    runners = select_tests._load_runners()
    assert select_tests._find_runner(0, select_tests.NpuType.CPU, runners).label == "cpu-runner"
    assert select_tests._find_runner(1, select_tests.NpuType.A2, runners).label == "a2-runner"
    assert select_tests._find_runner(2, select_tests.NpuType.A2, runners) is None
    assert select_tests._resolve_to_runners(
        {select_tests._DEFAULT_KEY: ["tests/ut/b.py", "tests/ut/a.py"], (1, select_tests.NpuType.A2): ["e2e.py"]},
        runners,
    ) == [
        {
            "num_npus": 0,
            "npu_type": "cpu",
            "runner": "cpu-runner",
            "tests": "tests/ut/a.py tests/ut/b.py",
            "image_tag": "cpu-img",
        },
        {"num_npus": 1, "npu_type": "a2", "runner": "a2-runner", "tests": "e2e.py", "image_tag": "a2-img"},
    ]
    with pytest.raises(SystemExit):
        select_tests._resolve_to_runners({(2, select_tests.NpuType.A2): ["x"]}, runners)
    assert "no runner available" in capsys.readouterr().err

    test_groups = [{"num_npus": 0, "npu_type": "cpu", "runner": "cpu-runner", "tests": "a b"}]
    select_tests._write_output(test_groups, ["m1", "m2"])
    assert "has_tests=true" in capsys.readouterr().out
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    select_tests._write_output([], [])
    assert "has_tests=false" in output.read_text()


def test_get_changed_files(monkeypatch):
    def fake_run(args, capture_output, text, check):
        assert args == ["git", "diff", "--name-only", "origin/main...HEAD"]
        assert capture_output and text and check
        return subprocess.CompletedProcess(args, 0, "a.py\n\nb.py\n", "")

    monkeypatch.setattr(select_tests.subprocess, "run", fake_run)
    assert select_tests._get_changed_files("origin/main") == ["a.py", "b.py"]


def test_main_end_to_end_changed_files_options_and_skip(tmp_path, monkeypatch, capsys):
    test_root = tmp_path / "tests"
    cpu_dir = test_root / "ut" / "cpu"
    a2_dir = test_root / "ut" / "npu" / "a2"
    e2e_light = test_root / "e2e" / "pull_request" / "light" / "one_card"
    e2e_full = test_root / "e2e" / "pull_request" / "full" / "two_cards"
    for path in (cpu_dir, a2_dir, e2e_light, e2e_full):
        path.mkdir(parents=True)
    cpu_keep = cpu_dir / "test_keep.py"
    cpu_skip = cpu_dir / "test_skip.py"
    changed_test = cpu_dir / "test_changed.py"
    a2_test = a2_dir / "test_a2.py"
    light_test = e2e_light / "test_light.py"
    full_test = e2e_full / "test_full.py"
    for path in (cpu_keep, cpu_skip, changed_test, a2_test, light_test, full_test):
        path.write_text("")

    config = [
        {
            "name": "cpu",
            "source_file_dependencies": ["src/cpu.py"],
            "tests": ["tests/ut/cpu", "tests/ut/missing/test_missing.py"],
            "skip_tests": ["tests/ut/cpu/test_skip.py"],
        },
        {
            "name": "npu",
            "base": "cpu",
            "source_file_dependencies": ["src/npu.py"],
            "tests": ["tests/ut/npu/a2"],
        },
        {
            "name": "e2e",
            "source_file_dependencies": ["src/e2e.py"],
            "tests": [
                "tests/e2e/pull_request/light/one_card",
                "tests/e2e/pull_request/full/two_cards",
            ],
        },
    ]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    runner_file = tmp_path / "runner_label.json"
    runner_file.write_text(
        json.dumps(
            {
                "cpu-runner": {"chip": "cpu", "npu_num": 0},
                "a2-runner": {"chip": "a2", "npu_num": 1},
                "a3-runner": {"chip": "a3", "npu_num": 2},
            }
        )
    )
    monkeypatch.setattr(select_tests, "_RUNNER_LABEL_PATH", runner_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_tests.py",
            "--config",
            str(config_path),
            "--changed-files",
            "src/cpu.py",
            "src/npu.py",
            "src/e2e.py",
            "tests/ut/cpu/test_changed.py",
            "--e2e-type",
            "light",
        ],
    )

    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=cpu,npu,e2e" in out
    assert "tests/ut/cpu/test_keep.py" in out
    assert "tests/ut/cpu/test_skip.py" not in out
    assert "tests/ut/missing/test_missing.py" in out
    assert "tests/ut/npu/a2/test_a2.py" in out
    assert "tests/e2e/pull_request/light/one_card/test_light.py" in out
    assert "tests/e2e/pull_request/full/two_cards/test_full.py" not in out
    assert "tests/ut/cpu/test_changed.py" in out

    monkeypatch.setattr(
        sys,
        "argv",
        ["select_tests.py", "--config", str(config_path), "--changed-files", "src/cpu.py", "--run-all-cpu"],
    )
    select_tests.main()
    assert "tests/ut/cpu/test_keep.py" in capsys.readouterr().out
