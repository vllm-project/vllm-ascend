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
    nodeid_config = [
        {
            "name": "nodeid",
            "tests": [
                "tests/e2e/pull_request/two_card/test_split.py::test_a",
                "tests/e2e/pull_request/two_card/test_split.py::test_b",
                "tests/e2e/pull_request/two_card/test_split.py::test_a",
            ],
        },
    ]

    assert select_tests._matches_path_dependency("pkg/a.py", "pkg")
    assert select_tests._matches_path_dependency("pkg/a.py", "pkg/")
    assert not select_tests._matches_path_dependency("pkg_extra/a.py", "pkg")
    assert select_tests._collect_test_dirs(["a", "b"], config) == (
        ["tests/e2e/x", "tests/ut/a", "tests/ut/b/test_y.py"],
        [],
    )
    assert select_tests._is_ut_path("tests/ut")
    assert select_tests._is_ut_path("tests/ut/a.py")
    assert select_tests._is_e2e_path("tests/e2e")
    assert select_tests._is_e2e_path("tests/e2e/a.py")
    assert select_tests._configured_nodeid_targets_for_file(
        "tests/e2e/pull_request/two_card/test_split.py",
        nodeid_config,
    ) == [
        "tests/e2e/pull_request/two_card/test_split.py::test_a",
        "tests/e2e/pull_request/two_card/test_split.py::test_b",
    ]
    assert select_tests._is_skipped_test_target(
        "tests/e2e/pull_request/two_card/test_split.py::test_a",
        {"tests/e2e/pull_request/two_card/test_split.py"},
    )
    assert select_tests._is_skipped_test_target(
        "tests/e2e/pull_request/two_card/test_split.py::test_a",
        {"tests/e2e/pull_request/two_card/test_split.py::test_a"},
    )


def test_route_helpers():
    # Use a controlled runner_mapping that covers all convention directories
    # documented in test_config.yaml (a2_2, a2, a3_4, a3_2, 310p). The real
    # config only defines a subset because the corresponding test directories
    # don't exist in the repo today, but the routing logic supports all of them.
    select_tests._load_runner_mapping(
        {
            "runner_mapping": {
                "tests/ut/.+/a2_2": {"default": "a2_x2"},
                "tests/ut/.+/a2": {"default": "a2_x1"},
                "tests/ut/.+/a3_4": {"default": "a3_x4"},
                "tests/ut/.+/a3_2": {"default": "a3_x2"},
                "tests/ut/.+/310p": {"default": "310p_x1"},
                "tests/e2e/pull_request/one_card": {"default": "a2_x1", "310p": "310p_x1"},
                "tests/e2e/pull_request/two_card": {"default": "a3_x2"},
                "tests/e2e/pull_request/four_card": {"default": "a3_x4", "310p": "310p_x4"},
            }
        }
    )

    assert select_tests._pytest_node_file_path("tests/e2e/test_x.py::TestCase::test_a") == "tests/e2e/test_x.py"
    assert select_tests._route_ut_dir("tests/ut/mod/a2_2/test_x.py") == (2, select_tests.NpuType.A2)
    assert select_tests._route_ut_dir("tests/ut/mod/a2_2/test_x.py::test_case") == (2, select_tests.NpuType.A2)
    assert select_tests._route_ut_dir("tests/ut/mod/a2/test_x.py") == (1, select_tests.NpuType.A2)
    assert select_tests._route_ut_dir("tests/ut/mod/a3_4/test_x.py") == (4, select_tests.NpuType.A3)
    assert select_tests._route_ut_dir("tests/ut/mod/a3_2/test_x.py") == (2, select_tests.NpuType.A3)
    assert select_tests._route_ut_dir("tests/ut/mod/310p/test_x.py") == (1, select_tests.NpuType._310P)
    assert select_tests._route_ut_dir("tests/ut/_310p/test_x.py") == select_tests._DEFAULT_KEY
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/four_card/") == (4, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/two_card/") == (2, select_tests.NpuType.A3)
    assert select_tests._route_e2e_dir("tests/e2e/pull_request/one_card/") == (1, select_tests.NpuType.A2)
    assert select_tests._route_e2e_dir("tests/e2e/other/") is None
    assert select_tests._route_e2e_file("tests/e2e/pull_request/four_card/test_x_310p.py") == (
        4,
        select_tests.NpuType._310P,
    )
    assert select_tests._route_e2e_file("tests/e2e/pull_request/one_card/test_x_310p.py") == (
        1,
        select_tests.NpuType._310P,
    )
    assert select_tests._route_e2e_file("tests/e2e/pull_request/two_card/test_x.py::test_case") == (
        2,
        select_tests.NpuType.A3,
    )


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

    one_card = tmp_path / "tests" / "e2e" / "pull_request" / "one_card"
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

    nodeid = f"{test_one}::test_specific_case"
    select_tests._scan_e2e_test_dir(nodeid, groups)
    assert nodeid in groups[(1, select_tests.NpuType.A2)]

    parent = tmp_path / "tests" / "e2e" / "pull_request"
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
            "partition": "1-1",
        },
        {
            "num_npus": 1,
            "npu_type": "a2",
            "runner": "a2-runner",
            "tests": "e2e.py",
            "image_tag": "a2-img",
            "partition": "1-1",
        },
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
    e2e_one_card = test_root / "e2e" / "pull_request" / "one_card"
    e2e_two_card = test_root / "e2e" / "pull_request" / "two_card"
    for path in (cpu_dir, a2_dir, e2e_one_card, e2e_two_card):
        path.mkdir(parents=True)
    cpu_keep = cpu_dir / "test_keep.py"
    cpu_skip = cpu_dir / "test_skip.py"
    changed_test = cpu_dir / "test_changed.py"
    a2_test = a2_dir / "test_a2.py"
    one_card_test = e2e_one_card / "test_one_card.py"
    two_card_test = e2e_two_card / "test_two_card.py"
    for path in (cpu_keep, cpu_skip, changed_test, a2_test, one_card_test, two_card_test):
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
                "tests/e2e/pull_request/one_card",
                "tests/e2e/pull_request/two_card",
            ],
        },
        {
            "name": "nodeid",
            "source_file_dependencies": ["src/nodeid.py"],
            "tests": [
                "tests/e2e/pull_request/two_card/test_two_card.py::test_specific_case",
                "tests/e2e/pull_request/two_card/test_two_card.py::test_other_case",
            ],
            "skip_tests": ["tests/e2e/pull_request/two_card/test_two_card.py::test_other_case"],
        },
    ]
    config_path = tmp_path / "config.yaml"
    runner_mapping = {
        "tests/ut/.+/a2": {"default": "a2_x1"},
        "tests/e2e/pull_request/one_card": {"default": "a2_x1", "310p": "310p_x1"},
        "tests/e2e/pull_request/two_card": {"default": "a3_x2"},
    }
    _write_two_doc_config(config_path, config, {"runner_mapping": runner_mapping})
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
        ],
    )

    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=cpu,npu,e2e" in out
    assert "tests/ut/cpu/test_keep.py" in out
    assert "tests/ut/cpu/test_skip.py" not in out
    assert "tests/ut/missing/test_missing.py" in out
    assert "tests/ut/npu/a2/test_a2.py" in out
    assert "tests/e2e/pull_request/one_card/test_one_card.py" in out
    assert "tests/e2e/pull_request/two_card/test_two_card.py" in out
    assert "tests/ut/cpu/test_changed.py" in out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_tests.py",
            "--config",
            str(config_path),
            "--changed-files",
            "src/cpu.py",
            "--run-all-modules",
        ],
    )
    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=cpu,npu,e2e" in out
    assert "tests/ut/cpu/test_keep.py" in out
    assert "tests/ut/cpu/test_skip.py" not in out
    assert "tests/ut/npu/a2" in out
    assert "tests/e2e/pull_request/one_card/test_one_card.py" in out
    assert "tests/e2e/pull_request/two_card/test_two_card.py" in out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_tests.py",
            "--config",
            str(config_path),
            "--changed-files",
            "src/nodeid.py",
        ],
    )
    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=nodeid" in out
    assert "tests/e2e/pull_request/two_card/test_two_card.py::test_specific_case" in out
    assert "tests/e2e/pull_request/two_card/test_two_card.py::test_other_case" not in out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_tests.py",
            "--config",
            str(config_path),
            "--changed-files",
            "tests/e2e/pull_request/two_card/test_two_card.py",
        ],
    )
    select_tests.main()
    out = capsys.readouterr().out
    groups_line = next(line for line in out.splitlines() if line.startswith("test_groups="))
    test_groups = json.loads(groups_line.removeprefix("test_groups="))
    selected_tests = set(test_groups[0]["tests"].split())
    assert selected_tests == {"tests/e2e/pull_request/two_card/test_two_card.py::test_specific_case"}


def test_default_cpu_ut_always_runs(tmp_path, monkeypatch, capsys):
    test_root = tmp_path / "tests"
    cpu_dir = test_root / "ut" / "cpu"
    a2_dir = test_root / "ut" / "npu" / "a2"
    for path in (cpu_dir, a2_dir):
        path.mkdir(parents=True)
    cpu_test = cpu_dir / "test_cpu.py"
    a2_test = a2_dir / "test_a2.py"
    cpu_test.write_text("")
    a2_test.write_text("")

    config = [
        {
            "name": "default_cpu_ut",
            "optional": False,
            "cpu_only": True,
            "tests": ["tests/ut"],
        },
        {
            "name": "specific",
            "optional": True,
            "source_file_dependencies": ["src/specific.py"],
            "tests": ["tests/ut/npu/a2"],
        },
    ]
    config_path = tmp_path / "config.yaml"
    runner_mapping = {"tests/ut/.+/a2": {"default": "a2_x1"}}
    _write_two_doc_config(config_path, config, {"runner_mapping": runner_mapping})
    runner_file = tmp_path / "runner_label.json"
    runner_file.write_text(
        json.dumps(
            {
                "cpu-runner": {"chip": "cpu", "npu_num": 0},
                "a2-runner": {"chip": "a2", "npu_num": 1},
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
            "README.md",
        ],
    )
    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=default_cpu_ut" in out
    groups_line = next(line for line in out.splitlines() if line.startswith("test_groups="))
    test_groups = json.loads(groups_line.removeprefix("test_groups="))
    cpu_tests = {g["tests"] for g in test_groups if g["npu_type"] == "cpu"}
    assert any("test_cpu.py" in t for t in cpu_tests)
    a2_tests = [g for g in test_groups if g["npu_type"] == "a2"]
    assert not a2_tests

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_tests.py",
            "--config",
            str(config_path),
            "--changed-files",
            "src/specific.py",
        ],
    )
    select_tests.main()
    out = capsys.readouterr().out
    assert "matched_modules=default_cpu_ut,specific" in out
    groups_line = next(line for line in out.splitlines() if line.startswith("test_groups="))
    test_groups = json.loads(groups_line.removeprefix("test_groups="))
    cpu_tests = {g["tests"] for g in test_groups if g["npu_type"] == "cpu"}
    assert any("test_cpu.py" in t for t in cpu_tests)
    a2_tests = {g["tests"] for g in test_groups if g["npu_type"] == "a2"}
    assert any("test_a2.py" in t for t in a2_tests)


def _write_two_doc_config(path, modules, meta):
    """Write a two-document YAML config (modules + meta) for select_tests.py."""
    path.write_text(yaml.safe_dump(modules) + "---\n" + yaml.safe_dump(meta))


def test_explicit_e2e_tests_runs_only_specified_paths(tmp_path, monkeypatch, capsys):
    """--explicit-e2e-tests must bypass module matching and run only the
    user-specified paths, regardless of ``optional: false`` modules that
    would otherwise pull in the full suite."""
    test_root = tmp_path / "tests"
    e2e_one_card = test_root / "e2e" / "pull_request" / "one_card"
    e2e_two_card = test_root / "e2e" / "pull_request" / "two_card"
    e2e_four_card = test_root / "e2e" / "pull_request" / "four_card"
    for path in (e2e_one_card, e2e_two_card, e2e_four_card):
        path.mkdir(parents=True)
    one_a = e2e_one_card / "test_one_a.py"
    one_b = e2e_one_card / "test_one_b.py"
    one_310p = e2e_one_card / "test_one_310p.py"
    two_a = e2e_two_card / "test_two_a.py"
    two_b = e2e_two_card / "test_two_b.py"
    four_a = e2e_four_card / "test_four_a.py"
    for path in (one_a, one_b, one_310p, two_a, two_b, four_a):
        path.write_text("")

    # Module with ``optional: false`` would normally pull in the entire e2e
    # suite via _match_modules; explicit mode must skip it.
    config_modules = [
        {
            "name": "always_run_e2e",
            "optional": False,
            "source_file_dependencies": ["src/any.py"],
            "tests": [
                "tests/e2e/pull_request/one_card",
                "tests/e2e/pull_request/two_card",
                "tests/e2e/pull_request/four_card",
            ],
        },
    ]
    runner_mapping = {
        "tests/e2e/pull_request/one_card": {"default": "a2_x1", "310p": "310p_x1"},
        "tests/e2e/pull_request/two_card": {"default": "a3_x2"},
        "tests/e2e/pull_request/four_card": {"default": "a3_x4", "310p": "310p_x4"},
    }
    config_path = tmp_path / "config.yaml"
    _write_two_doc_config(config_path, config_modules, {"runner_mapping": runner_mapping})
    runner_file = tmp_path / "runner_label.json"
    runner_file.write_text(
        json.dumps(
            {
                "a2-runner": {"chip": "a2", "npu_num": 1},
                "a3-runner-2": {"chip": "a3", "npu_num": 2},
                "a3-runner-4": {"chip": "a3", "npu_num": 4},
                "310p-runner": {"chip": "310p", "npu_num": 1},
            }
        )
    )
    monkeypatch.setattr(select_tests, "_RUNNER_LABEL_PATH", runner_file)
    monkeypatch.chdir(tmp_path)

    # Use repo-relative paths because the script's _is_e2e_path / routing
    # patterns expect paths starting with "tests/".
    rel_one_a = "tests/e2e/pull_request/one_card/test_one_a.py"
    rel_one_b = "tests/e2e/pull_request/one_card/test_one_b.py"
    rel_one_310p = "tests/e2e/pull_request/one_card/test_one_310p.py"
    rel_two_a = "tests/e2e/pull_request/two_card/test_two_a.py"
    rel_two_b = "tests/e2e/pull_request/two_card/test_two_b.py"
    rel_four_a = "tests/e2e/pull_request/four_card/test_four_a.py"
    rel_e2e_one = "tests/e2e/pull_request/one_card"
    rel_ut_file = "tests/ut/test_ut.py"
    rel_missing = "tests/e2e/pull_request/one_card/does_not_exist.py"

    def run_explicit(*paths):
        capsys.readouterr()
        monkeypatch.setattr(
            sys,
            "argv",
            ["select_tests.py", "--config", str(config_path), "--explicit-e2e-tests", *paths],
        )
        select_tests.main()
        captured = capsys.readouterr()
        out, err = captured.out, captured.err
        groups_line = next((line for line in out.splitlines() if line.startswith("test_groups=")), None)
        assert groups_line is not None
        test_groups = json.loads(groups_line.removeprefix("test_groups="))
        return test_groups, out, err

    # 1. Single file routes to the correct runner.
    test_groups, out, _ = run_explicit(rel_one_a)
    matched = out.split("matched_modules=")[1].strip()
    assert matched == ""
    assert len(test_groups) == 1
    assert test_groups[0]["npu_type"] == "a2"
    assert test_groups[0]["num_npus"] == 1
    assert test_groups[0]["tests"].split() == [rel_one_a]

    # 2. Multiple files spanning different runners.
    test_groups, _, _ = run_explicit(rel_one_a, rel_two_a, rel_four_a)
    npu_keys = {(g["npu_type"], g["num_npus"]) for g in test_groups}
    assert npu_keys == {("a2", 1), ("a3", 2), ("a3", 4)}
    selected = {t for g in test_groups for t in g["tests"].split()}
    assert selected == {rel_one_a, rel_two_a, rel_four_a}

    # 3. _310p suffix overrides the default runner.
    test_groups, _, _ = run_explicit(rel_one_310p)
    assert test_groups[0]["npu_type"] == "310p"
    assert test_groups[0]["num_npus"] == 1

    # 4. Directory input rglobs all test_*.py under it.
    test_groups, _, _ = run_explicit(rel_e2e_one)
    selected = {t for g in test_groups for t in g["tests"].split()}
    assert selected == {rel_one_a, rel_one_b, rel_one_310p}
    npu_types = {g["npu_type"] for g in test_groups}
    assert npu_types == {"a2", "310p"}

    # 5. ::nodeid suffix is preserved and routed by file path.
    nodeid = f"{rel_one_a}::TestClass::test_method"
    test_groups, _, _ = run_explicit(nodeid)
    assert test_groups[0]["tests"].split() == [nodeid]
    assert test_groups[0]["npu_type"] == "a2"

    # 6. Non-e2e path is skipped with a warning.
    test_groups, _, err = run_explicit(rel_ut_file)
    assert test_groups == []
    assert "Skipping non-e2e path" in err

    # 7. Non-existent path is dropped with a warning; no test groups emitted.
    test_groups, out, err = run_explicit(rel_missing)
    assert test_groups == []
    assert "has_tests=false" in out
    assert "Path does not exist" in err
    assert rel_missing in err

    # 8. Mix of valid and invalid paths: only valid ones are routed.
    test_groups, _, err = run_explicit(rel_two_a, rel_ut_file, rel_missing)
    selected = {t for g in test_groups for t in g["tests"].split()}
    assert selected == {rel_two_a}
    assert "Skipping non-e2e path" in err

    # 9. Optional modules are NOT triggered in explicit mode even if their
    #    source_file_dependencies happen to match the explicit path.
    config_with_match = [
        {
            "name": "always_run_e2e",
            "optional": False,
            "source_file_dependencies": ["src/any.py"],
            "tests": [
                "tests/e2e/pull_request/one_card",
                "tests/e2e/pull_request/two_card",
                "tests/e2e/pull_request/four_card",
            ],
        },
        {
            "name": "would_match",
            "optional": True,
            "source_file_dependencies": [rel_e2e_one],
            "tests": [rel_e2e_one],
        },
    ]
    _write_two_doc_config(config_path, config_with_match, {"runner_mapping": runner_mapping})
    test_groups, _, _ = run_explicit(rel_two_b)
    selected = {t for g in test_groups for t in g["tests"].split()}
    assert selected == {rel_two_b}
    assert "test_two_a.py" not in selected

    # 10. ::nodeid is filtered out when the underlying file is in skip_tests.
    config_with_skip = [
        {
            "name": "with_skip",
            "optional": True,
            "source_file_dependencies": ["src/any.py"],
            "tests": [rel_e2e_one, rel_two_a, rel_two_b],
            "skip_tests": [rel_one_a],
        },
    ]
    _write_two_doc_config(config_path, config_with_skip, {"runner_mapping": runner_mapping})
    test_groups, out, _ = run_explicit(f"{rel_one_a}::TestClass::test_method")
    selected = {t for g in test_groups for t in g["tests"].split()}
    assert selected == set()
    assert "has_tests=false" in out

    # 11. Partition splits multiple same-runner tests; single test into a
    #     psize=5 runner yields 1 non-empty partition (others are dropped).
    config_with_partition = [
        {
            "name": "with_partition",
            "optional": True,
            "source_file_dependencies": ["src/any.py"],
            "tests": [rel_e2e_one],
        },
    ]
    _write_two_doc_config(
        config_path,
        config_with_partition,
        {"runner_mapping": runner_mapping, "partition": {"a2_x1": 5}},
    )
    test_groups, _, _ = run_explicit(rel_one_a, rel_one_b)
    a2_groups = [g for g in test_groups if g["npu_type"] == "a2"]
    assert len(a2_groups) >= 1
    a2_tests = {t for g in a2_groups for t in g["tests"].split()}
    assert a2_tests == {rel_one_a, rel_one_b}
    assert all(g["partition"].endswith("-5") for g in a2_groups)
