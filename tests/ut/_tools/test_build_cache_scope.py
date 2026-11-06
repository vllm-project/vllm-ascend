# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import shutil
import sys
from pathlib import Path

import pytest

from .build_cache_test_utils import build_cache_command, run_command
from .test_build_cache import (
    _assert_success,
    _extract_key,
    _load_engine,
    _make_operator_inputs,
    _run_cache,
    _write_builder,
)


@pytest.mark.parametrize(
    ("relative", "excludes", "expected"),
    [
        ("README.md", ["*.md"], "HIT"),
        ("docs/design.rst", ["docs/**"], "HIT"),
        ("tests/ut/test_tiling.cpp", ["tests/**"], "HIT"),
        ("tests/ut/test_tiling.cpp", [], "MISS"),
        ("op_host/tiling.cpp", ["*.md", "tests/**"], "MISS"),
        ("op_kernel/kernel.h", ["*.md", "tests/**"], "MISS"),
    ],
)
def test_exclusions_apply_to_prepared_inputs_and_operator_namespace(tmp_path, relative, excludes, expected):
    source, prepared = _make_operator_inputs(tmp_path)
    for root in (source, prepared):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before\n")
    counter = tmp_path / "counter"
    arguments = dict(
        cache_root=tmp_path / "cache",
        prepared_inputs=[prepared],
        operator_source=source,
        output_dir=tmp_path / "output",
        builder=_write_builder(tmp_path),
        counter=counter,
        excludes=excludes,
    )
    first = _run_cache(**arguments)
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    for root in (source, prepared):
        (root / relative).write_text("after\n")
    # Force restoration rather than accepting the previous output in place.
    (tmp_path / "output/kernel.o").unlink()
    second = _run_cache(**arguments)
    _assert_success(second)
    assert f"[build-cache] {expected}" in second.stdout
    assert counter.read_text() == ("1" if expected == "HIT" else "2")


@pytest.mark.skipif(shutil.which("env") is None, reason="production command uses POSIX env")
def test_move_whole_cache_to_identical_checkout(tmp_path):
    builder = _write_builder(tmp_path)
    counter = tmp_path / "counter"
    roots = [tmp_path / "workspace-a", tmp_path / "workspace-b"]
    for root in roots:
        root.mkdir()
        source, prepared = _make_operator_inputs(root)
        (source / "op_host").mkdir()
        (source / "op_host/tiling.h").write_text("int tiling = 1;\n")
        config = root / "custom-other/op_impl/ai_core/tbe/config/ascend910b"
        config.mkdir(parents=True)
        (config / "ops-info.json").write_text('{"Operator":{"value":1}}\n')

    def run(root: Path):
        opp = root / "custom-other"
        output = root / "stage"
        return run_command(
            build_cache_command(
                cache_root=root / "build_cache",
                domain="custom_operator",
                unit="test_unit",
                soc="ascend910b",
                output_dir=output,
                environment_profile="ascendc",
                prepared_inputs=[root / "prepared", opp / "op_impl/ai_core/tbe/config/ascend910b"],
                recipe_values=[],
                environment_values=[],
                environment_tools=[sys.executable],
                normalize_paths=[root, opp],
                operator_source=root / "source",
                operator="test_operator",
                action="TestOperator-0",
                publish_dir=root / "published",
                publish_state_dir=root / "publish-state",
                build_command=[
                    "env",
                    f"ASCEND_CUSTOM_OPP_PATH={opp}",
                    sys.executable,
                    str(builder),
                    str(output),
                    str(counter),
                    "counted",
                    "kernel.o",
                ],
            )
        )

    first = run(roots[0])
    _assert_success(first)
    assert "[build-cache] MISS" in first.stdout
    (roots[0] / "build_cache").rename(roots[1] / "build_cache")
    assert not (roots[0] / "build_cache").exists()
    assert not (roots[1] / "stage").exists()
    moved = run(roots[1])
    _assert_success(moved)
    assert "[build-cache] HIT" in moved.stdout
    assert _extract_key(first) == _extract_key(moved)
    assert (roots[1] / "published/kernel.o").read_text() == "artifact-1"
    assert counter.read_text() == "1"

    # Existing source identity must still reject changed host tiling definitions.
    (roots[1] / "source/op_host/tiling.h").write_text("int tiling = 2;\n")
    changed_host = run(roots[1])
    _assert_success(changed_host)
    assert "[build-cache] MISS" in changed_host.stdout
    config = roots[1] / "custom-other/op_impl/ai_core/tbe/config/ascend910b/ops-info.json"
    config.write_text('{"Operator":{"value":2}}\n')
    changed_config = run(roots[1])
    _assert_success(changed_config)
    assert "[build-cache] MISS" in changed_config.stdout
    assert counter.read_text() == "3"


def test_command_root_normalizes_but_set_env_retains_exact_value(tmp_path):
    engine = _load_engine("build_cache_scope_recipe")
    roots = [tmp_path / "a/custom", tmp_path / "b/custom-other"]

    def recipe(root, *, command_env=False):
        override = f"ASCEND_CUSTOM_OPP_PATH={root}"
        command = ["env", override, "compiler"] if command_env else ["compiler"]
        return engine._hash_recipe([], [], command, [root], [] if command_env else [override])[0]

    assert recipe(roots[0]) != recipe(roots[1])
    assert recipe(roots[0], command_env=True) == recipe(roots[1], command_env=True)
