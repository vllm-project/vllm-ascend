# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path

import pytest

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


def test_generated_opp_relocation_hits_but_changed_inputs_miss(tmp_path):
    source, prepared = _make_operator_inputs(tmp_path)
    builder = _write_builder(tmp_path)
    counter = tmp_path / "counter"

    def run(root: Path):
        config = root / "op_impl/ai_core/tbe/config/ascend910b"
        library = root.parent / "libcust_opmaster.so"
        return _run_cache(
            cache_root=tmp_path / "cache",
            prepared_inputs=[prepared, config, library],
            operator_source=source,
            output_dir=tmp_path / "output",
            builder=builder,
            counter=counter,
            normalize_paths=[root],
            set_env_values=[f"ASCEND_CUSTOM_OPP_PATH={root}"],
            normalize_set_env=["ASCEND_CUSTOM_OPP_PATH"],
        )

    roots = [tmp_path / "workspace-a/custom", tmp_path / "workspace-b/custom-other"]
    for root in roots:
        config = root / "op_impl/ai_core/tbe/config/ascend910b/ops-info.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"Operator":{"value":1}}\n')
        (root.parent / "libcust_opmaster.so").write_bytes(b"\x7fELF\x00tiling-v1")
    first = run(roots[0])
    _assert_success(first)
    (tmp_path / "output/kernel.o").unlink()
    moved = run(roots[1])
    _assert_success(moved)
    assert "[build-cache] HIT" in moved.stdout
    assert _extract_key(first) == _extract_key(moved)
    assert counter.read_text() == "1"

    config = roots[1] / "op_impl/ai_core/tbe/config/ascend910b/ops-info.json"
    config.write_text('{"Operator":{"value":2}}\n')
    changed_config = run(roots[1])
    _assert_success(changed_config)
    assert "[build-cache] MISS" in changed_config.stdout
    (roots[1].parent / "libcust_opmaster.so").write_bytes(b"\x7fELF\x00tiling-v2")
    changed_library = run(roots[1])
    _assert_success(changed_library)
    assert "[build-cache] MISS" in changed_library.stdout
    assert counter.read_text() == "3"


def test_only_explicit_set_env_normalization_changes_path_identity(tmp_path):
    engine = _load_engine("build_cache_scope_recipe")
    roots = [tmp_path / "a/custom", tmp_path / "b/custom-other"]

    def recipe(root, *, normalized=(), name="ASCEND_CUSTOM_OPP_PATH"):
        return engine._hash_recipe([], [], ["compiler"], [root], [f"{name}={root}"], normalized)[0]

    assert recipe(roots[0]) != recipe(roots[1])
    selected = ["ASCEND_CUSTOM_OPP_PATH"]
    assert recipe(roots[0], normalized=selected) == recipe(roots[1], normalized=selected)
    assert recipe(roots[0], normalized=selected, name="CPATH") != recipe(
        roots[1], normalized=selected, name="CPATH"
    )
