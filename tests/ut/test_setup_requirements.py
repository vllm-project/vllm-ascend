# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

import pytest


@pytest.fixture
def read_requirements(tmp_path):
    # Execute the real helper without triggering setup() or native build imports.
    setup_path = Path(__file__).parents[2] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_requirements")
    namespace = {"get_path": lambda filename: str(tmp_path / filename)}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(setup_path), "exec"), namespace)
    return namespace["get_requirements"]


def test_get_requirements_resolves_includes(read_requirements, tmp_path):
    (tmp_path / "requirements.txt").write_text("torch==2.10.0\n-r extra.txt\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("--extra-index-url https://example.com/simple\npytest>=8\n", encoding="utf-8")

    assert read_requirements() == ["torch==2.10.0", "pytest>=8"]


@pytest.mark.parametrize("filename", ["requirements.txt", "extra.txt"])
def test_get_requirements_reports_missing_file(read_requirements, tmp_path, filename):
    if filename != "requirements.txt":
        (tmp_path / "requirements.txt").write_text(f"-r {filename}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to read requirements.txt") as exc_info:
        read_requirements()

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert exc_info.value.__cause__.filename == str(tmp_path / filename)
    assert filename in str(exc_info.value)


def test_get_requirements_preserves_permission_error(read_requirements, tmp_path, monkeypatch):
    error = PermissionError(13, "Permission denied", str(tmp_path / "requirements.txt"))

    def deny_open(*args, **kwargs):
        raise error

    monkeypatch.setattr("builtins.open", deny_open)

    with pytest.raises(RuntimeError, match="Failed to read requirements.txt") as exc_info:
        read_requirements()

    assert exc_info.value.__cause__ is error
    assert "Permission denied" in str(exc_info.value)
