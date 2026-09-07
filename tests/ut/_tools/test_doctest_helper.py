# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[2] / "e2e/doctests/scripts/doctest_helper.py"
    spec = importlib.util.spec_from_file_location("doctest_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ordinary text", None),
        ("<!-- doctest: demo -->\n```bash\necho ok\n```", "echo ok\n"),
        ("    <!-- doctest: demo -->\n\n    ```python\n    if True:\n        pass\n    ```", "if True:\n    pass\n"),
        ("<!-- doctest: demo -->\n```bash\n```", ""),
    ],
)
def test_extract_block(helper, text, expected):
    assert helper.extract_doctest_block(text, "demo") == expected


@pytest.mark.parametrize(
    "body, message",
    [
        ("<!-- doctest: demo -->", "Duplicate doctest marker"),
        ("``` bash\necho ok\n```", "must be followed"),
        ("```json\n{}\n```", "must be followed"),
        ("", "must be followed"),
        ("```bash\necho ok", "not closed"),
    ],
)
def test_extract_invalid_block(helper, body, message):
    with pytest.raises(helper.DoctestError, match=message):
        helper.extract_doctest_block("<!-- doctest: demo -->\n" + body, "demo")


@pytest.mark.parametrize("change", ["prose", "code", "added", "removed", "missing"])
def test_block_changed(helper, change):
    base = "<!-- doctest: demo -->\n```bash\necho old\n```"
    head = base
    if change == "prose":
        head = "New introduction\n" + base
    elif change == "code":
        head = base.replace("echo old", "echo new")
    elif change == "added":
        base = None
    elif change == "removed":
        head = None
    else:
        base = head = "No marked blocks"
    if change == "missing":
        with pytest.raises(helper.DoctestError, match="does not exist at either ref"):
            helper.doctest_block_changed(base, head, "demo", "doc.md")
    else:
        assert helper.doctest_block_changed(base, head, "demo", "doc.md") is (change != "prose")


def test_macros(helper):
    extra = helper.parse_mkdocs_extra("extra:\n  version: 1.0\n  enabled: true\n  nested: [a, b]\n")
    assert extra == {"version": "1.0", "enabled": "true"}
    assert helper.expand_mkdocs_macros("v{{ version }} {{enabled}}", extra, "demo") == "v1.0 true"
    with pytest.raises(helper.DoctestError, match="Unknown mkdocs.yml macro"):
        helper.expand_mkdocs_macros("{{ missing }}", extra, "demo")


@pytest.mark.parametrize("text", ["[", "- item", "site_name: Docs", "extra: []"])
def test_invalid_mkdocs(helper, text):
    with pytest.raises(helper.DoctestError):
        helper.parse_mkdocs_extra(text)


@pytest.mark.parametrize("key", ["release_cann_version", "vllm_version", "vllm_ascend_version", "unrelated"])
def test_release_config_changed(helper, key):
    base = f"extra:\n  {key}: old\n"
    assert not helper.release_config_changed(base, base)
    assert helper.release_config_changed(base, base.replace("old", "new")) is (key != "unrelated")


@pytest.mark.parametrize(
    "markers, paths, quickstart, installation",
    [
        ([], [], [], []),
        (["quickstart-modelscope"], [], ["a2", "310p"], []),
        (["quickstart-standard-offline"], [], ["a2"], []),
        (["quickstart-300i-duo-online-serve"], [], ["310p"], []),
        (["installation-pip-install"], [], [], ["pip"]),
        (["installation-uv-install"], [], [], ["uv"]),
        (["installation-source-install"], [], [], ["source"]),
        (["installation-post-standard"], [], [], ["source"]),
        (["installation-post-standard", "installation-pip-install"], [], [], ["pip"]),
        (["installation-pip-install", "installation-uv-install"], [], [], ["pip", "uv"]),
        ([], ["tests/e2e/doctests/001-quickstart-test.sh"], ["a2", "310p"], []),
        ([], ["tests/e2e/doctests/002-installation-test.sh"], [], ["pip", "uv", "source"]),
        ([], ["tests/e2e/doctests/scripts/doctest_helper.py"], ["a2", "310p"], ["source"]),
        ([], ["mkdocs.yml"], ["a2", "310p"], ["source"]),
        ([], ["README.md"], [], []),
    ],
)
def test_select_doctests(helper, monkeypatch, markers, paths, quickstart, installation):
    def read_text(path, ref, **kwargs):
        if path == "mkdocs.yml":
            version = "new" if ref == "head" and path in paths else "old"
            return f"extra:\n  vllm_ascend_version: {version}\n"
        return "\n".join(
            f"<!-- doctest: {marker} -->\n```bash\n{'new' if ref == 'head' and marker in markers else 'old'}\n```"
            for marker in helper.DOCTEST_MARKERS_BY_FILE[path]
        )

    monkeypatch.setattr(helper, "get_changed_paths", lambda base, head: set(paths))
    monkeypatch.setattr(helper, "read_repo_text", read_text)
    assert helper.select_doctests("base", "head") == {"quickstart": quickstart, "installation": installation}


@pytest.mark.parametrize("enabled", [False, True])
def test_build_plan(helper, monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(helper, "REPO_ROOT", tmp_path)
    (tmp_path / "mkdocs.yml").write_text(
        "extra:\n  vllm_ascend_version: v1\n  release_cann_version: 8.5\n  release_image_python_version: 3.11\n",
        encoding="utf-8",
    )
    plan = helper.build_doctest_plan(["a2", "310p"] if enabled else [], ["pip", "uv", "source"] if enabled else [])
    assert plan == {
        "run_quickstart": enabled,
        "run_installation": enabled,
        "quickstart": {
            "include": [
                {"device": "a2", "os": "ubuntu", "image_tag": "v1"},
                {"device": "a2", "os": "openeuler", "image_tag": "v1-openeuler"},
                {"device": "310p", "os": "ubuntu", "image_tag": "v1-310p"},
                {"device": "310p", "os": "openeuler", "image_tag": "v1-310p-openeuler"},
            ]
            if enabled
            else [],
        },
        "installation": {
            "include": [
                {"method": method, "os": os_name, "image_tag": tag}
                for method in ("pip", "uv", "source")
                for os_name, tag in (
                    ("ubuntu", "8.5-910b-ubuntu22.04-py3.11"),
                    ("openeuler", "8.5-910b-openeuler24.03-py3.11"),
                )
            ]
            if enabled
            else [],
        },
    }
