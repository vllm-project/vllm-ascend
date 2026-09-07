# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import io
from http.client import HTTPMessage
from pathlib import Path
from urllib.error import HTTPError

import pytest


@pytest.fixture(scope="module")
def helper():
    path = Path(__file__).resolve().parents[2] / "e2e/doctests/scripts/doctest_helper.py"
    spec = importlib.util.spec_from_file_location("doctest_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_plan(quickstart=(), installation=()):
    return {
        "quickstart": {"include": [{"image": image} for image in quickstart]},
        "installation": {"include": [{"method": method, "image": image} for method, image in installation]},
        "run_quickstart": bool(quickstart),
        "run_installation": bool(installation),
        "skipped": [],
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain text", None),
        ("<!-- doctest: demo -->\n```bash\necho ok\n```", "echo ok\n"),
        ("  <!-- doctest: demo -->\n\n  ```python\n  if True:\n    pass\n  ```", "if True:\n  pass\n"),
    ],
)
def test_extract_doctest_block(helper, text, expected):
    assert helper.extract_doctest_block(text, "demo") == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("<!-- doctest: demo -->\n<!-- doctest: demo -->", "Duplicate"),
        ("<!-- doctest: demo -->\n```json\n{}\n```", "must be followed"),
        ("<!-- doctest: demo -->\n```bash\necho ok", "not closed"),
    ],
)
def test_extract_doctest_block_rejects_invalid_input(helper, text, message):
    with pytest.raises(helper.DoctestError, match=message):
        helper.extract_doctest_block(text, "demo")


def test_mkdocs_helpers(helper):
    extra = helper.parse_mkdocs_extra("extra:\n  version: 1.0\n  enabled: true\n  nested: [a, b]\n")

    assert extra == {"version": "1.0", "enabled": "true"}
    assert helper.expand_mkdocs_macros("v{{ version }}", extra, "demo") == "v1.0"
    with pytest.raises(helper.DoctestError, match="Unknown mkdocs.yml macro"):
        helper.expand_mkdocs_macros("{{ missing }}", extra, "demo")


@pytest.mark.parametrize(
    "group_name, expected",
    [
        ("QUICKSTART_COMMON_MARKERS", {"quickstart": ["a2", "310p"], "installation": []}),
        ("QUICKSTART_A2_MARKERS", {"quickstart": ["a2"], "installation": []}),
        ("QUICKSTART_310P_MARKERS", {"quickstart": ["310p"], "installation": []}),
        ("INSTALLATION_COMMON_MARKERS", {"quickstart": [], "installation": ["pip"]}),
        ("INSTALLATION_PIP_MARKERS", {"quickstart": [], "installation": ["pip"]}),
        ("INSTALLATION_UV_MARKERS", {"quickstart": [], "installation": ["uv"]}),
        ("INSTALLATION_SOURCE_MARKERS", {"quickstart": [], "installation": ["source"]}),
    ],
)
def test_select_doctests_by_marker_group(helper, monkeypatch, group_name, expected):
    changed_group = getattr(helper, group_name)
    monkeypatch.setattr(helper, "get_changed_paths", lambda base, head: set())
    monkeypatch.setattr(
        helper,
        "any_doctest_blocks_changed",
        lambda base, head, markers, cache: markers == changed_group,
    )
    monkeypatch.setattr(helper, "read_repo_text", lambda path, ref: "extra:\n  version: same\n")

    assert helper.select_doctests("base", "head") == expected


@pytest.mark.parametrize(
    "paths, release_changed, expected",
    [
        ({"tests/e2e/doctests/001-quickstart-test.sh"}, False, {"quickstart": ["a2", "310p"], "installation": []}),
        (
            {"tests/e2e/doctests/002-installation-test.sh"},
            False,
            {"quickstart": [], "installation": ["source", "uv", "pip"]},
        ),
        (
            {"tests/e2e/doctests/scripts/doctest_helper.py"},
            False,
            {"quickstart": ["a2", "310p"], "installation": ["pip"]},
        ),
        (set(), True, {"quickstart": ["a2", "310p"], "installation": ["pip"]}),
        ({"README.md"}, False, {"quickstart": [], "installation": []}),
    ],
)
def test_select_doctests_by_path_or_release(helper, monkeypatch, paths, release_changed, expected):
    monkeypatch.setattr(helper, "get_changed_paths", lambda base, head: paths)
    monkeypatch.setattr(helper, "any_doctest_blocks_changed", lambda *args: False)
    monkeypatch.setattr(
        helper,
        "read_repo_text",
        lambda path, ref: f"extra:\n  vllm_ascend_version: {'new' if release_changed and ref == 'head' else 'old'}\n",
    )

    assert helper.select_doctests("base", "head") == expected


def test_build_doctest_plan(helper, monkeypatch):
    monkeypatch.setattr(
        helper,
        "load_mkdocs_extra",
        lambda: {
            "vllm_ascend_version": "v1",
            "release_cann_version": "8.5",
            "release_image_python_version": "3.11",
        },
    )

    plan = helper.build_doctest_plan(["a2", "310p"], ["source", "pip"], "quick/", "install/")

    assert plan["quickstart"]["include"] == [
        {"device": device, "os": os_name, "image": f"quick:v1{device_tag}{os_tag}"}
        for os_name, os_tag in (("ubuntu", ""), ("openeuler", "-openeuler"))
        for device, device_tag in (("a2", ""), ("310p", "-310p"))
    ]
    assert plan["installation"]["include"] == [
        {"method": method, "os": os_name, "image": f"install:8.5-910b-{os_tag}-py3.11"}
        for os_name, os_tag in (("ubuntu", "ubuntu22.04"), ("openeuler", "openeuler24.03"))
        for method in ("source", "pip")
    ]
    assert plan["run_quickstart"] is plan["run_installation"] is True
    assert plan["skipped"] == []


def test_check_plan_resources_filters_only_missing_images(helper, monkeypatch):
    plan = make_plan(
        ("quick:ok", "quick:missing"),
        (("pip", "install:shared"), ("uv", "install:shared"), ("pip", "install:missing")),
    )
    checked = []

    def image_exists(image):
        checked.append(image)
        return not image.endswith("missing")

    monkeypatch.setattr(helper, "registry_image_exists", image_exists)
    result = helper.check_plan_resources(plan)

    assert checked == ["quick:ok", "quick:missing", "install:shared", "install:missing"]
    assert result["quickstart"]["include"] == [{"image": "quick:ok"}]
    assert [entry["method"] for entry in result["installation"]["include"]] == ["pip", "uv"]
    assert result["run_quickstart"] is result["run_installation"] is True
    assert result["skipped"] == [
        "Quick Start skipped: image not found: quick:missing",
        "Installation skipped: image not found: install:missing",
    ]


def test_non_source_installation_skips_source_probe(helper, monkeypatch):
    plan = make_plan(installation=(("pip", "install:ok"), ("uv", "install:ok")))
    monkeypatch.setattr(helper, "registry_image_exists", lambda image: True)
    monkeypatch.setattr(helper, "source_ref_exists", lambda ref: pytest.fail("unexpected source probe"))

    result = helper.check_plan_resources(plan)

    assert [entry["method"] for entry in result["installation"]["include"]] == ["pip", "uv"]
    assert result["skipped"] == []


@pytest.mark.parametrize(
    "methods, expected_methods, expected_run",
    [(["source"], [], False), (["pip", "source"], ["pip"], True)],
)
def test_missing_source_ref_filters_only_source(helper, monkeypatch, methods, expected_methods, expected_run):
    plan = make_plan(installation=tuple((method, "install:ok") for method in methods))
    monkeypatch.setattr(helper, "registry_image_exists", lambda image: True)
    monkeypatch.setattr(helper, "load_mkdocs_extra", lambda: {"vllm_ascend_version": "v1"})
    monkeypatch.setattr(helper, "source_ref_exists", lambda ref: False)

    result = helper.check_plan_resources(plan)

    assert [entry["method"] for entry in result["installation"]["include"]] == expected_methods
    assert result["run_installation"] is expected_run
    assert result["skipped"] == ["Installation source skipped: source ref not found: v1"]


@pytest.mark.parametrize("status", [200, 404, 500])
def test_registry_image_http_status(helper, monkeypatch, status):
    def urlopen(request, timeout):
        if isinstance(request, str):
            return io.BytesIO(b'{"token": "token"}')
        if status != 200:
            raise HTTPError(request.full_url, status, "error", HTTPMessage(), None)
        return io.BytesIO()

    monkeypatch.setattr(helper, "urlopen", urlopen)

    if status == 500:
        with pytest.raises(HTTPError):
            helper.registry_image_exists("registry.example/project/image:v1")
    else:
        assert helper.registry_image_exists("registry.example/project/image:v1") is (status == 200)
