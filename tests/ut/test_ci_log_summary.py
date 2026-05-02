#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "ci_log_summary",
    _ROOT / ".github" / "workflows" / "scripts" / "ci_log_summary.py",
)
assert _SUMMARY_SPEC and _SUMMARY_SPEC.loader
_summary = importlib.util.module_from_spec(_SUMMARY_SPEC)
_SUMMARY_SPEC.loader.exec_module(_summary)


def test_process_local_log_skips_good_commit_when_fetch_remote_meta_false(monkeypatch):
    log_text = """
[1/10] FAILED (exit code 4)  tests/e2e/multicard/4-cards/test_kimi_k2.py  (26s)
""".strip()

    def _should_not_call_good_commit():
        raise AssertionError("get_good_commit must not run when fetch_remote_meta is False")

    monkeypatch.setattr(_summary, "get_good_commit", _should_not_call_good_commit)
    monkeypatch.setattr(_summary, "extract_bad_commit", lambda log_text, resolve_remote=False: None)

    result = _summary.process_local_log(log_text, fetch_remote_meta=False)

    assert result["good_commit"] is None
    assert result["failed_test_files"] == ["tests/e2e/multicard/4-cards/test_kimi_k2.py"]


def test_process_local_log_fetches_good_commit_by_default(monkeypatch):
    monkeypatch.setattr(_summary, "get_good_commit", lambda: "good-sha")
    monkeypatch.setattr(_summary, "extract_bad_commit", lambda log_text, resolve_remote=False: None)

    result = _summary.process_local_log("RuntimeError: boom")

    assert result["good_commit"] == "good-sha"


def test_render_summary_heading_is_ci_failure_summary():
    body = _summary.render_summary(
        {
            "run_id": None,
            "run_url": None,
            "failed_test_files": ["tests/e2e/a.py"],
            "failed_test_cases": [],
            "distinct_errors": [],
            "code_bugs": [],
            "env_flakes": [],
        },
        step_name="nightly",
        mode="e2e",
    )
    assert body.startswith("## CI failure summary: nightly\n")
