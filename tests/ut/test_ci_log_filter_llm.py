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
_FILTER_SPEC = importlib.util.spec_from_file_location(
    "ci_log_filter_llm",
    _ROOT / ".github" / "workflows" / "scripts" / "ci_log_filter_llm.py",
)
assert _FILTER_SPEC and _FILTER_SPEC.loader
_filter = importlib.util.module_from_spec(_FILTER_SPEC)
_FILTER_SPEC.loader.exec_module(_filter)


def test_select_important_lines_finds_error_and_guarded_info():
    lines = [
        "INFO: harmless\n",
        "ERROR: something broke\n",
        "INFO: worker thread timeout waiting for acl\n",
    ]
    got = _filter.select_important_lines(lines)
    assert any("ERROR" in g for g in got)
    assert any("timeout" in g for g in got)
    assert not any("harmless" in g for g in got)


def test_build_llm_log_bundle_includes_traceback_region():
    text = """\
starting
Traceback (most recent call last):
  File "t.py", line 1, in f
    raise RuntimeError("bad")
RuntimeError: bad
== short test summary ==
FAILED tests/e2e/foo.py::test_x - RuntimeError: bad
"""
    bundle = _filter.build_llm_log_bundle(text, context_before=2, context_after=2)
    assert "Phase A" in bundle
    assert "Phase B" in bundle
    assert "Traceback" in bundle
    assert "FAILED tests/e2e/foo.py" in bundle


def test_build_llm_log_bundle_includes_merged_artifact_section_headers():
    text = """\
=== /tmp/collected-logs/node0/var/log/vllm-0_logs.txt ===
INFO: boot
ERROR: HCCL init failed on rank 1
=== /tmp/collected-logs/node1/var/log/vllm-0-1_logs.txt ===
pod vllm-foo-0-1 phase=Running ready=false
"""
    bundle = _filter.build_llm_log_bundle(text, context_before=1, context_after=2)
    assert "=== /tmp/collected-logs/node0/" in bundle
    assert "HCCL init failed" in bundle


def test_clip_text_truncates():
    s = "a" * 1000
    out = _filter.clip_text(s, max_chars=100)
    assert len(out) < len(s)
    assert "omitted" in out


def test_clip_text_handles_tiny_limit():
    s = "abcdef"
    out = _filter.clip_text(s, max_chars=3)
    assert out == "abc"
