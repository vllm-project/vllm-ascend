#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Post-process nightly/weekly benchmark results into per-case JSON files.

For each accuracy/performance benchmark entry:
  1. Read a preset JSON template
  2. Patch nested testcase_info fields (preserve base_info)
  3. Write a new JSON file
  4. Upload via tools/upload_to_openlibing.py

Missing preset/script files only emit warnings and never fail the test.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PRESET_JSON_PATH = Path("/root/.cache/upload_perf/test.json")
OUTPUT_DIR = Path("/root/.cache/upload_perf/results")
UPLOAD_LABEL = "performance"
_DATASET_PREFIX = "vllm-ascend/"


def _default_upload_script_path() -> Path:
    cwd_candidate = Path("tools/upload_to_openlibing.py")
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "upload_to_openlibing.py"
        if candidate.is_file():
            return candidate
    return cwd_candidate


POSTPROCESS_SCRIPT_PATH = _default_upload_script_path()


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def resolve_suite_name(config_base_path: str | None = None) -> str:
    """Return 'weekly' or 'nightly' from CONFIG_BASE_PATH."""
    base = config_base_path if config_base_path is not None else os.getenv("CONFIG_BASE_PATH", "")
    normalized = base.replace("\\", "/")
    return "weekly" if "weekly" in normalized else "nightly"


def resolve_testcase_name(config_yaml_path: str | None = None, fallback: str = "") -> str:
    """Return YAML stem from CONFIG_YAML_PATH (without .yaml/.yml)."""
    raw = config_yaml_path if config_yaml_path is not None else os.getenv("CONFIG_YAML_PATH", "")
    if not raw:
        return fallback
    name = Path(raw).name
    for suffix in (".yaml", ".yml"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _extract_dataset_name(case_config: dict[str, Any]) -> str:
    dataset_path = str(case_config.get("dataset_path", "") or "")
    if dataset_path.startswith(_DATASET_PREFIX):
        return dataset_path[len(_DATASET_PREFIX) :]
    return ""


def _extract_output_tps(result: Any) -> float | None:
    if not (isinstance(result, list) and len(result) == 2):
        return None
    _, result_json = result
    if not isinstance(result_json, dict):
        return None
    metric = result_json.get("Output Token Throughput", {})
    if not isinstance(metric, dict):
        return None
    total_str = metric.get("total", "")
    try:
        return round(float(str(total_str).replace("token/s", "").strip()), 4)
    except (ValueError, AttributeError):
        return None


def merge_postprocess_payload(
    preset: dict[str, Any],
    case_config: dict[str, Any],
    result: Any,
    *,
    testcase_name: str,
    suite_name: str | None = None,
) -> dict[str, Any]:
    """Deep-copy preset and patch nested fields per the preset JSON schema."""
    payload = copy.deepcopy(preset)
    testcase_info = payload.setdefault("testcase_info", {})
    if not isinstance(testcase_info, dict):
        testcase_info = {}
        payload["testcase_info"] = testcase_info

    testcase_info["featureFullName"] = suite_name or resolve_suite_name()
    testcase_info["Testcase_Name"] = testcase_name

    test_env = testcase_info.get("testEnv")
    if not isinstance(test_env, dict):
        test_env = {}
        testcase_info["testEnv"] = test_env

    test_env["request_rate"] = case_config.get("request_rate", 0)
    if "max_out_len" in case_config:
        test_env["output_len"] = case_config["max_out_len"]
    if "batch_size" in case_config:
        test_env["Concurrency"] = case_config["batch_size"]
    if "num_prompts" in case_config:
        test_env["data_num"] = case_config["num_prompts"]
    test_env["data_set"] = _extract_dataset_name(case_config)

    testcase_info["extraTestEnv"] = {}

    indicator: dict[str, Any] = {}
    case_type = case_config.get("case_type")
    if case_type == "accuracy":
        if isinstance(result, (int, float)):
            indicator["accuracy"] = round(float(result), 4)
    elif case_type == "performance":
        output_tps = _extract_output_tps(result)
        if output_tps is not None:
            indicator["output_tps"] = output_tps
    testcase_info["testIndicator"] = indicator

    return payload


def _load_preset_json(preset_path: Path) -> dict[str, Any] | None:
    if not preset_path.is_file():
        print(f"Warning: Preset JSON not found, skip postprocess: {preset_path}")
        return None
    try:
        return json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to read preset JSON {preset_path}: {exc}")
        return None


def _run_postprocess_script(script_path: Path, output_path: Path) -> None:
    if not script_path.is_file():
        print(f"Warning: Postprocess script not found, skip running: {script_path}")
        return
    cmd = [
        sys.executable,
        str(script_path),
        "--label",
        UPLOAD_LABEL,
        "--files",
        str(output_path),
    ]
    env = os.environ.copy()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError as exc:
        print(f"Warning: Failed to run postprocess script {script_path}: {exc}")
        return
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        print(
            f"Warning: Postprocess script exited with code {completed.returncode} "
            f"for {output_path}" + (f": {stderr}" if stderr else "")
        )


def postprocess_one_benchmark(
    case_key: str,
    case_config: dict[str, Any],
    result: Any,
    *,
    job_name: str,
    testcase_name: str | None = None,
    preset_path: Path = PRESET_JSON_PATH,
    script_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Read preset JSON, patch nested fields, write output JSON, and upload."""
    if script_path is None:
        script_path = POSTPROCESS_SCRIPT_PATH

    preset = _load_preset_json(preset_path)
    if preset is None:
        return None

    resolved_name = testcase_name or resolve_testcase_name(fallback=job_name)
    payload = merge_postprocess_payload(
        preset,
        case_config,
        result,
        testcase_name=resolved_name,
    )

    safe_job = _safe_name(job_name or "benchmark")
    safe_case = _safe_name(case_key)
    output_path = output_dir / f"{safe_job}_{safe_case}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: Failed to write postprocess JSON {output_path}: {exc}")
        return None

    print(f"Postprocess JSON written to {output_path}")
    _run_postprocess_script(script_path, output_path)
    return output_path


def postprocess_benchmark_results(
    items: list[tuple[str, dict[str, Any], Any]],
    *,
    job_name: str,
    testcase_name: str | None = None,
    preset_path: Path = PRESET_JSON_PATH,
    script_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Post-process every (case_key, case_config, result) entry."""
    resolved_name = testcase_name or resolve_testcase_name(fallback=job_name)
    written: list[Path] = []
    for case_key, case_config, result in items:
        path = postprocess_one_benchmark(
            case_key,
            case_config,
            result,
            job_name=job_name,
            testcase_name=resolved_name,
            preset_path=preset_path,
            script_path=script_path,
            output_dir=output_dir,
        )
        if path is not None:
            written.append(path)
    return written
