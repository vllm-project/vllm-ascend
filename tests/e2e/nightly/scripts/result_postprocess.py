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
  4. Invoke an external Python script on that file

Missing preset/script files only emit warnings and never fail the test.
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRESET_JSON_PATH = Path("/root/.cache/xxx/test.json")
POSTPROCESS_SCRIPT_PATH = Path("/root/.cache/xxx/xxx.py")
OUTPUT_DIR = Path("/root/.cache/xxx/results")

PERF_METRIC_RENAME: dict[str, str] = {
    "Benchmark Duration": "Benchmark_Duration(BD)",
    "Prefill Token Throughput": "Prefill_Token_Throughput(PTT)",
    "Input Token Throughput": "Input_Token_Throughput(ITT)",
    "Output Token Throughput": "Output_Token_Throughput(OTT)",
    "Total Token Throughput": "Total_Token_Throughput(TTT)",
}


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _extract_dataset_name(case_config: dict[str, Any]) -> str:
    dataset_path = case_config.get("dataset_path", "")
    dataset_conf = case_config.get("dataset_conf", "")
    if dataset_path:
        return dataset_path.split("/", 1)[-1]
    if dataset_conf:
        return dataset_conf.split("/")[0]
    return ""


def _extract_perf_metrics(result: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not (isinstance(result, list) and len(result) == 2):
        return metrics
    _, result_json = result
    if not isinstance(result_json, dict):
        return metrics
    for metric_name, metric_data in result_json.items():
        if not isinstance(metric_data, dict):
            continue
        total_str = metric_data.get("total", "")
        try:
            value = float(str(total_str).replace("token/s", "").replace("ms", "").replace("s", "").strip())
            metrics[PERF_METRIC_RENAME.get(metric_name, metric_name)] = round(value, 4)
        except (ValueError, AttributeError):
            continue
    return metrics


def merge_postprocess_payload(
    preset: dict[str, Any],
    case_config: dict[str, Any],
    result: Any,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Deep-copy preset and patch nested fields per the preset JSON schema.

    Hierarchy follows:
      testcase_info.featureFullName / testEnv / extraTestEnv / testIndicator
      base_info (left unchanged)
    """
    payload = copy.deepcopy(preset)
    testcase_info = payload.setdefault("testcase_info", {})
    if not isinstance(testcase_info, dict):
        testcase_info = {}
        payload["testcase_info"] = testcase_info

    testcase_info["featureFullName"] = model_name

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

    case_type = case_config.get("case_type")
    if case_type == "accuracy":
        test_env["data_set"] = _extract_dataset_name(case_config)
    else:
        # Performance cases leave data_set empty.
        test_env["data_set"] = ""

    testcase_info["extraTestEnv"] = {}

    indicator: dict[str, Any] = {}
    if case_type == "accuracy":
        if isinstance(result, (int, float)):
            indicator["accuracy"] = round(float(result), 4)
    elif case_type == "performance":
        indicator.update(_extract_perf_metrics(result))
    testcase_info["testIndicator"] = indicator

    return payload


def _load_preset_json(preset_path: Path) -> dict[str, Any] | None:
    if not preset_path.is_file():
        logger.warning("Preset JSON not found, skip postprocess: %s", preset_path)
        return None
    try:
        return json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read preset JSON %s: %s", preset_path, exc)
        return None


def _run_postprocess_script(script_path: Path, output_path: Path) -> None:
    if not script_path.is_file():
        logger.warning("Postprocess script not found, skip running: %s", script_path)
        return
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        logger.warning("Failed to run postprocess script %s: %s", script_path, exc)
        return
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.warning(
            "Postprocess script exited with code %s for %s%s",
            completed.returncode,
            output_path,
            f": {stderr}" if stderr else "",
        )


def postprocess_one_benchmark(
    case_key: str,
    case_config: dict[str, Any],
    result: Any,
    *,
    job_name: str,
    model_name: str,
    preset_path: Path = PRESET_JSON_PATH,
    script_path: Path = POSTPROCESS_SCRIPT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Read preset JSON, patch nested fields, write output JSON, and run xxx.py."""
    preset = _load_preset_json(preset_path)
    if preset is None:
        return None

    payload = merge_postprocess_payload(
        preset,
        case_config,
        result,
        model_name=model_name,
    )

    safe_job = _safe_name(job_name or "benchmark")
    safe_case = _safe_name(case_key)
    output_path = output_dir / f"{safe_job}_{safe_case}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write postprocess JSON %s: %s", output_path, exc)
        return None

    logger.info("Postprocess JSON written to %s", output_path)
    _run_postprocess_script(script_path, output_path)
    return output_path


def postprocess_benchmark_results(
    items: list[tuple[str, dict[str, Any], Any]],
    *,
    job_name: str,
    model_name: str,
    preset_path: Path = PRESET_JSON_PATH,
    script_path: Path = POSTPROCESS_SCRIPT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Post-process every (case_key, case_config, result) entry."""
    written: list[Path] = []
    for case_key, case_config, result in items:
        path = postprocess_one_benchmark(
            case_key,
            case_config,
            result,
            job_name=job_name,
            model_name=model_name,
            preset_path=preset_path,
            script_path=script_path,
            output_dir=output_dir,
        )
        if path is not None:
            written.append(path)
    return written
