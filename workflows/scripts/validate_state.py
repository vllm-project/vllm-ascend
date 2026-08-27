# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
state.json 校验脚本

功能：
1. 检查 JSON 是否可解析（禁止 // 注释）
2. 检查必填键是否齐全（workflow / operator / env_summary / results / stages）
3. 检查 stages 键集合与取值是否合法（pending / running / completed / skipped / failed）
4. 检查 results 取值为 null 或 bool

用法：
python validate_state.py operators/{operator_name}/state.json

返回：
- 0: 校验通过
- 1: 校验失败
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("validate_state")


EXPECTED_STAGES = [
    "1", "CP1", "2", "CP2", "2.5", "CP2.5", "3", "CP3",
    "4", "CP4", "5", "CP5", "6", "6a", "6b", "CP6", "7",
]
STAGE_STATUSES = {"pending", "running", "completed", "skipped", "failed"}
REQUIRED_TOP = {"workflow", "operator", "env_summary", "results", "stages"}
RESULTS_KEYS = ("build", "precision", "performance")


def validate(state_path):
    """校验 state.json"""
    p = Path(state_path)
    if not p.is_file():
        return False, f"❌ state.json 不存在: {state_path}"

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"❌ JSON 解析失败: {e}"

    if not isinstance(data, dict):
        return False, "❌ state.json 顶层必须是 JSON 对象"

    missing = REQUIRED_TOP - set(data.keys())
    if missing:
        return False, f"❌ 缺少必填顶层键: {sorted(missing)}"

    if not isinstance(data["results"], dict):
        return False, "❌ results 必须是 JSON 对象"

    for key in RESULTS_KEYS:
        val = data["results"].get(key)
        if val is not None and not isinstance(val, bool):
            return False, f"❌ results.{key} 必须为 null 或 bool，当前: {val!r}"

    if not isinstance(data["stages"], dict):
        return False, "❌ stages 必须是 JSON 对象"

    stage_keys = list(data["stages"].keys())
    if stage_keys != EXPECTED_STAGES:
        return False, f"❌ stages 键集合不匹配。期望: {EXPECTED_STAGES}，实际: {stage_keys}"

    for s, info in data["stages"].items():
        if not isinstance(info, dict) or "status" not in info:
            return False, f"❌ stages.{s} 缺少 status 字段"
        status = info["status"]
        if status not in STAGE_STATUSES:
            return False, f"❌ stages.{s}.status 非法: {status!r}，合法取值: {sorted(STAGE_STATUSES)}"

    return True, "✅ state.json 校验通过"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        LOGGER.info("用法: python validate_state.py operators/{operator_name}/state.json")
        sys.exit(1)

    ok, msg = validate(sys.argv[1])
    if ok:
        LOGGER.info(msg)
    else:
        LOGGER.error(msg)
    sys.exit(0 if ok else 1)
