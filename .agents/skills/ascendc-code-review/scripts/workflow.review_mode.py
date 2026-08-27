#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
检视模式路由器。读 workflow.review-thresholds.yaml，按变更行数/文件数判定 mode，
输出 JSON {mode, routing, guidance}。code-fetch / file-review 入口调用，阈值单一来源在 yaml。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

CFG = Path(__file__).parent / "workflow.review-thresholds.yaml"


def load_cfg():
    """读 workflow.review-thresholds.yaml。"""
    try:
        import yaml  # PyYAML 6.x
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"PyYAML is required (pip install pyyaml): {exc}") from exc
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_large(lines, files, large_cfg):
    """大型 PR 双高判定：文件数 > min_files 且 行数 >= min_lines。files 为空则不触发。"""
    if files is None:
        return False
    return files > large_cfg["min_files"] and lines >= large_cfg["min_lines"]


def pick_mode(lines, modes_cfg):
    """按行数选 minimal/compact/standard 条目。"""
    if lines < modes_cfg["minimal"]["max_lines"]:
        return "minimal", modes_cfg["minimal"]
    if lines < modes_cfg["compact"]["max_lines"]:
        return "compact", modes_cfg["compact"]
    return "standard", modes_cfg["standard"]


def decide(lines, files, cfg):
    """返回 (mode, routing, guidance)。"""
    if is_large(lines, files, cfg["large_pr"]):
        large = cfg["large_pr"]
        return "large", large["routing"], large["guidance"]
    name, entry = pick_mode(lines, cfg["review_modes"])
    return name, entry["routing"], entry["guidance"]


def main():
    ap = argparse.ArgumentParser(description="检视模式路由器")
    ap.add_argument("--lines", type=int, required=True,
                    help="变更行数（PR=diff 新增+删除排除注释空行；文件检视=代码总行数）")
    ap.add_argument("--files", type=int, default=None, help="变更文件数（文件检视模式可省略）")
    args = ap.parse_args()
    try:
        cfg = load_cfg()
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    mode, routing, guidance = decide(args.lines, args.files, cfg)
    logger.info("mode=%s routing=%s lines=%d files=%s", mode, routing, args.lines, args.files)
    logger.info(json.dumps({"mode": mode, "routing": routing, "guidance": guidance}, ensure_ascii=False))


if __name__ == "__main__":
    main()
