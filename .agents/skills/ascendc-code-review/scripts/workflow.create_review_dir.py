#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
检视工作目录创建工具

由 plan-design / clause-grouping 子 Agent（router）在阶段0调用，
在 /tmp 下创建本次检视的结构化 yaml 输出目录，供后续 clause-review /
design-check 子 Agent 落盘 yaml 结果。

目录命名规则：
  - PR 检视:    pr{pr号}_{随机串}      例如 pr1234_a3b7x9
  - 文件检视:   file_{随机串}           例如 file_k8m2q3

用法:
  python3 create_review_dir.py --type pr --id 1234
  python3 workflow.create_review_dir.py --type file

输出: stdout 打印目录绝对路径
退出码: 0=成功, 1=参数错误, 2=目录创建失败
"""

import argparse
import logging
import os
import random
import string
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def gen_random_token(length: int = 6) -> str:
    """生成由小写字母+数字组成的随机串"""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def build_dir_name(review_type: str, review_id: str) -> str:
    """按命名规则构造目录名"""
    token = gen_random_token()
    if review_type == "pr":
        if not review_id:
            raise ValueError("PR 检视必须提供 --id（PR 号）")
        return f"pr{review_id}_{token}"
    elif review_type == "file":
        return f"file_{token}"
    else:
        raise ValueError(f"未知检视类型: {review_type}（仅支持 pr / file）")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="创建检视 yaml 输出目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["pr", "file"],
        help="检视类型: pr=PR检视, file=文件检视",
    )
    parser.add_argument(
        "--id",
        default="",
        help="检视标识: PR 检视时为 PR 号; 文件检视时留空",
    )
    args = parser.parse_args()

    try:
        dir_name = build_dir_name(args.type, args.id)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    dir_path = os.path.join("/tmp", dir_name)

    try:
        os.makedirs(dir_path, mode=0o755, exist_ok=False)
    except FileExistsError:
        # 极小概率碰撞，重试一次
        dir_name = build_dir_name(args.type, args.id)
        dir_path = os.path.join("/tmp", dir_name)
        try:
            os.makedirs(dir_path, mode=0o755, exist_ok=False)
        except Exception as e:
            logger.error("目录创建失败: %s", e)
            return 2
    except Exception as e:
        logger.error("目录创建失败: %s", e)
        return 2

    # stdout 仅打印目录绝对路径，供 router 子 Agent 捕获后回传主 Agent
    print(dir_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
