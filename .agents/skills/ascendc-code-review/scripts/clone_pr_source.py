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
"""克隆 GitCode PR 完整源码到本地（无论 PR 是否合入）

用法:
    python clone_pr_source.py --repo https://gitcode.com/cann/ops-transformer --pr 4356 --clone-dir ./pr_source/4356

ref 优先级:
    1. refs/merge-requests/{PR}/head  — PR 分支最新提交（每次 push 自动更新，首选）
    2. refs/merge-requests/{PR}/merge — 虚拟合并提交（平台异步生成，可能延迟更新，fallback）

目录已存在时:
    重新 fetch + checkout，确保获取最新 PR 代码（不再跳过）

与 get_gitcode_pr_diff.py 互补：
    - get_gitcode_pr_diff.py → 获取 diff 文件
    - clone_pr_source.py      → 获取完整源码（用于子 Agent 追溯变量定义/初始化/上游校验）
"""

import argparse
import logging
import os
import re
import subprocess
import sys

ALLOWED_GITCODE_DOMAIN = "gitcode.com"

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_repo_url(url: str) -> tuple[str, str, str]:
    """解析仓库链接，返回 (owner, repo, .git URL)"""
    if not url.startswith(f"https://{ALLOWED_GITCODE_DOMAIN}/"):
        raise ValueError(f"只支持 {ALLOWED_GITCODE_DOMAIN} 仓库，当前 URL: {url}")

    url = url.rstrip("/")
    url = re.sub(r"/pulls/\d+$", "", url)
    url = re.sub(r"\.git$", "", url)

    match = re.search(r"gitcode\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"无法解析 owner/repo: {url}")

    return match.group(1), match.group(2), f"{url}.git"


def run_git(cmd: list[str], cwd: str | None = None) -> bool:
    """执行 git 命令，成功返回 True，失败打印日志返回 False"""
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.debug("命令失败: %s — %s", " ".join(cmd), e.stderr.strip())
        return False


def clone_repo(repo_url: str, clone_dir: str) -> None:
    """克隆仓库到指定目录

    使用全量克隆（非 --depth=1）：子 Agent 需在仓库内做跨文件溯源，且
    git merge-base 计算依赖完整历史——浅克隆会让 merge-base 失败，
    导致 agent 自行算 diff 时回退到两点全量发散（见 issue #427）。
    """
    logger.info("克隆 %s 到 %s ...", repo_url, clone_dir)
    if not run_git(["git", "clone", repo_url, clone_dir]):
        raise RuntimeError("克隆仓库失败")


def fetch_pr_ref(pr_number: int, clone_dir: str) -> None:
    """获取 PR 代码：优先 head ref（最新提交），回退 merge ref（虚拟合并提交）

    head ref 每次 push 自动更新，始终指向 PR 分支最新提交；
    merge ref 是平台异步生成的虚拟合并提交，可能延迟更新，仅作 fallback。
    """
    head_ref = f"refs/merge-requests/{pr_number}/head"
    merge_ref = f"refs/merge-requests/{pr_number}/merge"

    logger.info("获取 PR #%d 代码...", pr_number)

    if run_git(["git", "fetch", "origin", f"{head_ref}:pr_ref"], cwd=clone_dir):
        run_git(["git", "checkout", "pr_ref"], cwd=clone_dir)
        logger.info("已 checkout PR #%d head 提交（最新）", pr_number)
        return

    logger.info("head ref 不存在，尝试 merge ref...")
    if run_git(["git", "fetch", "origin", f"{merge_ref}:pr_ref"], cwd=clone_dir):
        run_git(["git", "checkout", "pr_ref"], cwd=clone_dir)
        logger.info("已 checkout PR #%d merge 提交", pr_number)
        return

    raise RuntimeError(f"无法获取 PR #{pr_number} 代码（head 和 merge ref 均不存在）")


def update_existing_clone(pr_number: int, clone_dir: str) -> None:
    """更新已存在的克隆目录：重新 fetch PR ref 并 checkout"""
    logger.info("目录已存在，更新 PR #%d 代码: %s", pr_number, clone_dir)
    # 删除旧的 pr_ref 分支（如果存在），避免 fetch 时冲突
    run_git(["git", "branch", "-D", "pr_ref"], cwd=clone_dir)
    fetch_pr_ref(pr_number, clone_dir)


def main():
    parser = argparse.ArgumentParser(description="克隆 GitCode PR 完整源码")
    parser.add_argument("--repo", required=True, help="仓库链接")
    parser.add_argument("--pr", required=True, type=int, help="PR 编号")
    parser.add_argument("--clone-dir", required=True, help="克隆目标目录")
    args = parser.parse_args()

    if os.path.exists(args.clone_dir):
        try:
            update_existing_clone(args.pr, args.clone_dir)
        except RuntimeError as e:
            logger.error("更新已有克隆失败: %s", e)
            sys.exit(1)
        return

    try:
        _, _, repo_url = parse_repo_url(args.repo)
        clone_repo(repo_url, args.clone_dir)
        fetch_pr_ref(args.pr, args.clone_dir)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
