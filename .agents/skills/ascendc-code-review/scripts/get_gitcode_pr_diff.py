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
"""获取 GitCode PR Diff（无 token）

用法:
    python get_gitcode_pr_diff.py --repo <url> --pr <num> [--output <file>] [--stat] [--file-filter <pattern>]

支持的仓库:
    - cann/ops-transformer
    - cann/ops-math
    - cann/ops-nn
    - cann/ops-cv
    - 其他 gitcode.com/cann/* 下的仓库

示例:
    # ops-transformer 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-transformer --pr 3228

    # ops-math 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-math --pr 123 --stat

    # ops-nn 仓库
    python get_gitcode_pr_diff.py --repo https://gitcode.com/cann/ops-nn --pr 456 \
    --file-filter "*.asc" --output diff.txt

与 ascendc-ops-reviewer 集成:
    python skills/ascendc-code-review/scripts/get_gitcode_pr_diff.py --repo <url> --pr <num>
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

# 常量定义
ALLOWED_GITCODE_DOMAIN = "gitcode.com"
TEMP_DIR_PREFIX = "gitcode_pr_"

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


def parse_repo_url(url: str) -> tuple[str, str]:
    """解析仓库链接，返回 (owner, repo)

    支持格式:
        - https://gitcode.com/owner/repo
        - https://gitcode.com/owner/repo.git
        - https://gitcode.com/owner/repo/pulls/123

    Args:
        url: 仓库链接

    Returns:
        tuple[str, str]: (owner, repo)

    Raises:
        ValueError: 当 URL 格式不正确时抛出
    """
    # 验证 URL 格式 - 只允许 https://gitcode.com 开头
    if not url.startswith(f"https://{ALLOWED_GITCODE_DOMAIN}/"):
        raise ValueError(f"只支持 {ALLOWED_GITCODE_DOMAIN} 仓库，当前 URL: {url}")

    url = url.rstrip("/")

    # 使用更精确的正则移除末尾的 .git 和 /pulls/xxx
    url = re.sub(r"/pulls/\d+$", "", url)
    url = re.sub(r"\.git$", "", url)

    # 提取 owner/repo
    match = re.search(r"gitcode\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"无法从 URL 解析 owner/repo: {url}")

    owner = match.group(1)
    repo = match.group(2)

    return owner, repo


def run_git_command(
    cmd: list[str], cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """执行 git 命令

    Args:
        cmd: git 命令列表
        cwd: 工作目录
        check: 是否检查返回码

    Returns:
        subprocess.CompletedProcess: 命令执行结果

    Raises:
        subprocess.CalledProcessError: 当命令执行失败且 check=True 时抛出
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error("Git 命令失败: %s", " ".join(cmd))
        logger.error("工作目录: %s", cwd or "当前目录")
        logger.error("错误信息: %s", e.stderr)
        raise


def _extract_first_diff(diff_content: str) -> str:
    """提取 git show -m 输出中的第一个 diff

    git show -m 会为每个 parent 生成一个 diff，
    merge commit 有两个 parent，我们只需要第一个。

    Args:
        diff_content: git show -m 的完整输出

    Returns:
        str: 第一个 diff 的内容
    """
    if not diff_content:
        return diff_content

    lines = diff_content.splitlines(keepends=True)
    first_diff_lines: list[str] = []

    for line in lines:
        if line.startswith("commit ") and first_diff_lines:
            break
        first_diff_lines.append(line)

    if len(first_diff_lines) < len(lines):
        return "".join(first_diff_lines)

    return diff_content


def _apply_file_filter(diff_content: str, file_filter: str) -> str:
    """应用文件路径过滤到 diff 内容

    Args:
        diff_content: diff 内容
        file_filter: 文件路径过滤模式（通配符）

    Returns:
        str: 过滤后的 diff 内容
    """
    filtered_lines: list[str] = []
    current_file: str | None = None
    include_file = False

    for line in diff_content.splitlines(keepends=True):
        if line.startswith("diff --git"):
            match = re.search(r"diff --git a/(.*?) b/(.*)", line)
            if match:
                current_file = match.group(2)
                include_file = bool(
                    current_file and fnmatch.fnmatch(current_file, file_filter)
                )
            else:
                current_file = None
                include_file = False

        if include_file:
            filtered_lines.append(line)

    return "".join(filtered_lines)


def _get_base_branch_from_api(
    owner: str, repo: str, pr_number: int
) -> str | None:
    """通过 GitCode API 查询 PR 的真实目标分支（base.label）

    PR 可能目标于发布分支（如 9.1.0、9.0.0）而非 master，仅凭 main/master
    推断会把分支历史上累积的无关变更算进 diff（见 issue #463）。

    Args:
        owner: 仓库 owner
        repo: 仓库名
        pr_number: PR 编号

    Returns:
        str | None: 目标分支名（如 "9.1.0"）；查询失败返回 None（走 fallback）
    """
    url = (
        f"https://{ALLOWED_GITCODE_DOMAIN}/api/v5/repos/"
        f"{owner}/{repo}/pulls/{pr_number}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
        base = (data.get("base") or {}).get("label")
        logger.debug("API 查询 PR #%d 目标分支: %s", pr_number, base)
        return base
    except (OSError, ValueError) as e:
        logger.debug("API 查询目标分支失败: %s", e)
        return None


def _resolve_base_branch(
    repo_dir: str, owner: str, repo: str, pr_number: int
) -> str:
    """确定 PR 的真实目标分支

    优先用 GitCode API 的 base.label（支持 9.1.0 等发布分支），
    失败时 fallback 到 main/master 推断并告警（避免静默用错 base）。

    Args:
        repo_dir: bare 仓库目录
        owner: 仓库 owner
        repo: 仓库名
        pr_number: PR 编号

    Returns:
        str: 目标分支名称（如 "9.1.0"、"master"）

    Raises:
        RuntimeError: 当无法确定基础分支时抛出
    """
    result = run_git_command(["git", "branch"], cwd=repo_dir)
    # 精确匹配分支名（按行 strip 后全等比较），避免子串匹配误判：
    # 例如 "9.1.0" in branches 会命中 "9.1.0-beta.1" 那行的子串，
    # "main" 会命中 "mainline"/"domain"。见 atomic 机器人对 PR#617 的 P3 审查意见。
    branches = {line.strip().lstrip("* ").strip() for line in result.stdout.splitlines()}

    base = _get_base_branch_from_api(owner, repo, pr_number)
    if base and base in branches:
        return base

    if base:
        logger.warning(
            "API 返回目标分支 %s 但本地不存在，回退到 main/master 推断", base
        )
    else:
        logger.warning(
            "未能从 API 获取目标分支，回退到 main/master 推断，diff 可能不准确"
        )

    for branch in ["main", "master"]:
        if branch in branches:
            return branch

    raise RuntimeError("无法确定基础分支（未找到 main 或 master）")


def _cleanup_temp_dir(temp_dir: str) -> None:
    """清理临时目录

    Args:
        temp_dir: 临时目录路径
    """
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        logger.warning("清理临时目录失败: %s", e)


def get_pr_diff_git(
    repo_url: str,
    pr_number: int,
    file_filter: str | None = None,
    stat_only: bool = False,
) -> str:
    """通过 git 命令获取 PR diff

    优先使用 head 引用（PR 分支最新提交），确保始终获取最新代码：
    - refs/merge-requests/{PR}/head 指向 PR 分支最新提交（每次 push 自动更新）
    - refs/merge-requests/{PR}/merge 指向虚拟合并提交（可能延迟更新，仅作 fallback）

    Args:
        repo_url: 仓库链接（.git 格式）
        pr_number: PR 编号
        file_filter: 文件路径过滤模式（可选）
        stat_only: 是否仅返回统计信息

    Returns:
        str: PR diff 内容

    Raises:
        subprocess.CalledProcessError: 当 git 命令执行失败时抛出
    """
    # 参数验证
    if not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError(f"PR 编号必须是正整数，当前值: {pr_number}")

    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    repo_dir = os.path.join(temp_dir, "repo")

    try:
        logger.info("正在克隆仓库...")
        run_git_command(
            [
                "git",
                "clone",
                "--bare",
                repo_url,
                repo_dir,
            ]
        )

        # 从 repo_url 解析 owner/repo，用于 API 查询 PR 真实目标分支
        owner, repo = parse_repo_url(repo_url)

        # 确定 PR 真实目标分支（API 取真值，支持 9.1.0 等发布分支）
        base_branch = _resolve_base_branch(repo_dir, owner, repo, pr_number)

        # 优先获取 PR head 引用（PR 分支最新提交，每次 push 自动更新）
        # fallback 到 merge 引用（虚拟合并提交，可能延迟更新）
        logger.info("正在获取 PR #%d head 引用...", pr_number)
        head_ref = f"pr_{pr_number}_head"
        use_head_ref = True
        head_fetch_result = run_git_command(
            [
                "git",
                "fetch",
                "origin",
                f"refs/merge-requests/{pr_number}/head:{head_ref}",
            ],
            cwd=repo_dir,
            check=False,
        )
        if head_fetch_result.returncode != 0:
            logger.info("head 引用不存在，fallback 到 merge 引用")
            use_head_ref = False

        logger.info("正在生成 diff...")

        if use_head_ref:
            # 显式计算 merge-base 后用两点 diff，确保只含 PR 实际修改的文件，
            # 不被分支历史上累积的无关变更污染（三点在某些场景下行为不一致）。
            # 用 check=False 捕获"无共同历史"（git merge-base 此时退出码 1、
            # stdout 为空），转为清晰的 RuntimeError 而非 CalledProcessError。
            mb_result = run_git_command(
                ["git", "merge-base", base_branch, head_ref],
                cwd=repo_dir,
                check=False,
            )
            merge_base = mb_result.stdout.strip()
            if mb_result.returncode != 0 or not merge_base:
                raise RuntimeError(
                    f"无法计算 merge-base（{base_branch} 与 {head_ref} 无共同历史）"
                )
            range_spec = f"{merge_base}..{head_ref}"
            if stat_only:
                result = run_git_command(
                    ["git", "diff", "--stat", range_spec],
                    cwd=repo_dir,
                )
            else:
                result = run_git_command(
                    ["git", "diff", range_spec],
                    cwd=repo_dir,
                )
            diff_content = result.stdout
        else:
            merge_ref = f"mr_{pr_number}_merge"
            run_git_command(
                [
                    "git",
                    "fetch",
                    "origin",
                    f"refs/merge-requests/{pr_number}/merge:{merge_ref}",
                ],
                cwd=repo_dir,
            )

            if stat_only:
                result = run_git_command(
                    ["git", "show", "-m", "--stat", merge_ref],
                    cwd=repo_dir,
                )
            else:
                result = run_git_command(
                    ["git", "show", "-m", merge_ref],
                    cwd=repo_dir,
                )
            diff_content = result.stdout
            diff_content = _extract_first_diff(diff_content)

        if file_filter and diff_content and not stat_only:
            diff_content = _apply_file_filter(diff_content, file_filter)

        return diff_content

    finally:
        _cleanup_temp_dir(temp_dir)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="获取 GitCode PR 的 diff 内容（无需 token）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # ops-transformer 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-transformer --pr 3228
  %(prog)s --repo https://gitcode.com/cann/ops-transformer --pr 3228 --stat

  # ops-math 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-math --pr 123 --output pr_123.diff

  # ops-nn 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-nn --pr 456 --file-filter "*.asc"

  # ops-cv 仓库
  %(prog)s --repo https://gitcode.com/cann/ops-cv --pr 789 --stat --verbose
        """,
    )
    parser.add_argument(
        "--repo", required=True, help="仓库链接，如 https://gitcode.com/owner/repo"
    )
    parser.add_argument("--pr", required=True, type=int, help="PR 编号")
    parser.add_argument("--output", help="输出文件路径（默认输出到 stdout）")
    parser.add_argument(
        "--file-filter", help="文件路径过滤，支持通配符（如 *.asc、**/*.py）"
    )
    parser.add_argument("--stat", action="store_true", help="仅显示变更统计信息")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    return parser


def validate_and_get_repo_url(repo_url_str: str) -> tuple[str, str, str]:
    """验证并构建仓库 URL

    Args:
        repo_url_str: 用户提供的仓库链接字符串

    Returns:
        tuple[str, str, str]: (owner, repo, 完整的 .git URL)

    Raises:
        ValueError: 当 URL 格式不正确时抛出
    """
    owner, repo = parse_repo_url(repo_url_str)
    repo_url = f"https://{ALLOWED_GITCODE_DOMAIN}/{owner}/{repo}.git"
    return owner, repo, repo_url


def setup_logging(verbose: bool, owner: str, repo: str, pr_number: int) -> None:
    """设置日志级别

    Args:
        verbose: 是否显示详细日志
        owner: 仓库 owner
        repo: 仓库名称
        pr_number: PR 编号
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("仓库: %s/%s", owner, repo)
        logger.debug("PR: #%d", pr_number)


def write_output(diff_content: str, output_path: str | None, verbose: bool) -> None:
    """输出 diff 结果

    Args:
        diff_content: diff 内容
        output_path: 输出文件路径（None 表示输出到 stdout）
        verbose: 是否显示详细日志
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(diff_content)
        if verbose:
            logger.debug("已写入: %s", output_path)
    else:
        print(diff_content)


def main() -> None:
    """主函数 - 解析命令行参数并获取 PR diff"""
    parser = create_argument_parser()
    args = parser.parse_args()

    owner, repo, repo_url = validate_and_get_repo_url(args.repo)
    setup_logging(args.verbose, owner, repo, args.pr)

    try:
        diff_content = get_pr_diff_git(
            repo_url=repo_url,
            pr_number=args.pr,
            file_filter=args.file_filter,
            stat_only=args.stat,
        )
        if not diff_content:
            logger.info("未找到变更或 diff 为空")
            sys.exit(0)

        write_output(diff_content, args.output, args.verbose)
    except Exception as e:
        logger.error("获取 diff 失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
