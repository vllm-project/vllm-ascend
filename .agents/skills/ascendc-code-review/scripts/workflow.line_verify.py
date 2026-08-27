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
行号校对工具 — 扫描 yaml 目录，原地修正行号

由主 Agent 在阶段2调用。扫描指定目录下所有 yaml 文件（clause-review 与
design-check 两类输出），对 FAIL/SUSPICIOUS 项做行号校对，原地修改 yaml
中的行号字段。

拆分路由逻辑（保持现有 pr-review.line-verify.md + common.line-verify.md 行为）:
  - clause 类 yaml:
      * PR 模式（--diff 必传）: 先做 diff 范围红线校验，关键代码行不在 diff
        变更范围内 → 标记 out_of_range=true（报告阶段归入范围外备注）
      * 文件检视模式（--diff 不传）: 仅做行号校对
  - design 类 yaml:
      * 无 diff 红线（设计偏差常指向未变更代码，不做范围过滤）
      * 仅校对 deviations / doc_violations 的行号

用法:
  文件检视:
    python3 workflow.line_verify.py --dir /tmp/file_xxx --repo /path/to/source
  PR 检视:
    python3 workflow.line_verify.py --dir /tmp/pr1234_xxx --diff /path/to.diff \\
        --repo /path/to/repo

输出:
  - yaml 文件原地更新行号字段（start_line/end_line/code_location/violation_location）
  - stdout 打印校对摘要（处理 yaml 数、FAIL/SUSPICIOUS 项数、out_of_range 数、
    行号修正数、待确认数）
退出码: 0=成功, 1=参数错误, 2=目录读取失败
"""

import argparse
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


# ─── diff 解析 ────────────────────────────────────────────────────

class DiffHunk:
    """单个 hunk 的变更信息"""

    def __init__(self, file_path: str, new_start: int, new_len: int,
                 added_lines: List[str], deleted_lines: List[str]):
        self.file_path = file_path
        self.new_start = new_start
        self.new_len = new_len
        self.added_lines = added_lines  # diff 中以 '+' 开头的行内容（去掉 '+' 前缀）
        self.deleted_lines = deleted_lines  # diff 中以 '-' 开头的行内容（去掉 '-' 前缀）


def _save_current_hunk(hunks_by_file, state):
    """保存当前 hunk 到 hunks_by_file（如有），返回 reset 后的 in_hunk"""
    if state["in_hunk"] and state["file"] is not None:
        added = state["added"]
        deleted = state["deleted"]
        new_len = state["new_len"]
        # 简单校验：added 行数不应超过 new_len（hunk 声明的新文件行数）
        if new_len > 0 and len(added) > new_len:
            logger.warning(
                "hunk 解析异常: added 行数(%d)超过 new_len(%d), file=%s",
                len(added), new_len, state["file"],
            )
        hunks_by_file.setdefault(state["file"], []).append(
            DiffHunk(state["file"], state["new_start"], new_len, added, deleted)
        )
    return False


def _parse_hunk_header(line: str) -> Optional[tuple]:
    """解析 hunk 头，返回 (new_start, new_len) 或 None"""
    m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
    if not m:
        return None
    new_start = int(m.group(1))
    new_len = int(m.group(2)) if m.group(2) else 1
    return (new_start, new_len)


def _process_diff_line(line: str, state: dict, hunks_by_file):
    """处理单行 diff，更新 state 或保存 hunk"""
    # 文件头: +++ b/path 或 +++ a/path 或 +++ path（--no-prefix 格式）
    m = re.match(r"^\+\+\+ (?:[ab]/)?(.+)$", line)
    if m:
        state["in_hunk"] = _save_current_hunk(hunks_by_file, state)
        state["file"] = m.group(1).strip()
        state["added"] = []
        state["deleted"] = []
        return

    # hunk 头
    hunk = _parse_hunk_header(line)
    if hunk:
        state["in_hunk"] = _save_current_hunk(hunks_by_file, state)
        state["new_start"] = hunk[0]
        state["new_len"] = hunk[1]
        state["added"] = []
        state["deleted"] = []
        state["in_hunk"] = True
        return

    if state["in_hunk"]:
        # added 行：以 + 开头，但不是文件头（+++ b/ 或 +++ a/ 或 +++ path）
        if line.startswith("+") and not re.match(r"^\+\+\+ (?:[ab]/)?\S", line):
            state["added"].append(line[1:].rstrip("\n"))
        elif line.startswith("-") and not line.startswith("---"):
            state["deleted"].append(line[1:].rstrip("\n"))
        elif line.startswith(" "):
            pass  # 上下文行
        # hunk 结束的判定靠下一个 @@ 或下一个文件头


def parse_diff(diff_path: str) -> Dict[str, List[DiffHunk]]:
    """解析 unified diff，返回 {file_path: [DiffHunk]}"""
    if not diff_path or not os.path.isfile(diff_path):
        return {}

    hunks_by_file: Dict[str, List[DiffHunk]] = {}
    state = {"file": None, "added": [], "deleted": [], "new_start": 0, "new_len": 0, "in_hunk": False}

    with open(diff_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            _process_diff_line(line, state, hunks_by_file)

    # 保存最后一个 hunk
    _save_current_hunk(hunks_by_file, state)

    return hunks_by_file


def is_code_in_diff(code_lines: List[str], hunks: List[DiffHunk]) -> bool:
    """判断代码片段的关键行是否出现在 diff 变更范围内（+行或-行）。

    取代码片段中若干关键行（非空、非纯大括号、非纯注释），检查是否在 diff
    任一 hunk 的 added_lines（新增行）或 deleted_lines（删除行）中出现。
    任一关键行命中 → True。

    检查范围覆盖新增行和删除行，确保删除型/化简型 PR 中引用被删代码的发现
    不会被误判为 out_of_range。
    """
    if not hunks:
        return False

    # 收集所有 hunk 的变更行（added + deleted）
    all_changed = set()
    for h in hunks:
        for ln in h.added_lines:
            stripped = ln.strip()
            if stripped:
                all_changed.add(stripped)
        for ln in h.deleted_lines:
            stripped = ln.strip()
            if stripped:
                all_changed.add(stripped)

    # 提取关键行
    key_lines = []
    for ln in code_lines:
        s = ln.strip()
        if not s:
            continue
        # 跳过纯大括号 / 纯分号
        if s in ("{", "}", "};", "})"):
            continue
        # 跳过纯注释行
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        key_lines.append(s)

    if not key_lines:
        # 没有关键行可判断，保守认为在范围内
        return True

    for kl in key_lines:
        if kl in all_changed:
            return True
    return False


# ─── 源码行号定位 ────────────────────────────────────────────────

def normalize_line(s: str, is_diff: bool = False) -> str:
    """归一化单行：去首尾空白。is_diff=True 时额外去 diff 标记前缀（+/-）"""
    s = s.strip()
    if not is_diff or not s:
        return s
    if s[0] in ("+", "-"):
        s = s[1:].strip()
    return s


def split_and_normalize(code: str) -> List[str]:
    """将代码片段拆行并归一化，丢弃空行，返回非空归一化行列表"""
    if not code:
        return []
    result = []
    for raw in code.splitlines():
        n = normalize_line(raw, is_diff=False)
        if n:
            result.append(n)
    return result


def _normalize_file_lines(f) -> List[tuple]:
    """遍历文件行，归一化并跳过空行，返回 [(line_num, normalized_content), ...]"""
    result = []
    for idx, raw in enumerate(f, start=1):
        n = normalize_line(raw, is_diff=False)
        if n:
            result.append((idx, n))
    return result


def _load_normalized_lines(file_path: str) -> Optional[List[tuple]]:
    """加载文件并归一化，返回 [(line_num, normalized_content), ...]（跳过空行）"""
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return _normalize_file_lines(f)
    except Exception:
        return None


def _match_all_consecutive(
    file_lines: List[tuple],
    target_lines: List[str],
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> List[tuple]:
    """在 file_lines 中搜索所有连续匹配 target_lines 的位置。

    返回 [(start_line, end_line), ...]，空列表表示无匹配。
    """
    if not target_lines or len(file_lines) < len(target_lines):
        return []

    search_lines = file_lines
    if window_start is not None and window_end is not None:
        search_lines = [
            (ln, c) for ln, c in file_lines
            if window_start <= ln <= window_end
        ]

    if len(search_lines) < len(target_lines):
        return []

    matches = []
    for i in range(len(search_lines) - len(target_lines) + 1):
        matched = True
        for j, target in enumerate(target_lines):
            if search_lines[i + j][1] != target:
                matched = False
                break
        if matched:
            start = search_lines[i][0]
            end = search_lines[i + len(target_lines) - 1][0]
            matches.append((start, end))
    return matches


def locate_snippet_in_file(
    file_path: str,
    code: str,
    hint_start: Optional[int] = None,
    window: int = 50,
) -> Optional[tuple]:
    """在源文件中定位代码片段的行号范围。

    策略：
      1. 若有 hint_start（子 Agent 写的 start_line），优先在 ±window 范围内搜索
         - 窗口内命中（1处或多处）→ 取第一个，窗口内可信赖
      2. 窗口内未命中 → 全文件搜索
         - 恰好 1 处匹配 → 返回该位置
         - 多处匹配 → 返回 None（无法确定正确位置，标记 unconfirmed）
      3. 全文件也无匹配 → 返回 None

    返回 (start_line, end_line) 或 None。
    """
    target_lines = split_and_normalize(code)
    if not target_lines:
        return None

    file_lines = _load_normalized_lines(file_path)
    if file_lines is None:
        return None

    # 策略1：窗口优先
    if hint_start and hint_start > 0:
        w_start = max(1, hint_start - window)
        w_end = hint_start + window + len(target_lines)
        matches = _match_all_consecutive(file_lines, target_lines, w_start, w_end)
        if matches:
            return matches[0]

    # 策略2：全文件搜索
    matches = _match_all_consecutive(file_lines, target_lines)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning("snippet 在文件中多处匹配（%d 处），标记待确认", len(matches))
    return None


def resolve_file_path(file_path: str, repo_path: str) -> str:
    """将 yaml 中的 file_path 解析为可读的绝对路径。

    优先级:
      1. 若 file_path 是绝对路径且存在 → 直接用
      2. 若 repo_path 非空，拼接 repo_path/file_path
      3. 否则原样返回
    """
    if not file_path:
        return file_path
    if os.path.isabs(file_path) and os.path.isfile(file_path):
        return file_path
    if repo_path:
        candidate = os.path.join(repo_path, file_path)
        if os.path.isfile(candidate):
            return candidate
    return file_path


# ─── clause yaml 校对 ────────────────────────────────────────────

def verify_clause_yaml(
    data: dict,
    diff_hunks_by_file: Dict[str, List[DiffHunk]],
    repo_path: str,
    is_pr_mode: bool,
) -> Tuple[int, int, int, int]:
    """校对单个 clause yaml。返回 (fail_count, out_of_range_count, line_fixed, line_unconfirmed)"""
    status = data.get("status")
    if status not in ("FAIL", "SUSPICIOUS"):
        return (0, 0, 0, 0)

    fail_count = 1
    snippet = data.get("code_snippet")
    if not snippet or not isinstance(snippet, dict):
        return (fail_count, 0, 0, 0)

    file_path = snippet.get("file_path", "")
    code = snippet.get("code", "")
    if not isinstance(code, str):
        code = str(code) if code else ""

    # PR 模式: diff 范围红线校验
    out_of_range_count = 0
    if is_pr_mode:
        if not diff_hunks_by_file:
            # diff 解析失败/为空，fail-safe 标记所有发现为范围外
            data["out_of_range"] = True
            data["_dirty"] = True
            out_of_range_count = 1
            logger.warning("diff 红线校验跳过（diff 解析为空），标记范围外: %s", file_path)
        else:
            hunks = diff_hunks_by_file.get(file_path, [])
            if not hunks:
                # file_path 在 diff 中未命中（可能路径不一致），fail-safe 标记范围外
                data["out_of_range"] = True
                data["_dirty"] = True
                out_of_range_count = 1
                logger.warning("diff 红线校验跳过（file_path 未命中 diff）: %s", file_path)
            elif not is_code_in_diff(code.splitlines() if code else [], hunks):
                data["out_of_range"] = True
                data["_dirty"] = True
                out_of_range_count = 1
                # 范围外发现仍尝试校对行号，但不强制

    # 行号校对: 连续匹配整个 snippet 定位行号
    abs_path = resolve_file_path(file_path, repo_path)
    hint_start = snippet.get("start_line")
    if not isinstance(hint_start, int):
        try:
            hint_start = int(hint_start) if hint_start else None
        except (ValueError, TypeError):
            hint_start = None

    result = locate_snippet_in_file(abs_path, code, hint_start=hint_start)

    if result is not None:
        new_start, new_end = result
        old_start = snippet.get("start_line")
        line_fixed = 1 if old_start != new_start else 0
        snippet["start_line"] = new_start
        snippet["end_line"] = new_end
        snippet["line_verified"] = True
        data["_dirty"] = True
        return (fail_count, out_of_range_count, line_fixed, 0)
    else:
        snippet["line_verified"] = False
        data["_dirty"] = True
        return (fail_count, out_of_range_count, 0, 1)


# ─── design yaml 校对 ────────────────────────────────────────────

def verify_design_yaml(
    data: dict,
    repo_path: str,
) -> Tuple[int, int, int]:
    """校对 design yaml。返回 (deviation_count, line_fixed, line_unconfirmed)"""
    line_fixed = 0
    line_unconfirmed = 0
    deviation_count = 0
    dirty = False

    # 校对 deviations（S1-S7 的 ❌ 项）
    deviations = data.get("deviations", []) or []
    for dev in deviations:
        deviation_count += 1
        fixed_line = _fix_location(dev, "code_location", repo_path)
        if fixed_line is True:
            line_fixed += 1
            dirty = True
        elif fixed_line is False:
            line_unconfirmed += 1
            dirty = True

    # 校对 doc_violations（D8 的 ❌ 项）
    doc_violations = data.get("doc_violations", []) or []
    for dv in doc_violations:
        deviation_count += 1
        # D8 违规位置可能在文档文件中，也传 repo_path 以支持相对路径
        fixed_line = _fix_location(dv, "violation_location", repo_path)
        if fixed_line is True:
            line_fixed += 1
            dirty = True
        elif fixed_line is False:
            line_unconfirmed += 1
            dirty = True

    if dirty:
        data["_dirty"] = True
    return (deviation_count, line_fixed, line_unconfirmed)


def _parse_location(loc: str) -> tuple:
    """解析定位字符串，返回 (file_path, start_line, end_line)。

    支持格式：
      path              → (path, None, None)
      path:line         → (path, line, line)
      path:start-end    → (path, start, end)
      path:l1,l2,l3     → (path, l1, l3)  取首尾
    """
    parts = loc.rsplit(":", 1)
    if len(parts) != 2:
        return (loc, None, None)

    file_path, line_str = parts

    # 范围格式: start-end
    if "-" in line_str:
        range_parts = line_str.split("-", 1)
        try:
            start = int(range_parts[0])
            end = int(range_parts[1])
            return (file_path, start, end)
        except ValueError:
            return (file_path, None, None)

    # 逗号多行格式: l1,l2,l3
    if "," in line_str:
        nums = []
        for p in line_str.split(","):
            try:
                nums.append(int(p))
            except ValueError:
                pass
        if nums:
            return (file_path, nums[0], nums[-1])
        return (file_path, None, None)

    # 单行格式: line
    try:
        line = int(line_str)
        return (file_path, line, line)
    except ValueError:
        return (file_path, None, None)


def _fix_location(item: dict, field: str, repo_path: str) -> Optional[bool]:
    """校验 item[field] 中的行号是否在合理范围内。

    支持格式: 'path'、'path:line'、'path:start-end'、'path:l1,l2,l3'

    design yaml 没有完整代码片段，无法做内容匹配，仅做粗略校验：
    检查文件是否存在 + 行号是否在文件行数范围内。

    返回 True=修正了, False=无法定位, None=无需修正/在合理范围内
    """
    loc = item.get(field, "")
    if not loc:
        return None

    file_path, start_line, end_line = _parse_location(loc)
    if start_line is None:
        logger.warning("行号格式不支持: %s=%s", field, loc)
        item["line_verified"] = False
        return False

    abs_path = resolve_file_path(file_path, repo_path)
    if not os.path.isfile(abs_path):
        return False

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)
        if 1 <= start_line <= total_lines and (end_line is None or 1 <= end_line <= total_lines):
            item["line_verified"] = True
            return None  # 行号在合理范围内，未修改
        else:
            item["line_verified"] = False
            return False
    except Exception:
        return False


# ─── 主流程 ──────────────────────────────────────────────────────

def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="行号校对工具 — 扫描 yaml 目录原地修正行号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, help="yaml 输出目录路径")
    parser.add_argument("--diff", default="", help="diff 文件路径（PR 模式必传，文件检视不传）")
    parser.add_argument(
        "--repo", default="",
        help="完整源码路径（PR 模式必传；文件检视可传源码根目录用于行号定位）",
    )
    return parser.parse_args()


def _process_single_yaml_file(fpath, fname, stats, ctx):
    """处理单个 yaml 文件：加载、校对、写回

    ctx: dict 含 diff_hunks_by_file / repo_path / is_pr_mode
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("跳过无法解析的 yaml: %s (%s)", fname, e)
        return

    if not isinstance(data, dict):
        return

    yaml_type = data.get("type", "clause")
    stats["total_yaml"] += 1

    if yaml_type == "design":
        dev_count, fixed, unconfirmed = verify_design_yaml(data, ctx["repo_path"])
        stats["total_deviations"] += dev_count
        stats["total_line_fixed"] += fixed
        stats["total_line_unconfirmed"] += unconfirmed
    else:
        fail, oor, fixed, unconfirmed = verify_clause_yaml(
            data, ctx["diff_hunks_by_file"], ctx["repo_path"], ctx["is_pr_mode"]
        )
        stats["total_fail"] += fail
        stats["total_out_of_range"] += oor
        stats["total_line_fixed"] += fixed
        stats["total_line_unconfirmed"] += unconfirmed

    # 仅当字段被修改时才写回（避免丢失 PASS 条例的原文格式）
    if data.pop("_dirty", False):
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        except Exception as e:
            logger.warning("写回 yaml 失败: %s (%s)", fname, e)


def _process_yaml_dir(yaml_dir, diff_hunks_by_file, repo_path, is_pr_mode):
    """遍历 yaml 目录，逐文件校对并写回。返回统计 dict。"""
    stats = dict(total_yaml=0, total_fail=0, total_out_of_range=0,
                 total_line_fixed=0, total_line_unconfirmed=0, total_deviations=0)
    ctx = {"diff_hunks_by_file": diff_hunks_by_file, "repo_path": repo_path, "is_pr_mode": is_pr_mode}

    for fname in sorted(os.listdir(yaml_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        fpath = os.path.join(yaml_dir, fname)
        if not os.path.isfile(fpath):
            continue
        _process_single_yaml_file(fpath, fname, stats, ctx)

    return stats


def main() -> int:
    args = _parse_args()

    yaml_dir = args.dir
    if not os.path.isdir(yaml_dir):
        logger.error("目录不存在: %s", yaml_dir)
        return 2

    is_pr_mode = bool(args.diff)
    diff_hunks_by_file: Dict[str, List[DiffHunk]] = {}
    if is_pr_mode:
        diff_hunks_by_file = parse_diff(args.diff)
        if not diff_hunks_by_file:
            logger.warning("diff 文件解析无结果或文件不存在: %s", args.diff)

    stats = _process_yaml_dir(yaml_dir, diff_hunks_by_file, args.repo, is_pr_mode)

    # 以下为脚本最终产物输出（校对摘要）
    mode_str = "PR 检视" if is_pr_mode else "文件检视"
    logger.info(f"行号校对完成 [{mode_str}]")
    logger.info(f"  yaml 目录: {yaml_dir}")
    logger.info(f"  处理 yaml: {stats['total_yaml']} 个")
    logger.info(f"  FAIL/SUSPICIOUS 项: {stats['total_fail']} 个")
    if is_pr_mode:
        logger.info(f"  范围外(out_of_range): {stats['total_out_of_range']} 个")
    logger.info(f"  设计偏差项: {stats['total_deviations']} 个")
    logger.info(f"  行号修正: {stats['total_line_fixed']} 处")
    logger.info(f"  行号待确认: {stats['total_line_unconfirmed']} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())
