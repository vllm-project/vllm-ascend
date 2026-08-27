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
报告拼接工具 — 读取 yaml 目录，组装 md 报告正文

由主 Agent 在阶段3（common.report-write.md 薄壳内）调用。读取指定目录下
所有 yaml 文件（clause-review 与 design-check 两类输出），按报告格式
组装成完整 md 报告。

报告头部元信息（代码文件、侧别、检视文档列表、总条例数、设计文档来源、
检视时间）由主 Agent 自行填充，脚本在头部章节使用占位符 {{KEY}}。

分流逻辑:
  - clause 类 + category=clause → 检视统计表 + 分级发现章节（HIGH/MED/LOW）
  - clause 类 + category=style → 代码风格章节（不进统计表）
  - clause 类 + out_of_range=true → 范围外备注章节（计入统计，单独章节展示）
  - design 类 → 设计一致性检查章节（S1-S7 + D8 判定表 + ❌项详情）

用法:
  python3 workflow.assemble_report.py --dir /tmp/pr1234_xxx \\
      --output /path/to/report.md

输出:
  - 写入报告 md 文件到 --output 指定路径
  - stdout 打印报告路径 + 统计摘要
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


def _md_cell(val) -> str:
    """转义 markdown 表格单元格中的 | 字符，防止破坏表格结构"""
    if val is None:
        return ""
    return str(val).replace("|", "\\|")


def _parse_confidence_pct(conf_val) -> int:
    """从 confidence_value 字段解析出整数百分比，用于判断是否 ≥70%"""
    s = str(conf_val).strip().replace("%", "").replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def _render_evidence_row(e: dict) -> str:
    """渲染单行证据表格"""
    return f"| {_md_cell(e.get('type', ''))} | {_md_cell(e.get('score', ''))} | {_md_cell(e.get('desc', ''))} |"


# ─── 置信度校验 ──────────────────────────────────────────────────

def parse_score(score_str) -> int:
    """解析证据分值字符串（如 '+40%'、'-15%'）为整数"""
    if score_str is None:
        return 0
    s = str(score_str).strip().replace("%", "").replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def verify_confidence(d: dict, fname: str = "") -> bool:
    """校验并修正 clause yaml 的 confidence_value。

    正向证据分值求和 + 负向证据分值求和 = 正确自信值。
    若与 yaml 中写的 confidence_value 不一致 → 原地修正。
    返回 True=已修正, False=无需修正。
    """
    evidence = d.get("evidence")
    if not evidence:
        return False

    # 严格校验：evidence 必须是 dict
    if not isinstance(evidence, dict):
        raise ValueError(
            f"{fname}: evidence must be a mapping "
            f"(positive/negative/confidence_value), "
            f"got {type(evidence).__name__}: {repr(evidence)[:100]}"
        )

    positive = evidence.get("positive", []) or []
    negative = evidence.get("negative", []) or []

    # 严格校验：positive/negative 必须是 list
    if not isinstance(positive, list):
        raise ValueError(f"{fname}: evidence.positive must be a list, got {type(positive).__name__}")
    if not isinstance(negative, list):
        raise ValueError(f"{fname}: evidence.negative must be a list, got {type(negative).__name__}")

    total = 0
    for e in positive:
        if not isinstance(e, dict):
            raise ValueError(f"{fname}: evidence.positive items must be mappings, got {type(e).__name__}")
        total += parse_score(e.get("score", ""))
    for e in negative:
        if not isinstance(e, dict):
            raise ValueError(f"{fname}: evidence.negative items must be mappings, got {type(e).__name__}")
        total += parse_score(e.get("score", ""))

    correct_value = f"{total}%"
    current_value = str(evidence.get("confidence_value", "")).strip()

    if current_value != correct_value:
        evidence["confidence_value"] = correct_value
        return True
    return False


# ─── yaml 加载 ───────────────────────────────────────────────────

def _validate_clause_schema(data: dict, fname: str, schema_errors: List[str]):
    """校验 FAIL/SUSPICIOUS clause 的 code_snippet 结构"""
    status = data.get("status", "")
    if status not in ("FAIL", "SUSPICIOUS"):
        return
    snip = data.get("code_snippet")
    if snip is not None and not isinstance(snip, dict):
        schema_errors.append(
            f"{fname}: code_snippet must be a mapping "
            f"(file_path/start_line/end_line/code), "
            f"got {type(snip).__name__}: {repr(snip)[:100]}"
        )


def _writeback_yaml(data: dict, fpath: str, fname: str):
    """将修改后的 data 写回 yaml 文件"""
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except Exception as e:
        logger.warning("置信度修正写回失败: %s (%s)", fname, e)


def _fix_confidence_and_writeback(data: dict, fname: str, fpath: str, schema_errors: List[str]) -> int:
    """校验置信度并写回磁盘，返回 confidence_fixed (0 或 1)"""
    try:
        if verify_confidence(data, fname):
            _writeback_yaml(data, fpath, fname)
            return 1
    except ValueError as e:
        schema_errors.append(str(e))
    return 0


def _process_single_yaml(fname: str, fpath: str, schema_errors: List[str]) -> tuple:
    """加载单个 yaml 文件，做 schema 校验 + 置信度修正。

    返回 (data, is_design, confidence_fixed)。
    data 为 None 表示跳过该文件。
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("跳过无法解析的 yaml: %s (%s)", fname, e)
        return (None, False, 0)
    if not isinstance(data, dict):
        return (None, False, 0)

    if data.get("type") == "design":
        return (data, True, 0)

    _validate_clause_schema(data, fname, schema_errors)
    confidence_fixed = _fix_confidence_and_writeback(data, fname, fpath, schema_errors)

    return (data, False, confidence_fixed)


def load_yaml_files(yaml_dir: str) -> Tuple[List[dict], Optional[dict]]:
    """加载目录下所有 yaml，返回 (clause_list, design_data)。

    加载时对 clause 类 yaml 做置信度校验：若 evidence 中各证据分值之和与
    confidence_value 不一致，原地修正为正确值。
    """
    clause_list: List[dict] = []
    design_data: Optional[dict] = None
    confidence_fixed = 0
    schema_errors: List[str] = []

    for fname in sorted(os.listdir(yaml_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        fpath = os.path.join(yaml_dir, fname)
        if not os.path.isfile(fpath):
            continue

        data, is_design, cfixed = _process_single_yaml(fname, fpath, schema_errors)
        if data is None:
            continue
        confidence_fixed += cfixed

        if is_design:
            design_data = data
        else:
            clause_list.append(data)

    # schema 校验失败：列出所有问题后抛异常（由 main 捕获）
    if schema_errors:
        logger.error("%d 个 yaml schema 校验失败:", len(schema_errors))
        for err in schema_errors:
            logger.error("  %s", err)
        raise ValueError(f"{len(schema_errors)} 个 yaml schema 校验失败")

    if confidence_fixed > 0:
        logger.info("置信度校验修正 %d 个 yaml 的 confidence_value", confidence_fixed)

    return (clause_list, design_data)


# ─── 统计 ────────────────────────────────────────────────────────

def compute_stats(clause_list: List[dict]) -> Tuple[int, int, int, int]:
    """计算统计：返回 (total, pass, fail, suspicious)

    不含 style 类，含 out_of_range。
    """
    total = 0
    pass_n = 0
    fail_n = 0
    susp_n = 0
    for d in clause_list:
        if d.get("category") == "style":
            continue
        total += 1
        status = d.get("status", "")
        if status == "PASS":
            pass_n += 1
        elif status == "FAIL":
            fail_n += 1
        elif status == "SUSPICIOUS":
            susp_n += 1
    return (total, pass_n, fail_n, susp_n)


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n * 100 / total:.1f}%"


# ─── clause 条目渲染 ────────────────────────────────────────────

def render_evidence_table(evidence: dict) -> str:
    """渲染假设检验证据表"""
    if not evidence:
        return ""

    lines = []
    positive = evidence.get("positive", []) or []
    negative = evidence.get("negative", []) or []
    conf_val = evidence.get("confidence_value", "")

    if positive:
        lines.append("正向证据：")
        lines.append("| 证据类型 | 分值 | 证据描述 |")
        lines.append("|---------|------|---------|")
        for e in positive:
            row = _render_evidence_row(e)
            lines.append(row)

    if negative:
        lines.append("")
        lines.append("负向证据：")
        lines.append("| 证据类型 | 分值 | 证据描述 |")
        lines.append("|---------|------|---------|")
        for e in negative:
            row = _render_evidence_row(e)
            lines.append(row)

    if conf_val:
        lines.append("")
        verdict = "判定违规" if _parse_confidence_pct(conf_val) >= 70 else "未达违规阈值"
        lines.append(f"自信值 = Σ正向 + Σ负向 = {conf_val} ≥ 70% → {verdict}")
    return "\n".join(lines)


def _infer_fence_lang(file_path: str) -> str:
    """按文件后缀推断代码块语言标签"""
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    lang_map = {
        ".py": "python",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".c": "c",
        ".cu": "cpp",
        ".asc": "cpp",
        ".cce": "cpp",
    }
    return lang_map.get(ext, "")


def render_code_snippet(snippet) -> str:
    """渲染代码片段块（full 模式）：路径/起始行号/中止行号各行独立 + 代码片段"""
    if not snippet or not isinstance(snippet, dict):
        return ""
    file_path = snippet.get("file_path", "")
    start = snippet.get("start_line", "")
    end = snippet.get("end_line", "")
    code = snippet.get("code", "")
    if not isinstance(code, str):
        code = str(code) if code else ""
    verified = snippet.get("line_verified")

    lines = [f"- **代码文件**：{file_path}"]
    lines.append(f"- **起始行号**：{start}")
    lines.append(f"- **中止行号**：{end}")
    if verified is False:
        lines.append("- **行号状态**：待确认")
    lang = _infer_fence_lang(file_path)
    lines.append(f"- **代码片段**：")
    lines.append(f"  ```{lang}")
    for code_line in code.splitlines():
        lines.append(f"  {code_line}")
    lines.append(f"  ```")
    return "\n".join(lines)


def render_code_location(snippet) -> str:
    """渲染代码位置（simple 模式）：路径/起始行号/中止行号各行独立，不含代码片段"""
    if not snippet or not isinstance(snippet, dict):
        return ""
    file_path = snippet.get("file_path", "")
    start = snippet.get("start_line", "")
    end = snippet.get("end_line", "")
    verified = snippet.get("line_verified")

    lines = [f"- **代码文件**：{file_path}"]
    lines.append(f"- **起始行号**：{start}")
    lines.append(f"- **中止行号**：{end}")
    if verified is False:
        lines.append("- **行号状态**：待确认")
    return "\n".join(lines)


def render_clause_finding(d: dict, mode: str = "simple") -> str:
    """渲染单个 FAIL/SUSPICIOUS 发现

    mode:
      - simple（默认）: 仅位置信息 + 置信度 + 问题描述 + 修复建议，不含代码片段和证据列表
      - full: 完整渲染（含代码片段 + 假设检验证据表）
    """
    lines = []
    clause_id = d.get("clause_id", "")
    title = d.get("clause_title", "")
    confidence = d.get("confidence", "")
    status = d.get("status", "")

    lines.append(f"### [{clause_id}] {title}")
    lines.append(f"- **状态**：{status} | **置信度**：{confidence}")
    lines.append(f"- **问题描述**：{d.get('problem_desc', '')}")

    snippet = d.get("code_snippet")
    if snippet and isinstance(snippet, dict):
        if mode == "full":
            lines.append(render_code_snippet(snippet))
        else:
            lines.append(render_code_location(snippet))

    evidence = d.get("evidence")
    if evidence and mode == "full":
        lines.append(f"- **假设检验证据**：")
        lines.append("")
        lines.append(render_evidence_table(evidence))

    fix = d.get("fix_suggestion", "")
    if fix:
        lines.append(f"- **修复建议**：{fix}")

    return "\n".join(lines)


def render_style_finding(d: dict) -> str:
    """渲染单个 style FAIL 条目（表格行）"""
    clause_id = d.get("clause_id", "")
    title = d.get("clause_title", "")
    # 从快速索引推断严重级别（style 条例无 confidence 字段，用 status + 原严重级别）
    # yaml 中未存严重级别字段，这里用空占位，或从 problem_desc 推断
    severity = d.get("severity", "")
    problem = d.get("problem_desc", "")
    snippet = d.get("code_snippet", {})
    if not isinstance(snippet, dict):
        snippet = {}
    file_path = snippet.get("file_path", "")
    start = snippet.get("start_line", "")
    end = snippet.get("end_line", "")
    loc = f"{file_path}:{start}-{end}" if file_path else ""
    fix = d.get("fix_suggestion", "")
    cells = [_md_cell(clause_id), _md_cell(title), _md_cell(severity),
             _md_cell(problem), _md_cell(loc), _md_cell(fix)]
    return f"| {cells[0]} {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} |"


# ─── design 渲染 ────────────────────────────────────────────────

def _compute_design_rating(strategies: list) -> str:
    """根据 S1-S7 verdict 计算总体评级"""
    verdicts = [s.get("verdict", "") for s in strategies]
    if "❌" in verdicts:
        return "不一致"
    if "⚠️" in verdicts:
        return "部分一致"
    return "一致"


def _render_strategy_row(sid, name, design_desc, impl_desc, verdict) -> str:
    """渲染单行 strategy 判定表"""
    return (
        f"| {_md_cell(sid)} | {_md_cell(name)} | {_md_cell(design_desc)} "
        f"| {_md_cell(impl_desc)} | {_md_cell(verdict)} |"
    )


def _render_design_table(lines: List[str], strategies: list):
    """渲染 S1-S7 + D8 判定表"""
    lines.append("| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |")
    lines.append("|------|------|---------|---------|------|")
    for s in strategies:
        sid = s.get("id", "")
        name = s.get("name", "")
        design_desc = s.get("design_desc", "")
        impl_desc = s.get("impl_desc", "")
        verdict = s.get("verdict", "")
        row = _render_strategy_row(sid, name, design_desc, impl_desc, verdict)
        lines.append(row)
    lines.append("")


def _render_design_deviations(lines: List[str], deviations: list):
    """渲染 S1-S7 ❌ 项详情"""
    if not deviations:
        return
    lines.append("### 设计一致性偏差（❌ 项详情）")
    lines.append("")
    for dev in deviations:
        sid = dev.get("strategy_id", "")
        desc = dev.get("desc", "")
        loc = dev.get("code_location", "")
        basis = dev.get("design_basis", "")
        lines.append(f"**{sid}**：{desc}")
        lines.append(f"- 代码位置（校对后行号）：{loc}")
        lines.append(f"- 设计依据：{basis}")
        lines.append("")


def _render_doc_violations(lines: List[str], doc_violations: list):
    """渲染 D8 文档格式违规"""
    if not doc_violations:
        return
    lines.append("### D8 文档格式违规")
    lines.append("")
    lines.append("| 文档名 | 违规位置 | 违规描述 | 修复建议 |")
    lines.append("|-------|---------|---------|---------|")
    for dv in doc_violations:
        doc_name = dv.get("doc_name", "")
        loc = dv.get("violation_location", "")
        desc = dv.get("desc", "")
        fix = dv.get("fix_suggestion", "")
        lines.append(f"| {_md_cell(doc_name)} | {_md_cell(loc)} | {_md_cell(desc)} | {_md_cell(fix)} |")
    lines.append("")


def render_design_section(design_data: dict) -> str:
    """渲染设计一致性检查章节"""
    lines = ["## 设计一致性检查", ""]
    lines.append("- 文档来源：{{DOCS_INPUT}}")
    lines.append("")

    strategies = design_data.get("strategies", []) or []
    lines.append(f"- 总体评级：{_compute_design_rating(strategies)}")
    lines.append("")

    _render_design_table(lines, strategies)

    deviations = design_data.get("deviations", []) or []
    _render_design_deviations(lines, deviations)

    doc_violations = design_data.get("doc_violations", []) or []
    _render_doc_violations(lines, doc_violations)

    return "\n".join(lines)


# ─── 报告组装 ───────────────────────────────────────────────────

def finding_sort_key(d: dict) -> tuple:
    """发现项排序键：按文件路径分组，同文件内按起始行号排序，使同一文件的多个检视意见连续排列"""
    snippet = d.get("code_snippet", {}) or {}
    file_path = snippet.get("file_path", "")
    start_line = snippet.get("start_line", 0)
    try:
        start_line = int(start_line)
    except (ValueError, TypeError):
        start_line = 0
    return (file_path, start_line)


def _classify_findings(clause_list: List[dict]):
    """将 clause_list 分类为 high/med/low/out_of_range/style 五组

    范围外发现单独收集到 out_of_range_findings（单独章节展示），
    但已计入统计表（compute_stats 不再排除 out_of_range）。
    """
    high_findings = []
    med_findings = []
    low_findings = []
    out_of_range_findings = []
    style_rows = []

    for d in clause_list:
        category = d.get("category", "clause")
        status = d.get("status", "")
        out_of_range = d.get("out_of_range", False)

        if category == "style":
            if status == "PASS":
                continue
            else:
                style_rows.append(render_style_finding(d))
            continue

        if out_of_range:
            out_of_range_findings.append(d)
            continue

        if status == "PASS":
            continue

        confidence = d.get("confidence", "")
        if confidence == "HIGH":
            high_findings.append(d)
        elif confidence == "MED":
            med_findings.append(d)
        elif confidence == "LOW":
            low_findings.append(d)
        else:
            # 无置信度字段，按 status 兜底
            if status == "FAIL":
                high_findings.append(d)
            else:
                med_findings.append(d)

    return high_findings, med_findings, low_findings, out_of_range_findings, style_rows


def _render_header(lines: List[str], clause_list: List[dict]):
    """渲染报告头部 + 检视统计表"""
    lines.append("# 代码检视报告")
    lines.append("")
    lines.append("## 检视概览")
    lines.append("- 代码文件：{{CODE_FILE}}")
    lines.append("- 代码侧别：{{SIDE}}")
    lines.append("- 检视文档：{{DOC_LIST}}")
    lines.append("- 总条例数：{{TOTAL}}")
    lines.append("- 设计文档来源：{{DOCS_INPUT}}")
    lines.append("- 检视时间：{{TIMESTAMP}}")
    lines.append("")

    total, pass_n, fail_n, susp_n = compute_stats(clause_list)
    lines.append("## 检视统计")
    lines.append("| 状态 | 条例数 | 占比 |")
    lines.append("|------|------|------|")
    lines.append(f"| PASS | {pass_n} | {pct(pass_n, total)} |")
    lines.append(f"| FAIL（发现问题）| {fail_n} | {pct(fail_n, total)} |")
    lines.append(f"| SUSPICIOUS（需关注）| {susp_n} | {pct(susp_n, total)} |")
    lines.append("")


def _render_findings(lines: List[str], findings: List[dict], title: str, mode: str):
    """渲染分级发现章节（HIGH/MED/LOW）"""
    if not findings:
        return
    lines.append(f"## {title}")
    lines.append("")
    for d in findings:
        lines.append(render_clause_finding(d, mode))
        lines.append("")


def _render_out_of_range(lines: List[str], findings: List[dict], mode: str = "full"):
    """渲染范围外备注章节（格式与正文发现一致）"""
    if not findings:
        return
    lines.append("## 范围外备注（PR diff 未覆盖）")
    lines.append("")
    lines.append("> 以下发现引用的代码不在本次 PR diff 变更范围内，但仍有参考价值。")
    lines.append("")
    for d in findings:
        lines.append(render_clause_finding(d, mode))
        lines.append("")


def _render_style(lines: List[str], style_rows: List[str], clause_list: List[dict]):
    """渲染代码风格章节"""
    if style_rows:
        lines.append("## 代码风格")
        lines.append("")
        lines.append("> 来自 cpp-style 检视，不走假设检验，违反即 FAIL。不并入上方统计表。")
        lines.append("")
        lines.append("| 条例 | 严重级别 | 问题描述 | 代码位置（校对后行号） | 修复建议 |")
        lines.append("|------|---------|---------|----------------------|---------|")
        for row in style_rows:
            lines.append(row)
        lines.append("")
    else:
        # 若有 style PASS 但无 FAIL，仍显示「全部符合」
        style_pass = [d for d in clause_list if d.get("category") == "style" and d.get("status") == "PASS"]
        if style_pass:
            lines.append("## 代码风格")
            lines.append("")
            lines.append("全部符合代码风格规范。")
            lines.append("")


def assemble_report(clause_list: List[dict], design_data: Optional[dict], mode: str = "full") -> str:
    """组装完整报告 md

    mode:
      - full（默认）: 完整渲染（含代码片段 + 假设检验证据表）
      - simple: clause 发现仅含位置信息（文件路径/起始行号/中止行号）+ 置信度 + 问题描述 + 修复建议
    """
    lines: List[str] = []

    # 头部 + 检视统计表
    _render_header(lines, clause_list)

    # 设计一致性检查章节（条件）
    if design_data is not None:
        lines.append(render_design_section(design_data))

    # 分级 clause 发现
    high, med, low, oor, style_rows = _classify_findings(clause_list)

    # 按文件路径分组排序，使同一文件的多个检视意见连续排列
    high.sort(key=finding_sort_key)
    med.sort(key=finding_sort_key)
    low.sort(key=finding_sort_key)
    oor.sort(key=finding_sort_key)

    _render_findings(lines, high, "发现问题（HIGH 置信度）", mode)
    _render_findings(lines, med, "需关注（MED 置信度）", mode)
    _render_findings(lines, low, "疑似（LOW 置信度）", mode)
    _render_out_of_range(lines, oor, mode)
    _render_style(lines, style_rows, clause_list)

    return "\n".join(lines)


# ─── 主流程 ──────────────────────────────────────────────────────

def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="报告拼接工具 — yaml 目录 → md 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, help="yaml 输出目录路径")
    parser.add_argument("--output", required=True, help="报告 md 输出路径")
    parser.add_argument(
        "--mode", default="full", choices=["simple", "full"],
        help="报告模式: simple=仅位置信息+置信度, full=完整渲染（含代码片段+证据表，默认）",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    yaml_dir = args.dir
    if not os.path.isdir(yaml_dir):
        logger.error("目录不存在: %s", yaml_dir)
        return 2

    try:
        clause_list, design_data = load_yaml_files(yaml_dir)
    except ValueError:
        return 1

    report_md = assemble_report(clause_list, design_data, mode=args.mode)

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logger.error("创建输出目录失败: %s", e)
            return 2

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_md)
    except Exception as e:
        logger.error("写报告失败: %s", e)
        return 2

    # 以下为脚本最终产物输出（报告统计摘要）
    total, pass_n, fail_n, susp_n = compute_stats(clause_list)
    style_count = sum(1 for d in clause_list if d.get("category") == "style")
    oor_count = sum(1 for d in clause_list if d.get("out_of_range"))
    has_design = "是" if design_data else "否"

    logger.info("报告拼接完成")
    logger.info(f"  yaml 目录: {yaml_dir}")
    logger.info(f"  报告路径: {args.output}")
    logger.info(f"  报告模式: {args.mode}")
    logger.info(f"  clause yaml: {len(clause_list)} 个（style: {style_count}, 范围外: {oor_count}）")
    logger.info(f"  检视统计: 总{total} / PASS {pass_n} / FAIL {fail_n} / SUSPICIOUS {susp_n}")
    logger.info(f"  设计一致性: {has_design}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
