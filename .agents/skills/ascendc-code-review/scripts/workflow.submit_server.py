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
yaml collector 服务 — 接收子 Agent 通过 curl 提交的 yaml 内容，写入最终目录。

由主 Agent 在阶段2（逐条检视波次派发）前启动，阶段2 结束后关闭。
子 Agent 不接触 yaml 输出目录，只通过 HTTP 端点提交结果，物理隔离 sibling yaml。

用法:
  python3 workflow.submit_server.py <output_dir> <port>

  output_dir: yaml 最终输出目录（由 workflow.create_review_dir.py 创建）
  port:       监听端口（由主 Agent 选定空闲端口）

端点:
  POST /submit?group={group_name}&clause={clause_id}
      body: yaml 内容（raw text）
      → 校验 yaml 合法性，强制覆盖 group_name 字段，写入 {output_dir}/{group}_{clause}.yaml

  GET /health
      → 返回 "ok"，供主 Agent 确认服务已启动

设计要点:
  - 子 Agent 只知道端口号，不知道 output_dir 路径
  - 文件命名由 collector 统一生成，消除安全化碰撞隐患
  - 纯标准库实现（http.server + yaml），无第三方依赖
  - 并发安全：每个请求独立写不同文件名，无锁竞争
  - 多检视并行：每个检视任务启动独立 collector 实例（不同端口 + 不同 output_dir）

退出码: 0=正常退出（SIGTERM）, 1=参数错误, 2=端口占用
"""

import http.server
import logging
import os
import re
import signal
import sys
import urllib.parse
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


OUTPUT_DIR = ""
PORT = 0


def _try_fix_mojibake(s):
    """URL query 中的中文若被 http.server 按 iso-8859-1 解码成 mojibake，尝试还原为 UTF-8。

    http.server 的 BaseHTTPRequestHandler 按 HTTP 协议用 iso-8859-1 解码请求行，
    子 Agent 的 curl 若不对中文做百分号编码，原始 UTF-8 字节会被逐字节按 latin-1
    解读成乱码（如 "通用" → "é\\x80\\x9aç\\x94¨"）。此处做一次 latin-1→utf-8 重解析兜底。
    body 路径走 UTF-8 解码天然无此问题，本函数仅用于 URL query 参数的兜底。
    """
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _safe_filename(name):
    """去掉文件路径分隔符和控制字符，消除路径穿越与命名碰撞隐患"""
    return name.replace("/", "_").replace("\\", "_").replace("\x00", "")


def _unique_fname(fname, output_dir):
    """文件名已存在时追加 _dup1, _dup2... 避免覆盖（large PR 下同一 clause_id
    被多个 group 检视，若子 Agent 漏写 group 会导致撞名覆盖，此处保底保留全部结果）"""
    candidate = fname
    idx = 1
    while os.path.exists(os.path.join(output_dir, candidate)):
        base, ext = os.path.splitext(fname)
        candidate = f"{base}_dup{idx}{ext}"
        idx += 1
    return candidate


def _validate_pass_schema(data: dict) -> list:
    """校验 PASS 条例：禁止携带 evidence/confidence"""
    errors = []
    if "evidence" in data:
        errors.append("PASS 条例禁止携带 evidence 字段（仅 FAIL/SUSPICIOUS 需要）")
    if "confidence" in data:
        errors.append("PASS 条例禁止携带 confidence 字段（仅 FAIL/SUSPICIOUS 需要）")
    return errors


def _validate_fail_snippet(data: dict) -> list:
    """校验 FAIL/SUSPICIOUS 的 code_snippet 结构"""
    errors = []
    snip = data.get("code_snippet")
    if snip is None:
        errors.append("FAIL/SUSPICIOUS 条例缺少 code_snippet 字段")
    elif not isinstance(snip, dict):
        errors.append(
            f"code_snippet 必须是 mapping (file_path/start_line/end_line/code), "
            f"实际类型: {type(snip).__name__}"
        )
    else:
        if not snip.get("file_path"):
            errors.append("code_snippet.file_path 缺失或为空")
        if not snip.get("code"):
            errors.append("code_snippet.code 缺失或为空（FAIL/SUSPICIOUS 必须附代码片段）")
    return errors


def _validate_evidence_list(items, key: str) -> list:
    """校验 evidence.positive/negative 列表项"""
    errors = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(
                f"evidence.{key}[{i}] 必须是 mapping (type/score/desc), "
                f"实际类型: {type(item).__name__}"
            )
        elif not item.get("score"):
            errors.append(f"evidence.{key}[{i}].score 缺失或为空")
    return errors


def _validate_fail_evidence(data: dict) -> list:
    """校验 FAIL/SUSPICIOUS 的 evidence 结构"""
    errors = []
    ev = data.get("evidence")
    if ev is None:
        errors.append("FAIL/SUSPICIOUS 条例缺少 evidence 字段")
        return errors
    if not isinstance(ev, dict):
        errors.append(
            f"evidence 必须是 mapping (positive/negative/confidence_value), "
            f"实际类型: {type(ev).__name__}"
        )
        return errors

    for key in ("positive", "negative"):
        val = ev.get(key)
        if val is None:
            errors.append(f"evidence.{key} 缺失")
        elif not isinstance(val, list):
            errors.append(f"evidence.{key} 必须是 list, 实际类型: {type(val).__name__}")
        else:
            errors.extend(_validate_evidence_list(val, key))
    if not ev.get("confidence_value"):
        errors.append("evidence.confidence_value 缺失或为空")
    return errors


def _validate_yaml_schema(data: dict) -> list:
    """校验 clause 类 yaml schema，返回错误列表（空列表=通过）。"""
    errors = []
    for field in ("clause_id", "status"):
        if not data.get(field):
            errors.append(f"缺少必填字段: {field}")

    status = data.get("status", "")
    if status == "PASS":
        errors.extend(_validate_pass_schema(data))
        return errors
    if status not in ("FAIL", "SUSPICIOUS"):
        errors.append(f"status 值非法: 期望 PASS/FAIL/SUSPICIOUS, 实际 '{status}'")
        return errors

    # problem_desc 必填（常见错误：子 Agent 写成 description）
    if not data.get("problem_desc"):
        if data.get("description"):
            errors.append("字段名错误: 应为 'problem_desc'，实际写成了 'description'")
        else:
            errors.append("缺少必填字段: problem_desc")

    # fix_suggestion 必填（常见错误：子 Agent 写成 suggestion）
    if not data.get("fix_suggestion"):
        if data.get("suggestion"):
            errors.append("字段名错误: 应为 'fix_suggestion'，实际写成了 'suggestion'")
        else:
            errors.append("缺少必填字段: fix_suggestion")

    errors.extend(_validate_fail_snippet(data))
    errors.extend(_validate_fail_evidence(data))
    return errors


def _validate_design_strategies(strategies) -> list:
    """校验 strategies 列表"""
    errors = []
    if strategies is None:
        errors.append("design yaml 缺少 strategies 字段（S1-S7 + D8 判定列表）")
        return errors
    if not isinstance(strategies, list):
        errors.append(f"strategies 必须是 list, 实际类型: {type(strategies).__name__}")
        return errors
    if len(strategies) == 0:
        errors.append("strategies 为空列表（至少需要 S1-S7 + D8 共 8 项）")
        return errors
    for i, s in enumerate(strategies):
        if not isinstance(s, dict):
            errors.append(f"strategies[{i}] 必须是 mapping, 实际类型: {type(s).__name__}")
            continue
        if not s.get("id"):
            errors.append(f"strategies[{i}].id 缺失")
        if not s.get("name"):
            errors.append(f"strategies[{i}].name 缺失")
        if "verdict" not in s:
            errors.append(f"strategies[{i}].verdict 缺失（期望 ✅/❌/N/A）")
    return errors


_LOCATION_RE = re.compile(
    r"^[\w./\-]+\.\w+(?::\d+(?:-\d+)?)?$"
)


def _validate_design_items(data: dict, field: str, id_field: str) -> list:
    """校验 deviations 或 doc_violations 列表"""
    errors = []
    items = data.get(field)
    if items is None:
        return errors
    if not isinstance(items, list):
        errors.append(f"{field} 必须是 list, 实际类型: {type(items).__name__}")
        return errors
    loc_key = "violation_location" if field == "doc_violations" else "code_location"
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{field}[{i}] 必须是 mapping, 实际类型: {type(item).__name__}")
            continue
        if not item.get(id_field):
            errors.append(f"{field}[{i}].{id_field} 缺失")
        if not item.get("desc"):
            errors.append(f"{field}[{i}].desc 缺失")
        loc_val = item.get(loc_key, "")
        if not loc_val:
            errors.append(f"{field}[{i}].{loc_key} 缺失")
        elif not _LOCATION_RE.match(str(loc_val)):
            errors.append(
                f"{field}[{i}].{loc_key} 格式非法（期望 文件路径:行号 或 文件路径:起始行-中止行，"
                f"禁止拼接多位置或附加说明文本，多位置请拆成多条记录），实际: {loc_val!r}"
            )
    return errors


def _validate_design_schema(data: dict) -> list:
    """校验 design 类 yaml schema，返回错误列表。"""
    errors = []
    errors.extend(_validate_design_strategies(data.get("strategies")))
    errors.extend(_validate_design_items(data, "deviations", "strategy_id"))
    errors.extend(_validate_design_items(data, "doc_violations", "doc_name"))
    return errors


def _resolve_group_and_fname(data: dict, url_group: str, clause: str) -> tuple:
    """解析 group 并生成文件名，返回 (group, fname)"""
    raw_group = data.get("group_name")
    if not isinstance(raw_group, str):
        raw_group = ""
    group = raw_group or url_group

    group_safe = _safe_filename(group) if group else ""
    clause_safe = _safe_filename(clause)
    if group_safe:
        fname = f"{group_safe}_{clause_safe}.yaml"
    else:
        fname = f"{clause_safe}.yaml"

    fname = _unique_fname(fname, OUTPUT_DIR)
    return (group, fname)


class CollectorHandler(http.server.BaseHTTPRequestHandler):
    """处理子 Agent 的 yaml 提交请求"""

    def log_message(self, *args):
        """静默日志，避免干扰主 Agent 的输出"""
        pass

    def _handle_submit(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(404, "not found")
            return

        params = urllib.parse.parse_qs(parsed.query)
        url_group = _try_fix_mojibake(params.get("group", [""])[0])
        clause = params.get("clause", [""])[0]

        if not clause:
            self.send_error(400, "missing clause parameter")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""

        # 校验 yaml 合法性
        try:
            data = yaml.safe_load(body)
        except Exception as e:
            self.send_error(400, "invalid yaml", str(e))
            return

        if not isinstance(data, dict):
            self.send_error(400, "yaml root must be a mapping")
            return

        # schema 校验：按 yaml 类型分流
        if data.get("type") == "design":
            errors = _validate_design_schema(data)
        else:
            errors = _validate_yaml_schema(data)
        if errors:
            self.send_error(400, "schema validation failed", "; ".join(errors))
            return

        group, fname = _resolve_group_and_fname(data, url_group, clause)
        if group:
            data["group_name"] = group

        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            self.send_error(500, f"write failed: {e}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"ok: {fname}\n".encode("utf-8"))

    def _handle_health(self):
        """健康检查端点"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404, "not found")

    # BaseHTTPRequestHandler 约定方法名映射
    do_POST = _handle_submit
    do_GET = _handle_health


def main() -> int:
    global OUTPUT_DIR, PORT

    if len(sys.argv) != 3:
        logger.error("用法: python3 workflow.submit_server.py <output_dir> <port>")
        return 1

    OUTPUT_DIR = sys.argv[1]
    PORT = int(sys.argv[2])

    if not os.path.isdir(OUTPUT_DIR):
        logger.error("output_dir 不存在: %s", OUTPUT_DIR)
        return 1

    # 主 Agent kill 时进程收到 SIGTERM 直接终止（恢复默认行为，退出码 0）
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), CollectorHandler)
    except OSError as e:
        logger.error("端口 %s 占用: %s", PORT, e)
        return 2

    # 以下为脚本最终产物输出（collector 启动确认）
    logger.info(f"collector listening on http://127.0.0.1:{PORT}, output: {OUTPUT_DIR}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
