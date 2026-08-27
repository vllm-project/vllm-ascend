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
"""轻量获取 CANN 日落（废弃）API/头文件清单。

动态解析官方「废弃接口/返回码列表」（Runtime API(C/Python)）与发行说明「算子库」章节，
输出「废弃符号 -> 替代」清单。返回码在 C/Python 两表重复、aclnn 跨版本重复，均去重后保留先出现项。
"""
import html
import json
import logging
import re
import sys
import urllib.error
import urllib.request

BASE = "https://www.hiascend.com"
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0"}
GW = {**UA, "Referer": "https://www.hiascend.com/"}
FALLBACK = [("920beta1", "9.2.0-beta.1"), ("910", "9.1.0"), ("900", "9.0.X")]

# 日落文档来源列表：(分类, URL 模板, 解析类型)
# URL 中的 {code}/{name} 在运行时由解析出的最新 CANN 版本填充
_DOC_BASE = "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/{code}"
SUNSET_DOCS = [
    ("Runtime API(C)",
     _DOC_BASE + "/API/runtimeapi/aclcppdevg_03_0019.html", "api"),
    ("Runtime API(Python)",
     _DOC_BASE + "/API/runtimeapi/aclpythondevg_01_0002.html", "api"),
    ("算子库",
     _DOC_BASE + "/softwareinst/releasenote/{name}/release-notes.md", "release"),
]

TOK = r"[A-Za-z0-9_.]+"
API = re.compile(
    r"(" + TOK + r")\s*接口\s*此接口后续版本会废弃，请使用\s*"
    r"(" + TOK + r"(?:\s*或\s*" + TOK + r")*)\s*接口。")
RC = re.compile(
    r"(ACL_ERROR_[A-Za-z0-9_]+)\s*(?:返回码)?\s*此返回码后续版本会废弃，请使用\s*"
    r"(ACL_[A-Za-z0-9_]+)\s*返回码。")
ENUM = re.compile(
    r"(ACL_OPT_[A-Za-z0-9_]+).*?请使用\s*(ACL_OPT_[A-Za-z0-9_]+)", re.S)


def get(url, headers=None):
    headers = headers or UA
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", "replace")


def src(code, path):
    return f"{BASE}/doc_center/source/zh/CANNCommunityEdition/{code}/{path}"


def resolve_version():
    code = name = None
    try:
        data = json.loads(get(
            f"{BASE}/ascendgateway/ascendservice/doc/queryProductLatestVersions?lang=zh", GW))
        code = next((p.get("versionCode") for p in data.get("data", [])
                     if p.get("productCode") == "CANNCommunityEdition"), None)
    except (urllib.error.URLError, ValueError) as exc:
        logger.warning("解析最新版本（queryProductLatestVersions）失败: %s", exc)
    try:
        data = json.loads(get(
            f"{BASE}/ascendgateway/ascendservice/doc/version/"
            f"zh/CANNCommunityEdition/latest/softwareinst/releasenote/release-notes.md", GW))
        info = data.get("data") or {}
        code = code or info.get("versionCode")
        name = info.get("versionName")
    except (urllib.error.URLError, ValueError) as exc:
        logger.warning("解析最新版本（release-notes）失败: %s", exc)
    if code and name:
        return code, name
    for c, n in FALLBACK:
        try:
            get(src(c, "API/runtimeapi/aclcppdevg_03_0019.html"))
            return c, n
        except urllib.error.URLError as exc:
            logger.info("回退版本 %s 不可用: %s", c, exc)
            continue
    raise RuntimeError("无法解析最新 CANN 版本")


def to_text(raw):
    raw = re.sub(r"<script[\s\S]*?</script>", "", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", "", raw, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "\n", raw))


def parse_api_page(raw, category):
    text = "".join(to_text(raw).split())
    out = []
    for m in API.finditer(text):
        out.append({"cat": category, "sym": m.group(1),
                    "rep": re.split(r"\s*或\s*", m.group(2))})
    for m in RC.finditer(text):
        out.append({"cat": category, "sym": m.group(1), "rep": [m.group(2)]})
    for m in ENUM.finditer(text):
        out.append({"cat": category, "sym": m.group(1), "rep": [m.group(2)]})
    return out


def normalize_header(dep):
    d = dep.replace("${install_path}/ascend-toolkit/latest/", "")
    d = d.replace("${install_path}/", "")
    if "op_proto" in d:
        return "op_proto/inc/"
    if "/*.h" in d:
        return d.split("/*.h")[0].rstrip("/") + "/"
    if d.endswith(".so"):
        return d.rsplit("/", 1)[-1]
    return d


def parse_release_notes(md):
    sec = re.search(r"### 算子库\s*\n(.*?)(?=\n### )", md, re.S)
    if not sec:
        return []
    out, deadline = [], None
    for line in sec.group(1).splitlines():
        m = re.search(r"(\d{4}\.\d{1,2}\.\d{1,2})之后的版本删除", line)
        if m:
            deadline = m.group(1)
            continue
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or "废弃的目录" in cells[0] or set(cells[0]) <= set("-: "):
            continue
        apis = re.findall(r"aclnn[A-Za-z0-9_]+", cells[0])
        reps = re.split(r"\s*和\s*", cells[1].rstrip("接口").strip())
        for a in apis:
            out.append({"cat": "算子库", "sym": a, "rep": reps, "deadline": deadline})
        if not apis:
            pattern = normalize_header(cells[0])
            if pattern:
                out.append({"cat": "算子库(头文件/库)", "sym": cells[0],
                            "rep": reps, "deadline": deadline, "pattern": pattern})
    return out


def main():
    code, name = resolve_version()
    items = []
    for cat, url, kind in SUNSET_DOCS:
        raw = get(url.format(code=code, name=name))
        items += parse_api_page(raw, cat) if kind == "api" else parse_release_notes(raw)

    seen, out = set(), []
    for it in items:
        key = it.get("pattern") or it["sym"]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    grouped = {}
    for it in out:
        grouped.setdefault(it["cat"], []).append(it)

    logger.info(f"# 日落（废弃）API/头文件清单 | CANN {code} ({name})")
    for cat, its in grouped.items():
        logger.info(f"## {cat}")
        for it in its:
            dl = f"[删除期限 {it['deadline']}] " if it.get("deadline") else ""
            logger.info(f"{dl}{it['sym']} -> {' / '.join(it['rep'])}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
