#!/usr/bin/env python3
"""
llm_eval.py - CI 版 LLM 语义检测层

按 h2_registry.json 逐 H2 调用 OpenAI 兼容 API（GLM/DeepSeek/Kimi 等），
产出与本地 skill 子智能体同 schema 的逐 H2 JSON 文件
（correctness_problems_h2_{n}.json / completeness_* / usability_*），
供 doc-evaluator/scripts/merge_and_score.py 直接收编。

两类 H2：
  - 常规 H2：长文自动分块，逐块并行调用；
  - 脚本联动 H2（正确性 H2-2 安全合规 / H2-3 标点 / H2-4 链接）：
    与本地 skill 一致消费脚本前置产物——等脚本 JSON 就绪后全文单次调用，
    复核脚本 candidates[]（确认 → 落问题并带 _script/_script_llm_merged 标记）
    并执行各自规则文件的纯 LLM 规则；脚本 problems[] 已由 merge 直接收编，不重复上报。

用法（由 run_eval.py 调用，也可独立执行）:
  python llm_eval.py --doc <文档路径> --temp-dir <temp目录> --standard-dir <01.评估标准目录> \
      [--dims 正确性,完整性,易理解] [--max-lines 700] [--concurrency 6]

环境变量:
  DOC_EVAL_BASE_URL / DOC_EVAL_API_KEY / DOC_EVAL_MODEL   必填
"""

import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import regex as re

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 维度中文名 → 输出文件前缀（与本地 skill 子智能体输出命名一致，merge 按前缀识别维度）
DIM_FILE_PREFIX = {
    "正确性": "correctness",
    "完整性": "completeness",
    "易理解": "usability",
}

# 脚本联动 H2（h2 名 → 脚本名 + 产物文件名）；联动脚本规则文件取自 registry 的 script_rule_file
SCRIPT_LINKED = {
    "不符合安全合规要求": ("security", "security_quickcheck_problems.json"),
    "标点符号错误": ("punctuation", "punctuation_quickcheck_problems.json"),
    "链接错误或失效": ("link", "link_quickcheck_problems.json"),
}

SEV_RE = re.compile(r"^\s*(致命|严重|一般|提示)\s*([FSNH])?\s*[-－]\s*(\d+(?:\.\d+)?)")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# 问题文本输出语言：默认中文（与规则文件一致）；DOC_EVAL_LANG=en 时
# issue/fix_suggestion 用英文（severity/h3 等结构化字段不受影响）
LANG = os.environ.get("DOC_EVAL_LANG", "zh").strip().lower()
_ISSUE_LANG = "English" if LANG.startswith("en") else "中文（简体）"
_QUOTE_RULE = "single quotes '...'" if LANG.startswith("en") else "中文全角引号「」"

# ---------------------------------------------------------------- 提示词 ----

SCHEMA_PROMPT = """\
# 输出格式（严格遵守）
仅输出一个 JSON 对象，禁止 markdown 代码块包裹，禁止输出任何其他文字：
{{"problems": [ ... ]}}

每条问题字段：
- "line": 字符串，问题所在行号（以输入「行号|内容」标注的行号为准）
- "original": 问题点最小片段（≤50字符）：从该行原文逐字摘录，用【】标出问题文字，
  上下文仅补充到该片段在该行内唯一为止；原样保留反引号、链接/图片语法 [文本](URL)
  的方括号与括号等 Markdown 标记；空格/不可见字符作问题点时【】须覆盖该字符及前后各至少1个字符
- "issue": 问题描述（{issue_lang}，说明违反哪条规则、为何是问题）
- "severity": "致命F-x.y" / "严重S-x.y" / "一般N-x.y" / "提示H-x.y" 之一（x.y 为规则条目号）
- "h2": 固定写 "{h2}"
- "h3": 仅规则条目号（如 "2.1"），禁止附带文字
- "fixable": "✅"（AI 可精确修复）或 "❌"（需人工判断）；代码块内的问题一律 "❌"
- "patch": 仅当 fixable=✅ 且 line 有效时提供 {{"old": "行内原文片段", "new": "替换后完整片段"}}；
  old 须摘自该行原文且在该行内唯一；new 是 old 覆盖片段替换后的完整文本（old 之外内容由脚本保留拼接，
  禁止在 new 中重复）；禁止修改代码块内代码逻辑（仅可改注释与格式）；禁止改动链接/图片语法边界字符；
  不满足条件时省略 patch 字段
- "fix_suggestion": fixable=❌ 或无 patch 时给修复建议（{issue_lang}）；有 patch 时可省略
{extra_schema}

# 硬性要求
1. 只报告上述规则明确覆盖、有把握的问题，宁缺毋滥；无问题输出 {{"problems": []}}
2. JSON 字符串值内禁止未转义英文双引号 "（引用术语/原文一律用{quote_rule}）；字符串单行不换行
3. 每条 original 去掉【】后必须能逐字连续出现在 line 行原文中；line 行为空行时禁止上报（行号错位）
4. issue 中引用的错误词必须能在 original 中找到
"""

SCRIPT_LINKED_EXTRA_SCHEMA = """
5. 本检测项为「脚本+LLM 联动」，输入中附有脚本预筛的 candidates[]（每条自带 rule/signal/context/review
   复核口径）：逐条结合 context 复核——确认成立者作为一条问题输出（h3 取该 candidate 的 rule 号，
   并加字段 "_script": true、"_script_llm_merged": true，severity 按 candidate.review 口径定级，
   可给 patch/fix_suggestion）；不成立者直接忽略，禁止输出。candidates 之外由你按规则自查出的问题
   正常输出（不加 _script 字段）。
6. 输入中「脚本已直接判定的问题清单」已由系统记录，禁止重复上报其中任何一条。
"""

DEGRADED_EXTRA_SCHEMA = """
5. 本次运行脚本前置产物不可用，仅执行规则文件中的纯 LLM 语境/语义类规则；
   依赖脚本数据（可达性、确定性规则命中）的部分一律不报告。
"""


def _trim(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n（注：规则过长已截断，以已展示部分为准）"


def build_system_prompt(dim, h2, rule_text, scoring_text, script_rule_text=None, degraded=False):
    if SCRIPT_LINKED.get(h2):
        extra = DEGRADED_EXTRA_SCHEMA if degraded else SCRIPT_LINKED_EXTRA_SCHEMA
    else:
        extra = ""
    parts = [
        f"# 角色\n你是资深开源文档质量评测专家。本次任务：对文档进行检测维度「{dim}」、"
        f"检测项「{h2}」的质量检测。只检测该检测项规则覆盖的问题，不做其他维度检测。",
        "# 严重程度定级规则（四级定义与判定问句，逐条作答，第一个答「是」的级别即定级）\n" + _trim(scoring_text, 9000),
        f"# 检测项「{h2}」规则\n" + _trim(rule_text, 12000),
    ]
    if script_rule_text:
        parts.append("# 脚本已判定规则明细（candidates 复核口径参考）\n" + _trim(script_rule_text, 9000))
    parts.append(
        SCHEMA_PROMPT.replace("{h2}", h2)
        .replace("{extra_schema}", extra)
        .replace("{issue_lang}", _ISSUE_LANG)
        .replace("{quote_rule}", _QUOTE_RULE)
    )
    return "\n\n---\n\n".join(parts)


def numbered(lines, start=1):
    return "\n".join(f"{n}|{text}" for n, text in enumerate(lines, start=start))


def build_user_prompt(rel_path, total_lines, chunk_lines, start, extra=""):
    head = (
        f"文档：{rel_path}（共 {total_lines} 行，行号已标注在每行行首，"
        f"格式「行号|内容」，问题 line 字段直接使用该行号）"
    )
    if extra:
        head += f"\n\n{extra}"
    return f"{head}\n\n{numbered(chunk_lines, start)}"


# ---------------------------------------------------------------- API 调用 ----


def chat(base_url, api_key, model, messages, temperature=0.1, max_tokens=8192, timeout=300, retries=2):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(5 * attempt)
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError, OSError) as e:
            detail = ""
            if hasattr(e, "read"):
                with contextlib.suppress(Exception):
                    detail = e.read().decode("utf-8", errors="replace")[:300]
            last_err = f"{e} {detail}"
            print(f"[llm_eval] 调用失败（第 {attempt + 1} 次）: {last_err}", file=sys.stderr)
    raise RuntimeError(f"LLM API 调用失败: {last_err}")


def extract_json_obj(content):
    """从模型返回中提取 JSON 对象。兼容 ```json 包裹与 <think> 前缀。"""
    text = THINK_RE.sub("", content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("返回中未找到 JSON 对象")
    return json.loads(text[i : j + 1])


def extract_problems(content):
    obj = extract_json_obj(content)
    problems = obj.get("problems")
    if not isinstance(problems, list):
        raise ValueError("JSON 中缺少 problems 数组")
    return problems


def normalize_problem(p, dim, h2):
    """校验/修正单条问题，非法条目返回 None。"""
    if not isinstance(p, dict):
        return None
    line = str(p.get("line", "") or "").strip()
    if not re.fullmatch(r"\d+([-~至]\d+)?", line):
        return None
    original = str(p.get("original", "") or "").strip()
    if not original:
        return None
    sev = str(p.get("severity", "") or "").strip()
    m = SEV_RE.match(sev)
    if not m:
        return None
    h3 = str(p.get("h3", "") or m.group(3)).strip()
    patch = p.get("patch")
    if not (isinstance(patch, dict) and patch.get("old") and "new" in patch):
        patch = None
    out = {
        "line": line,
        "original": original[:80],
        "issue": str(p.get("issue", "") or "").strip(),
        "severity": f"{m.group(1)}{m.group(2)}-{m.group(3)}",
        "fixable": "✅" if str(p.get("fixable", "")).strip() == "✅" else "❌",
        "h2": h2,
        "h3": h3,
        "dim": dim,
    }
    if p.get("_script"):
        out["_script"] = True
        out["_script_llm_merged"] = True
    if patch:
        out["patch"] = {"old": str(patch["old"]), "new": str(patch["new"])}
    if str(p.get("fix_suggestion", "") or "").strip():
        out["fix_suggestion"] = str(p["fix_suggestion"]).strip()
    return out


# ---------------------------------------------------------------- 辅助 ----


def load_registry(standard_dir):
    path = os.path.join(standard_dir, "common", "h2_registry.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_scoring_rules(standard_dir):
    """评分规则第一~三节（定级用），第四节起为脚本用，截去。"""
    path = os.path.join(standard_dir, "common", "_文档评分规则.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    idx = text.find("## 四、")
    return text[:idx] if idx > 0 else text


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def wait_for_script_json(path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(path):
            return True
        time.sleep(5)
    return False


def iter_chunks(lines, max_lines, overlap=30):
    step = max(1, max_lines - overlap)
    for s in range(0, len(lines), step):
        e = min(len(lines), s + max_lines)
        yield s + 1, e, lines[s:e]  # 1-based 行号
        if e >= len(lines):
            break


# ---------------------------------------------------------------- 主流程 ----


def _call_llm(cfg, sys_prompt, user_prompt):
    return chat(
        cfg["base_url"],
        cfg["api_key"],
        cfg["model"],
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=cfg["max_tokens"],
    )


def eval_regular_h2(dim, item, lines, total, rel_path, cfg, scoring):
    """常规 H2：分块并行，合并各块问题。"""
    rule_text = read_text(os.path.join(cfg["standard_dir"], item["rule_file"]))
    sys_prompt = build_system_prompt(dim, item["h2"], rule_text, scoring)
    problems = []
    for start, end, chunk in iter_chunks(lines, cfg["max_lines"]):
        user = build_user_prompt(rel_path, total, chunk, start)
        raw = extract_problems(_call_llm(cfg, sys_prompt, user))
        problems.extend(raw)
    return problems


def eval_script_linked_h2(dim, item, lines, total, rel_path, cfg, scoring, temp_dir, warns):
    """脚本联动 H2：等脚本产物 → 全文单次调用（复核 candidates + 纯 LLM 规则）。"""
    script_name, script_json = SCRIPT_LINKED[item["h2"]]
    script_path = os.path.join(temp_dir, script_json)
    degraded = True
    extra = ""
    script_rule_file = item.get("script_rule_file")
    script_rule_text = read_text(os.path.join(cfg["standard_dir"], script_rule_file)) if script_rule_file else None
    if wait_for_script_json(script_path, cfg["script_wait_sec"]):
        try:
            data = json.loads(read_text(script_path))
            cands = data.get("candidates", [])
            script_probs = [
                {"line": p.get("line"), "original": p.get("original", "")[:50]}
                for p in data.get("problems", [])
                if isinstance(p, dict)
            ]
            degraded = False
            if script_name == "link":
                extra += (
                    f"脚本已直接判定的问题清单（已记录，禁止重复上报）：\n"
                    f"{json.dumps(script_probs, ensure_ascii=False)}\n\n"
                )
            if cands:
                extra += (
                    f"脚本预筛 candidates[]（逐条复核，口径见 candidate.review）：\n"
                    f"{json.dumps(cands, ensure_ascii=False)}\n\n"
                )
            else:
                extra += "脚本本次无预筛 candidates。\n\n"
        except (OSError, json.JSONDecodeError) as e:
            warns.append(f"{dim}-H2_{item['seq']}「{item['h2']}」脚本产物解析失败: {e}")
    else:
        warns.append(f"{dim}-H2_{item['seq']}「{item['h2']}」脚本产物未就绪（{script_name}），降级为纯 LLM 检测")
    sys_prompt = build_system_prompt(
        dim,
        item["h2"],
        read_text(os.path.join(cfg["standard_dir"], item["rule_file"])),
        scoring,
        script_rule_text,
        degraded,
    )
    user = build_user_prompt(rel_path, total, lines, 1, extra)
    return extract_problems(_call_llm(cfg, sys_prompt, user))


PO_RULE_PROMPT = (
    "You are reviewing a gettext .po translation file (Chinese msgid -> English msgstr)\n"
    "for a technical documentation project. Evaluate ONLY the English msgstr entries:\n"
    "- Accuracy: msgstr must faithfully convey the msgid meaning; flag mistranslation, omission or additions.\n"
    "- Terminology: API names, parameters, flags and code identifiers must remain untranslated and consistent.\n"
    "- Grammar, spelling and typos in msgstr.\n"
    "- Placeholders/markup from msgid ({}, {0}, %s, <tag>, `code`, URLs) must be preserved in msgstr.\n"
    "Skip the PO header metadata, obsolete entries (|#|) and empty msgstr (untranslated items are not errors).\n"
    "Report the file line of the offending msgstr, and put the problematic msgstr text (verbatim) in original.\n"
)


def eval_po_doc(doc_path, temp_dir, standard_dir, cfg):
    """gettext .po 翻译文件走单一「翻译质量」检测，不套用 21 个 H2。

    产出 correctness_problems_po.json——merge 按文件名前缀归维度，
    口径与逐 H2 文件完全一致。
    """
    scoring = load_scoring_rules(standard_dir)
    lines = read_text(doc_path).splitlines()
    total = len(lines)
    rel_path = os.path.relpath(doc_path, cfg.get("repo_root") or ".").replace(os.sep, "/")
    cfg = {**cfg, "standard_dir": standard_dir}
    fname = "correctness_problems_po.json"
    h2_name = "翻译质量"
    dim = "正确性"
    print(f"[llm_eval] {rel_path}: .po translation-quality detection ({total} lines)")
    warns = []
    probs_raw = []
    try:
        sys_prompt = build_system_prompt(dim, h2_name, PO_RULE_PROMPT, scoring, None, False)
        user = build_user_prompt(rel_path, total, lines, 1, "")
        probs_raw = extract_problems(_call_llm(cfg, sys_prompt, user))
    except Exception as e:  # noqa: BLE001 —— 失败写空文件，保证 merge --expect 可通过
        warns.append(f"[po] translation-quality detection failed: {e}")
        print(f"[llm_eval] {warns[-1]}", file=sys.stderr)
    deduped, seen = [], set()
    for p in probs_raw:
        n = normalize_problem(p, dim, h2_name)
        if not n:
            continue
        k = (n["line"], n["original"][:50], n["h2"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(n)
    out = {
        "dimension": dim,
        "files_loaded": 1,
        "h2_summaries": [{"h2": h2_name, "loaded_files": [os.path.basename(doc_path)], "problems": len(deduped)}],
        "problems": deduped,
    }
    with open(os.path.join(temp_dir, fname), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return [fname], warns


def eval_doc(doc_path, temp_dir, standard_dir, dims, cfg):
    """对单个文档执行全部选中 H2 的 LLM 检测，逐 H2 落盘。

    返回 (expect_files, warnings)。即使某个 H2 失败也写空文件，
    保证 merge --expect 可通过（失败信息进 warnings，最终体现在 PR 评论中）。
    """
    if str(doc_path).lower().endswith(".po"):
        return eval_po_doc(doc_path, temp_dir, standard_dir, cfg)
    registry = load_registry(standard_dir)
    scoring = load_scoring_rules(standard_dir)
    lines = read_text(doc_path).splitlines()
    total = len(lines)
    rel_path = os.path.relpath(doc_path, cfg.get("repo_root") or ".").replace(os.sep, "/")
    cfg = {**cfg, "standard_dir": standard_dir}

    regular, linked = [], []
    for dim in dims:
        for item in registry.get(dim, []):
            (linked if item["h2"] in SCRIPT_LINKED else regular).append((dim, item))

    print(
        f"[llm_eval] {rel_path}: {total} 行；常规 H2 {len(regular)} 个"
        f"（分块 ≤{cfg['max_lines']} 行），脚本联动 H2 {len(linked)} 个"
    )

    warns = []
    results = {}  # (dim, seq, h2) -> raw problems

    def run_regular(t):
        dim, item = t
        return t, eval_regular_h2(dim, item, lines, total, rel_path, cfg, scoring)

    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        futs = {pool.submit(run_regular, t): t for t in regular}
        for done, fut in enumerate(as_completed(futs), start=1):
            t = futs[fut]
            dim, item = t
            try:
                _, raw = fut.result()
                results[(dim, item["seq"], item["h2"])] = raw
                print(f"[llm_eval] ({done}/{len(regular)}) {dim}-H2_{item['seq']} {item['h2']}: {len(raw)} 条")
            except Exception as e:  # noqa: BLE001 —— 单任务失败不拖垮整体
                results[(dim, item["seq"], item["h2"])] = []
                msg = f"{dim}-H2_{item['seq']}「{item['h2']}」LLM 检测失败: {e}"
                warns.append(msg)
                print(f"[llm_eval] ({done}/{len(regular)}) {msg}", file=sys.stderr)

    # 脚本联动 H2：此时常规任务已消耗部分时间，脚本大概率已就绪；短轮询兜底
    cfg["script_wait_sec"] = cfg.get("script_wait_sec", 240)
    for dim, item in linked:
        key = (dim, item["seq"], item["h2"])
        try:
            raw = eval_script_linked_h2(dim, item, lines, total, rel_path, cfg, scoring, temp_dir, warns)
            results[key] = raw
            print(f"[llm_eval] {dim}-H2_{item['seq']} {item['h2']}: {len(raw)} 条")
        except Exception as e:  # noqa: BLE001
            results[key] = []
            msg = f"{dim}-H2_{item['seq']}「{item['h2']}」LLM 检测失败: {e}"
            warns.append(msg)
            print(f"[llm_eval] {msg}", file=sys.stderr)

    # 逐 H2 落盘（同 skill 口径：每 H2 一个小 JSON，禁止合并版大文件）
    expect_files = []
    for dim in dims:
        for item in registry.get(dim, []):
            probs_raw = results.get((dim, item["seq"], item["h2"]), [])
            deduped, seen = [], set()
            for p in probs_raw:
                n = normalize_problem(p, dim, item["h2"])
                if not n:
                    continue
                k = (n["line"], n["original"][:50], n["h2"])
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(n)
            fname = f"{DIM_FILE_PREFIX[dim]}_problems_h2_{item['seq']}.json"
            out = {
                "dimension": dim,
                "files_loaded": 1,
                "h2_summaries": [{"h2": item["h2"], "loaded_files": [item["rule_file"]], "problems": len(deduped)}],
                "problems": deduped,
            }
            with open(os.path.join(temp_dir, fname), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            expect_files.append(fname)
    return expect_files, warns


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CI 版 LLM 逐 H2 语义检测")
    parser.add_argument("--doc", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--standard-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--dims", default="正确性,完整性,易理解")
    parser.add_argument("--max-lines", type=int, default=700)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--script-wait-sec", type=int, default=240)
    args = parser.parse_args()

    cfg = {
        "base_url": os.environ.get("DOC_EVAL_BASE_URL", ""),
        "api_key": os.environ.get("DOC_EVAL_API_KEY", ""),
        "model": os.environ.get("DOC_EVAL_MODEL", ""),
        "max_lines": args.max_lines,
        "concurrency": max(1, args.concurrency),
        "max_tokens": args.max_tokens,
        "script_wait_sec": args.script_wait_sec,
        "repo_root": args.repo_root,
    }
    if not (cfg["base_url"] and cfg["api_key"] and cfg["model"]):
        print("[llm_eval] 缺少 DOC_EVAL_BASE_URL / DOC_EVAL_API_KEY / DOC_EVAL_MODEL", file=sys.stderr)
        return 2

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    for d in dims:
        if d not in DIM_FILE_PREFIX:
            print(f"[llm_eval] 未知维度: {d}", file=sys.stderr)
            return 2

    expect_files, warns = eval_doc(args.doc, args.temp_dir, args.standard_dir, dims, cfg)
    print("[llm_eval] expect=" + ",".join(expect_files))
    if warns:
        print("[llm_eval] warnings=" + " || ".join(warns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
