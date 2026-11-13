#!/usr/bin/env python3
"""
gen_comment.py - 由 summary.json 生成 PR 评论 markdown

评论结构：总分概览 → 文件明细表 → 折叠的问题清单（行号可点击跳转） → 说明页脚。
同一 PR 多次运行时由 post_comment.py 依据 MARKER 原位更新，不产生新评论。
"""

import json
import os
import sys

import regex as re

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MARKER = "<!-- doc-quality-bot v1 -->"
INLINE_MARKER = "<!-- doc-quality-bot inline v1 -->"
MAX_COMMENT_CHARS = 55000  # GitHub 评论上限 65536，留余量
MAX_ISSUES_TOTAL = 30  # 评论最多展示的问题条数
MAX_ISSUES_PER_FILE = 15
MAX_INLINE = 10  # 行内评论最多条数（F/S 优先）

SEV_RANK = {"F": 3, "S": 2, "N": 1, "H": 0}
SEV_NAME = {"F": "Critical", "S": "Major", "N": "Minor", "H": "Hint"}
DIM_NAME = {"正确性": "Correctness", "完整性": "Completeness", "易理解": "Understandability"}
SEV_RE = re.compile(r"(致命|严重|一般|提示)?\s*([FSNH])?\s*[-－]\s*(\d+(?:\.\d+)?)")

# H2 category names (from h2_registry.json) mapped to English for PR comments.
H2_EN = {
    # Correctness
    "单位符号使用错误": "Unit symbol error",
    "不符合安全合规要求": "Security non-compliance",
    "标点符号错误": "Punctuation error",
    "链接错误或失效": "Broken or invalid link",
    "命令/文件名/路径/参数/示例等错误": "Command/file/path/parameter/example error",
    "英文拼写错误": "English spelling error",
    "中文错别字词": "Chinese typo",
    "与产品实现不一致": "Inconsistency with product implementation",
    "代码样例无法执行": "Non-executable code sample",
    "写作不规范": "Non-standard writing",
    # Completeness
    "参数相关信息不完整": "Incomplete parameter information",
    "产品规格/功能特性/兼容性信息不完整": "Incomplete product spec/feature/compatibility info",
    "场景/背景/辅助信息不完整": "Incomplete scenario/background/auxiliary info",
    "操作步骤不完整": "Incomplete operation steps",
    "链接缺失": "Missing links",
    # Understandability
    "文档架构/大纲不合理": "Irrational document structure/outline",
    "文档内容不易理解": "Hard-to-understand content",
    "示例代码不易理解": "Hard-to-understand sample code",
    "操作步骤逻辑不清晰": "Unclear operation step logic",
    "跳转多": "Excessive cross-references",
    "AI亲和": "Low AI affinity",
}

# Grade labels from merge_and_score.py (upstream, Chinese) mapped to English.
# S/A/B/C/D are kept as-is (already English letters used by special modes).
GRADE_EN = {
    "好用": "Excellent",
    "易用": "Good",
    "可用": "Fair",
    "难用": "Poor",
    "不可用": "Unusable",
}

# Known Chinese substrings emitted by upstream script rules (link checker etc.)
# that need translating for the English-only PR comment.  Order matters:
# longer/more specific patterns first.
_KNOWN_ZH = [
    ("直连与代理两轮均不可达（", "unreachable via both direct and proxy ("),
    ("直连与代理两轮均不确定（", "inconclusive via both direct and proxy ("),
    ("直连与代理复测结果不一致/不确定", "direct and proxy retries disagree/inconclusive"),
    ("平台 API 复核资源不存在（", "platform API recheck: resource not found ("),
    ("链接不可达：", "Link unreachable: "),
    ("404 资源不存在", "404 Not Found"),
    ("410 内容已移除", "410 Gone"),
    ("451 因法律原因不可用", "451 Unavailable For Legal Reasons"),
    ("403 确定性拒绝", "403 Forbidden"),
    ("锚点不存在", "anchor not found"),
    ("跨文档文件缺失", "cross-doc file missing"),
]


def h2_en(h2):
    return H2_EN.get(str(h2 or ""), str(h2 or ""))


def grade_en(grade):
    g = str(grade or "")
    return GRADE_EN.get(g, g)


def translate_known_zh(text):
    """Translate known Chinese substrings from upstream script rules to English."""
    t = str(text or "")
    for zh, en in _KNOWN_ZH:
        t = t.replace(zh, en)
    return t


def parse_sev(severity):
    m = SEV_RE.search(str(severity or ""))
    if not m:
        return ("N", "", str(severity or ""))
    letter = m.group(2) or {"致命": "F", "严重": "S", "一般": "N", "提示": "H"}[m.group(1)]
    return (letter, m.group(3), str(severity))


def line_anchor(line):
    m = re.match(r"(\d+)", str(line or ""))
    return m.group(1) if m else None


def blob_link(repo, sha, path, line):
    n = line_anchor(line)
    base = f"https://github.com/{repo}/blob/{sha}/{path}"
    return f"{base}#L{n}" if n else base


def fmt_code(text):
    text = str(text or "")
    return f"`{text}`" if "`" not in text else f'"{text}"'


def severity_counts(entry):
    cnt = entry.get("severity_count") or {}
    return tuple(int(cnt.get(k, 0) or 0) for k in ("致命", "严重", "一般", "提示"))


def load_problems(out_dir, entry):
    f = entry.get("problems_file")
    if not f:
        return []
    p = os.path.join(out_dir, f)
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("problems", [])
    except (OSError, json.JSONDecodeError):
        return []


def also_hit_note(p):
    """被归并的其他规则命中（同一处缺陷被多条规则命中时折叠进 also_hit）。"""
    hits = p.get("also_hit")
    if not isinstance(hits, list) or not hits:
        return ""
    descs = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        h2 = h2_en(h.get("h2", ""))
        h3 = h.get("h3", "")
        rule = f"{h2} {h3}".strip()
        letter, code, _ = parse_sev(h.get("severity"))
        tag = f"{letter}-{code}" if code else letter
        descs.append(f"[{tag}] {rule}")
    if not descs:
        return ""
    return f"Also matched ({len(descs)} more rule{'s' if len(descs) > 1 else ''}): " + ", ".join(descs)


def problem_line(entry, repo, sha, p):
    letter, code, _ = parse_sev(p.get("severity"))
    tag = f"{letter}-{code}" if code else letter
    ln = line_anchor(p.get("line"))
    loc = f"[L{ln}]({blob_link(repo, sha, entry['path'], p.get('line'))})" if ln else "L—"
    h2 = h2_en(p.get("h2", ""))
    head = f"- **[{tag}]** {loc} · {h2}"
    parts = [head, fmt_code(p.get("original", ""))]
    if p.get("issue"):
        parts.append(f"— {translate_known_zh(p['issue'])}")
    text = " ".join(parts)
    patch = p.get("patch")
    if isinstance(patch, dict) and patch.get("old") is not None:
        text += f"\n  - Suggested fix: {fmt_code(patch['old'])} → {fmt_code(patch.get('new', ''))}"
    elif p.get("fix_suggestion"):
        text += f"\n  - Suggestion: {translate_known_zh(p['fix_suggestion'])}"
    note = also_hit_note(p)
    if note:
        text += f"\n  - {note}"
    return text


def sort_problems(problems):
    def key(p):
        letter, code, _ = parse_sev(p.get("severity"))
        return (-SEV_RANK.get(letter, 0), line_anchor(p.get("line")) or "999999999")

    return sorted(problems, key=key)


def render(out_dir, summary, inline_count=0):
    repo = summary["repo"]
    sha = summary["head_sha"]
    files = summary.get("files", [])
    ok_files = [e for e in files if e.get("status") == "ok"]
    bad_files = [e for e in files if e.get("status") != "ok"]

    lines = [MARKER, f"## Doc Quality Report (PR #{summary['pr']})", ""]
    if not files:
        if summary.get("deleted_only"):
            lines.append("This PR only deletes files; nothing to evaluate.")
        else:
            lines.append("No changed markdown files under docs/source; evaluation skipped.")
        lines.append("")
        return "\n".join(lines)

    # 概览
    tot = [sum(x) for x in zip(*(severity_counts(e) for e in ok_files))] if ok_files else [0, 0, 0, 0]
    total_issues = sum(tot)
    scores = [e["total_score"] for e in ok_files if isinstance(e.get("total_score"), (int, float))]
    overview = (
        f"Evaluated **{len(ok_files)}** file(s), found **{total_issues}** issue(s)"
        f" (Critical {tot[0]} / Major {tot[1]} / Minor {tot[2]} / Hint {tot[3]})"
    )
    if inline_count:
        overview += f"; **{inline_count}** Critical/Major issue(s) also posted as inline review comments"
    if scores and len(scores) == len(ok_files):
        avg = round(sum(scores) / len(scores), 1)
        grades = {grade_en(e.get("grade")) for e in ok_files}
        overview += f", average score **{avg}** (grade: {', '.join(sorted(str(g) for g in grades))})"
    lines += [overview, ""]

    # 文件表
    lines += ["| File | Score | Grade | F/S/N/H |", "| --- | --- | --- | --- |"]
    for e in ok_files:
        f, s, n, h = severity_counts(e)
        score = e.get("total_score")
        score = f"{score}" if score is not None else "—"
        lines.append(
            f"| [{e['path']}]({blob_link(repo, sha, e['path'], None)}) "
            f"| {score} | {grade_en(e.get('grade', '—'))} | {f}/{s}/{n}/{h} |"
        )
    lines.append("")

    # 问题清单（折叠）
    budget = MAX_ISSUES_TOTAL
    lines.append("<details>")
    lines.append(f"<summary><b>Issue details (sorted by severity, up to {MAX_ISSUES_TOTAL} shown)</b></summary>")
    lines.append("")
    for e in ok_files:
        probs = sort_problems(load_problems(out_dir, e))
        if not probs:
            continue
        shown = probs[: min(MAX_ISSUES_PER_FILE, budget)]
        budget -= len(shown)
        lines.append(f"### {e['path']}")
        for p in shown:
            lines.append(problem_line(e, repo, sha, p))
        hidden = len(probs) - len(shown)
        if hidden > 0:
            lines.append(f"\n({hidden} more not shown; see the full report in workflow Artifacts)")
        lines.append("")
        if budget <= 0:
            break
    lines.append("</details>")
    lines.append("")

    # 异常与警告
    warns = []
    for e in files:
        for w in e.get("warnings", []):
            warns.append(f"- {e['path']}: {w}")
    for e in bad_files:
        warns.append(f"- {e['path']}: evaluation failed ({e.get('status')})")
    for f in summary.get("skipped_overflow", []):
        warns.append(f"- {f}: exceeded the per-run file limit, not evaluated")
    if warns:
        lines += ["<details><summary><b>Warnings & not evaluated</b></summary>", "", *warns, "", "</details>", ""]

    # 页脚说明
    dims = ", ".join(DIM_NAME.get(d, d) for d in summary.get("dims", []))
    lines += [
        "---",
        f"<sup>Model: {summary.get('model', '—')} | Dimensions: "
        f"{dims} | Evaluated at: "
        f"{summary.get('generated_at', '—')} | "
        f"Full xlsx/md report in this workflow's Artifacts "
        f"(doc-quality-report-pr-{summary['pr']})</sup>",
        "",
        "<sup>Re-run: comment `/doc-eval` (requires write access). "
        "Severity: Critical F=the task will certainly fail | Major S=a workaround is required | "
        "Minor N=noticeable quality issue | Hint H=room for polish.</sup>",
    ]
    return "\n".join(lines)


def truncate_keep_marker(text, limit=MAX_COMMENT_CHARS):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    tail = "\n\n<sup>(issue list truncated — full content in Artifacts)</sup>"
    return cut + tail


def inline_body(p):
    """单条行内评论文本（钉在 diff 行上）。"""
    letter, code, _ = parse_sev(p.get("severity"))
    sev_name = SEV_NAME.get(letter, letter)
    tag = f"{letter}-{code}" if code else letter
    lines = [f"**[{tag}] {sev_name}** · {h2_en(p.get('h2', ''))}", ""]
    if p.get("issue"):
        lines += [translate_known_zh(p["issue"]), ""]
    patch = p.get("patch")
    if isinstance(patch, dict) and patch.get("old") is not None:
        old = patch["old"]
        new = patch.get("new", "")
        lines += ["Suggested change:", "", "```diff", f"- {old}", f"+ {new}", "```", ""]
    elif p.get("fix_suggestion"):
        lines += [f"Suggestion: {translate_known_zh(p['fix_suggestion'])}", ""]
    lines += [
        "<sub>Generated by doc-quality bot · score and full issue list in the pinned summary comment</sub>",
        INLINE_MARKER,
    ]
    return "\n".join(lines)


def build_inline(out_dir, summary):
    """选取行内评论候选：仅 F/S 级、行号可解析、按严重度排序，封顶 MAX_INLINE。

    是否在 PR diff 内由 post_comment.py 结合实际 diff 过滤。
    """
    repo = summary["repo"]
    sha = summary["head_sha"]
    out = []
    for e in [f for f in summary.get("files", []) if f.get("status") == "ok"]:
        for p in sort_problems(load_problems(out_dir, e)):
            if len(out) >= MAX_INLINE:
                break
            letter, _, _ = parse_sev(p.get("severity"))
            if letter not in ("F", "S"):
                break  # sort_problems 已按严重度降序，到 N 即停
            n = line_anchor(p.get("line"))
            if not n:
                continue
            body = inline_body(p)
            if len(body) > 6000:  # GitHub 行内评论体上限 ~64KB，留余量防超长 patch
                body = body[:6000] + f"\n{INLINE_MARKER}"
            out.append(
                {
                    "path": e["path"],
                    "line": int(n),
                    "body": body,
                    "severity": letter,
                    "h2": p.get("h2", ""),
                    "blob": blob_link(repo, sha, e["path"], p.get("line")),
                }
            )
    return out


def generate(summary_path, out_path):
    out_dir = os.path.dirname(os.path.abspath(summary_path))
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    inline = []
    if os.environ.get("DOC_EVAL_INLINE", "1") != "0":
        inline = build_inline(out_dir, summary)
    text = truncate_keep_marker(render(out_dir, summary, inline_count=len(inline)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    inline_path = os.path.join(out_dir, "inline.json")
    with open(inline_path, "w", encoding="utf-8") as f:
        json.dump(inline, f, ensure_ascii=False, indent=2)
    print(f"[gen_comment] comment generated: {out_path} ({len(text)} chars); inline candidates: {len(inline)} -> {inline_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_comment.py <summary.json> <output comment.md>", file=sys.stderr)
        sys.exit(2)
    generate(sys.argv[1], sys.argv[2])
