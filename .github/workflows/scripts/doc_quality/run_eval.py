#!/usr/bin/env python3
"""
run_eval.py - CI 编排器：复刻 doc-evaluator SKILL.md 的调度流程

对 PR 变更的 docs/source/**/*.md 逐文件执行：
  1. 确定性脚本前置层（标点/安全合规/链接，并行后台）
  2. LLM 逐 H2 语义检测（llm_eval.py，与脚本并行）
  3. merge_and_score.py 全量合并出分（--expect 校验）
  4. gen_report.py 生成 xlsx+md 完整报告（temp 自动清理）
  5. 汇总 summary.json 并调 gen_comment.py 生成 PR 评论

用法:
  python run_eval.py --skill-root <skill仓库根> --repo-root <被检仓库根> \
      --changed-file changed.txt --repo <owner/name> --head-sha <sha> \
      --pr <PR号> --out-dir <产物目录>

环境变量:
  DOC_EVAL_BASE_URL / DOC_EVAL_API_KEY / DOC_EVAL_MODEL   必填（LLM API）
  DOC_EVAL_DIMENSIONS   默认「正确性,完整性,易理解」
  DOC_EVAL_MAX_FILES    默认 8，超出部分跳过并记录
  DOC_EVAL_MAX_LINES    默认 700（单次 LLM 调用覆盖的行数，超出自动分块）
  DOC_EVAL_CONCURRENCY  默认 6（LLM 并发）
  DOC_EVAL_SKIP_LINK    设为 1 跳过链接检查脚本（如 npm 依赖安装失败）
"""

import json
import os
import shutil
import subprocess
import sys
import time

import regex as re

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CI_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_WATCH_SEC = 300  # 前置脚本最长等待
POLL_INTERVAL = 3

SCRIPT_JSONS = {
    "punctuation": "punctuation_quickcheck_problems.json",
    "security": "security_quickcheck_problems.json",
    "link": "link_quickcheck_problems.json",
}

# --- skill 仓路径常量（上游结构变动时只改这里）---
SKILL_STANDARD_DIR = "skills/01.评估标准"
SKILL_TOOLS_DIR = "doc-evaluator/scripts"

# 三个前置脚本（相对 SKILL_STANDARD_DIR）及安全规则目录
PUNCT_CHECKER = "00.正确性/标点符号/scripts/punctuation-checker.py"
SECURITY_CHECKER = "00.正确性/安全合规/scripts/prescan.py"
SECURITY_RULES_DIR = "00.正确性/安全合规"
LINK_CHECKER_DIR = "00.正确性/链接与导航/scripts"

# 打分与报告工具（相对 SKILL_TOOLS_DIR）
MERGE_SCRIPT = "merge_and_score.py"
GEN_REPORT_SCRIPT = "gen_report.py"

# 公共资源（相对 SKILL_STANDARD_DIR，启动校验用）
REGISTRY_REL = "common/h2_registry.json"


def standard_dir(skill_root):
    return os.path.join(skill_root, SKILL_STANDARD_DIR)


def doc_evaluator_scripts(skill_root):
    return os.path.join(skill_root, SKILL_TOOLS_DIR)


def safe_name(rel):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", rel)


def read_changed(changed_file, repo_root, path_prefix):
    """读取变更清单，过滤出仍存在（未被删除）的 md / .po 文件。

    md 限定在 path_prefix（docs/source/）下；.po 允许 docs/ 下任意
    locale 目录（翻译质量检测）。
    """
    with open(changed_file, encoding="utf-8") as f:
        names = [ln.strip().lstrip('"') for ln in f if ln.strip()]
    keep, deleted = [], []
    for n in dict.fromkeys(names):
        n = n.replace("\\", "/")
        lower = n.lower()
        is_md = lower.endswith(".md") and n.startswith(path_prefix)
        is_po = lower.endswith(".po") and n.startswith("docs/")
        if not is_md and not is_po:
            continue
        if os.path.isfile(os.path.join(repo_root, n)):
            keep.append(n)
        else:
            deleted.append(n)
    return keep, deleted


def launch_scripts(doc_path, temp_dir, std_dir, skip_link):
    """并行启动三个确定性前置脚本，返回 {name: (Popen, out_json)}。"""
    procs = {}
    punct_py = os.path.join(std_dir, PUNCT_CHECKER)
    prescan_py = os.path.join(std_dir, SECURITY_CHECKER)
    security_dir = os.path.join(std_dir, SECURITY_RULES_DIR)
    link_dir = os.path.join(std_dir, LINK_CHECKER_DIR)

    try:
        procs["punctuation"] = (
            subprocess.Popen(
                [sys.executable, punct_py, doc_path, "--out", os.path.join(temp_dir, SCRIPT_JSONS["punctuation"])],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
            SCRIPT_JSONS["punctuation"],
        )
    except OSError as e:
        print(f"[run_eval] 标点脚本启动失败: {e}", file=sys.stderr)

    try:
        procs["security"] = (
            subprocess.Popen(
                [
                    sys.executable,
                    prescan_py,
                    "--doc",
                    doc_path,
                    "--rules-dir",
                    security_dir,
                    "--out",
                    os.path.join(temp_dir, SCRIPT_JSONS["security"]),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
            SCRIPT_JSONS["security"],
        )
    except OSError as e:
        print(f"[run_eval] 安全脚本启动失败: {e}", file=sys.stderr)

    if skip_link:
        print("[run_eval] 已配置跳过链接检查")
    elif not os.path.isdir(os.path.join(link_dir, "node_modules")):
        print(
            "[run_eval][warn] link-checker 依赖未安装（node_modules 缺失），"
            "跳过链接检查；可在 workflow 中执行 npm ci 启用",
            file=sys.stderr,
        )
    else:
        try:
            procs["link"] = (
                subprocess.Popen(
                    [
                        "node",
                        "link-checker.js",
                        doc_path,
                        "--all-links",
                        "--out",
                        os.path.join(temp_dir, SCRIPT_JSONS["link"]),
                    ],
                    cwd=link_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ),
                SCRIPT_JSONS["link"],
            )
        except OSError as e:
            print(f"[run_eval] 链接脚本启动失败: {e}", file=sys.stderr)
    return procs


def wait_scripts(procs, temp_dir):
    """轮询脚本产物，返回 {name: ok|failed|timeout|skipped}。"""
    status = {}
    deadline = time.time() + SCRIPT_WATCH_SEC
    pending = dict(procs)
    while pending and time.time() < deadline:
        for name, (proc, out_json) in list(pending.items()):
            if os.path.isfile(os.path.join(temp_dir, out_json)):
                status[name] = "ok"
                del pending[name]
            elif proc.poll() is not None:
                status[name] = "ok" if os.path.isfile(os.path.join(temp_dir, out_json)) else "failed"
                del pending[name]
        if pending:
            time.sleep(POLL_INTERVAL)
    for name in pending:
        status[name] = "timeout"
    return status


def valid_problem_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return isinstance(obj, dict) and isinstance(obj.get("problems"), list)
    except (OSError, json.JSONDecodeError):
        return False


def run_merge(temp_dir, raw_path, expect, merge_py):
    """执行 merge --expect；退出码 2 时剔除无效期望文件重试（最多 3 次）。"""
    current = list(expect)
    dropped = []
    for _ in range(3):
        cmd = [
            sys.executable,
            merge_py,
            "--temp-dir",
            temp_dir,
            "--output",
            raw_path,
            "--expect",
            ",".join(current),
            "--quiet",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return current, dropped, None
        if r.returncode != 2:
            return current, dropped, f"merge 异常退出({r.returncode}): {r.stderr[-400:]}"
        bad = [
            f
            for f in current
            if not os.path.isfile(os.path.join(temp_dir, f)) or not valid_problem_json(os.path.join(temp_dir, f))
        ]
        if not bad:
            return current, dropped, "merge expectation check failed but no invalid file identified"
        dropped.extend(bad)
        current = [f for f in current if f not in bad]
    return current, dropped, "merge expectation still failing after dropping files"


def eval_one(rel, cfg):
    """评估单个文档，返回 summary 条目。"""
    skill_root = cfg["skill_root"]
    repo_root = cfg["repo_root"]
    std_dir = standard_dir(skill_root)
    merge_py = os.path.join(doc_evaluator_scripts(skill_root), MERGE_SCRIPT)
    gen_report_py = os.path.join(doc_evaluator_scripts(skill_root), GEN_REPORT_SCRIPT)

    doc_path = os.path.join(repo_root, rel)
    stem = safe_name(os.path.splitext(os.path.basename(rel))[0])
    ts = time.strftime("%Y%m%d%H%M%S")
    report_dir = os.path.join(cfg["out_dir"], f"report_{stem}")
    temp_dir = os.path.join(report_dir, f"temp-{ts}")
    os.makedirs(temp_dir, exist_ok=True)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(temp_dir, "detect_start.txt"), "w", encoding="utf-8") as f:
        f.write(start_time)

    warnings = []
    # .po 翻译文件只走 LLM 翻译质量检测；确定性脚本面向 md，不适用
    is_po = rel.lower().endswith(".po")
    procs = {} if is_po else launch_scripts(doc_path, temp_dir, std_dir, cfg["skip_link"])

    import llm_eval

    llm_expect, llm_warns = llm_eval.eval_doc(doc_path, temp_dir, std_dir, cfg["dims"], {**cfg, "repo_root": repo_root})
    warnings.extend(llm_warns)

    script_status = {} if is_po else wait_scripts(procs, temp_dir)
    for name, st in script_status.items():
        if st != "ok":
            warnings.append(f"prescan script {name} {st} (detection item falls back to LLM-only)")

    # 期望清单 = LLM 逐 H2 文件 + 就绪的脚本产物（就绪校验：JSON 含 problems 键）
    expect = list(llm_expect)
    for name, out_json in SCRIPT_JSONS.items():
        p = os.path.join(temp_dir, out_json)
        if os.path.isfile(p) and valid_problem_json(p):
            expect.append(out_json)

    raw_path = os.path.join(temp_dir, "problems_raw.json")
    expect, dropped, merge_err = run_merge(temp_dir, raw_path, expect, merge_py)
    if dropped:
        warnings.append("merge dropped invalid detection files: " + ", ".join(dropped))
    if merge_err or not os.path.isfile(raw_path):
        warnings.append(merge_err or "problems_raw.json not generated")
        return {"path": rel, "status": "merge_failed", "warnings": warnings}

    # 分数先读出（gen_report --cleanup-temp 会删 temp）
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    raw_copy = os.path.join(report_dir, "problems_raw.json")
    shutil.copyfile(raw_path, raw_copy)

    cmd = [
        sys.executable,
        gen_report_py,
        "--input",
        raw_path,
        "--output-dir",
        report_dir,
        "--doc-name",
        stem,
        "--doc-path",
        doc_path,
        "--model",
        cfg["model"],
        "--prescan",
        "是",
        "--no-log",
        "--cleanup-temp",
        "--detect-start-time",
        start_time,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        warnings.append(f"gen_report failed ({r.returncode}), temp dir kept: {r.stderr[-300:]}")

    return {
        "path": rel,
        "status": "ok",
        "total_score": raw.get("total_score"),
        "grade": raw.get("grade"),
        "dim_scores": raw.get("dim_scores"),
        "severity_count": raw.get("severity_count"),
        "problem_count": len(raw.get("problems", [])),
        "problems_file": os.path.relpath(raw_copy, cfg["out_dir"]).replace(os.sep, "/"),
        "warnings": warnings,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CI 文档质量评测编排器")
    parser.add_argument("--skill-root", required=True, help="doc-quality skill 仓库根")
    parser.add_argument("--repo-root", required=True, help="被检仓库根")
    parser.add_argument("--changed-file", required=True, help="变更文件清单（每行一个路径）")
    parser.add_argument("--path-prefix", default="docs/source/", help="仅评估该前缀下的 md（默认 docs/source/）")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--pr", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = {
        "skill_root": args.skill_root,
        "repo_root": os.path.abspath(args.repo_root),
        "repo": args.repo,
        "head_sha": args.head_sha,
        "pr": args.pr,
        "out_dir": os.path.abspath(args.out_dir),
        "model": os.environ.get("DOC_EVAL_MODEL", ""),
        "base_url": os.environ.get("DOC_EVAL_BASE_URL", ""),
        "api_key": os.environ.get("DOC_EVAL_API_KEY", ""),
        "dims": [
            d.strip() for d in os.environ.get("DOC_EVAL_DIMENSIONS", "正确性,完整性,易理解").split(",") if d.strip()
        ],
        "max_files": int(os.environ.get("DOC_EVAL_MAX_FILES", "8")),
        "max_lines": int(os.environ.get("DOC_EVAL_MAX_LINES", "700")),
        "concurrency": int(os.environ.get("DOC_EVAL_CONCURRENCY", "6")),
        "max_tokens": int(os.environ.get("DOC_EVAL_MAX_TOKENS", "8192")),
        "skip_link": os.environ.get("DOC_EVAL_SKIP_LINK", "") == "1",
    }
    if not (cfg["base_url"] and cfg["api_key"] and cfg["model"]):
        print("[run_eval] 缺少 DOC_EVAL_BASE_URL / DOC_EVAL_API_KEY / DOC_EVAL_MODEL", file=sys.stderr)
        return 2

    # 校验 skill 仓关键路径，结构变动时第一时间定位缺失项
    required = [
        os.path.join(args.skill_root, SKILL_STANDARD_DIR, REGISTRY_REL),
        os.path.join(args.skill_root, SKILL_TOOLS_DIR, MERGE_SCRIPT),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"[run_eval] skill 仓结构变动，缺失关键路径: {missing}", file=sys.stderr)
        return 1

    os.makedirs(cfg["out_dir"], exist_ok=True)

    keep, deleted = read_changed(args.changed_file, cfg["repo_root"], args.path_prefix)
    skipped_overflow = keep[cfg["max_files"] :]
    files = keep[: cfg["max_files"]]
    print(f"[run_eval] 待评估 {len(files)} 个文件（跳过超限 {len(skipped_overflow)}，已删除 {len(deleted)}）")

    sys.path.insert(0, CI_DIR)
    import gen_comment  # noqa: F401
    import llm_eval  # noqa: F401

    entries = []
    for rel in files:
        print(f"\n[run_eval] ===== 评估 {rel} =====")
        try:
            entries.append(eval_one(rel, cfg))
        except Exception as e:  # noqa: BLE001 —— 单文件失败不拖垮整批
            print(f"[run_eval] 评估 {rel} 失败: {e}", file=sys.stderr)
            entries.append({"path": rel, "status": "error", "warnings": [f"evaluation error: {e}"]})

    summary = {
        "pr": cfg["pr"],
        "repo": cfg["repo"],
        "head_sha": cfg["head_sha"],
        "model": cfg["model"],
        "dims": cfg["dims"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": entries,
        "skipped_overflow": skipped_overflow,
        "deleted_only": deleted,
    }
    summary_path = os.path.join(cfg["out_dir"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    from gen_comment import generate

    comment_path = os.path.join(cfg["out_dir"], "comment.md")
    generate(summary_path, comment_path)
    print(f"\n[run_eval] 完成：{summary_path}\n[run_eval] 评论: {comment_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, CI_DIR)
    sys.exit(main())
