#!/usr/bin/env python3
"""
analyze_failure_report.py  (NEW FILE - add to .github/workflows/scripts/)

Cross-reference CI test failures with hitest recommendations.

Pipeline:
  1. Scan each .txt log file for "short test summary info" block
  2. Extract FAILED lines from that block only (no full-file scan)
  3. Read hitest recommended_pytest_paths.txt
  4. Match: exact match + file-level match
  5. Generate a Markdown report

Usage:
  python analyze_failure_report.py --log-dir LOG_DIR --hitest-file RECOMMENDED.txt [--output report.md]
"""

import argparse
import contextlib
import sys
from pathlib import Path

import regex as re

# ============================================================
#  Utility: strip CI log noise
# ============================================================


def strip_ansi(text):
    """Remove ANSI color codes like \x1b[31m, \x1b[0m, etc."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def strip_timestamp(line):
    """Remove GitHub Actions timestamp prefix: YYYY-MM-DDTHH:MM:SS.fffffffZ"""
    return re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+", "", line)


def clean_line(line):
    """Strip BOM, ANSI codes, and timestamp from one log line."""
    line = line.lstrip("\ufeff")  # UTF-8 BOM marker
    return strip_ansi(strip_timestamp(line)).strip()


# ============================================================
#  Step 1: Extract FAILED tests from log files
# ============================================================


def extract_failed_from_logs(log_dir):
    """
    Recursively scan .log and .txt log files:
      - Locate "short test summary info" marker
      - Read subsequent lines until the next "=====" separator
      - Match "FAILED tests/...::..." lines
      - Deduplicate across all files
    """
    base = Path(log_dir)
    if not base.is_dir():
        print(f"::warning:: Log directory not found: {log_dir}")
        return []

    FAILED_PAT = re.compile(r"^FAILED\s+(tests/\S+?\.py::\S+?)\s")
    SEP_PAT = re.compile(r"^=+\s")

    all_failed = []
    seen = set()

    # Scan both real .log files (from run_selected_tests.sh) and mock .txt files
    candidates = []
    candidates.extend(base.rglob("*.log"))
    candidates.extend(base.rglob("*.txt"))
    for candidate in sorted(candidates):
        if candidate.suffix == ".txt" and "run-selected-tests" not in candidate.name:
            continue
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            print(f"::warning:: Cannot read {candidate.name}: {exc}")
            continue

        in_summary = False
        for line in lines:
            text = clean_line(line)

            # Enter: found the bookmark
            if "short test summary info" in text:
                in_summary = True
                continue

            if not in_summary:
                continue

            # Exit: hit the separator line ("======= 2 failed, 100 passed =======")
            if SEP_PAT.match(text):
                in_summary = False
                continue

            # Collect: FAILED line inside the block
            m = FAILED_PAT.match(text)
            if m:
                tp = m.group(1)
                if tp not in seen:
                    seen.add(tp)
                    all_failed.append(tp)

    return all_failed


# ============================================================
#  Step 2: Read hitest recommendations
# ============================================================


def read_recommended(hitest_file):
    """
    hitest.sh outputs one pytest path per line, e.g.:
        tests/ops/test_matmul.py::test_bf16
        tests/layers/test_attention.py
    """
    path = Path(hitest_file)
    if not path.exists():
        print(f"::warning:: Hitest file not found: {hitest_file}")
        return []
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff")
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("ERROR")]


# ============================================================
#  Step 3: Match
# ============================================================


def match_failed_vs_recommended(failed, recommended):
    """
    Two-level matching:
      Level 1 - Exact: "tests/foo.py::test_bar" in both lists
      Level 2 - File-level: recommended "tests/foo.py" (no function)
                  matches failed "tests/foo.py::anything"

    Returns {"hit": [...], "miss": [...], "untested": [...]}
      hit:       failed AND recommended
      miss:      failed but NOT recommended
      untested:  recommended but NOT in failed list
    """
    rec_set = set(recommended)

    # Map: file_path -> original recommended string
    rec_files = {}
    for r in recommended:
        file_part = r.split("::")[0] if "::" in r else r
        rec_files[file_part] = r

    hit = []
    miss = []
    hit_set = set()

    for f in failed:
        matched = False

        # Exact match
        if f in rec_set:
            hit.append(f)
            hit_set.add(f)
            matched = True
        else:
            # File-level match
            file_part = f.split("::")[0] if "::" in f else f
            if file_part in rec_files:
                hit.append(f)
                hit_set.add(f)
                matched = True

        if not matched:
            miss.append(f)

    # Recommended but not failed
    failed_files = {f.split("::")[0] if "::" in f else f for f in failed}
    untested = []
    for r in recommended:
        rf = r.split("::")[0] if "::" in r else r
        if rf not in failed_files and r not in hit_set:
            untested.append(r)

    return {"hit": hit, "miss": miss, "untested": untested}


# ============================================================
#  Step 4: Generate Markdown report
# ============================================================


def generate_report(failed, recommended, matched, log_dir, hitest_source="none"):
    """Produce a Markdown summary table."""
    hit = matched["hit"]
    miss = matched["miss"]
    untested = matched["untested"]

    out = []
    out.append("# Test Failure vs Hitest Recommendation Report")
    out.append("")
    out.append(f"**Log source**: `{log_dir}`")
    out.append("")

    # Recommendation source indicator
    if hitest_source == "artifact":
        out.append("> **[来源: Artifacts]** 推荐用例来自 hitest.yaml 上传（优先级最高）")
    elif hitest_source == "committed":
        out.append("> **[来源: 本地文件]** 推荐用例来自仓库 txt（Artifacts 未找到，回退方案）")
    else:
        out.append("> **[来源: 无]** 未找到推荐用例")
    out.append("")

    # ================================================================
    #  Section 1: Full Failed Test List
    # ================================================================
    out.append("---")
    out.append("")
    out.append(f"## 失败用例（共 {len(failed)} 个）")
    out.append("")
    if failed:
        for i, t in enumerate(failed, 1):
            tag = " **[命中推荐]**" if t in hit else " **[未命中推荐]**"
            out.append(f"{i}. `{t}`{tag}")
        out.append("")
    else:
        out.append("> 没有失败用例")
        out.append("")

    # ================================================================
    #  Section 2: Full Recommended Test List
    # ================================================================
    out.append("---")
    out.append("")
    out.append(f"## 推荐用例（共 {len(recommended)} 个）")
    out.append("")
    if recommended:
        failed_file_set = {f.split("::")[0] if "::" in f else f for f in failed}
        for i, r in enumerate(recommended, 1):
            rf = r.split("::")[0] if "::" in r else r
            tag = " **[已失败]**" if rf in failed_file_set else ""
            out.append(f"{i}. `{r}`{tag}")
        out.append("")
    else:
        out.append("> 无推荐用例")
        out.append("")

    # ================================================================
    #  Section 3: Core Conclusion
    # ================================================================
    out.append("---")
    out.append("")
    out.append("## 核心结论")
    out.append("")
    if not failed:
        out.append("> 本次 CI 没有失败用例，无需对比推荐列表。")
    elif len(miss) == 0:
        out.append("> **所有失败用例均在推荐用例范围内。**")
    else:
        total_failed = len(failed)
        out.append(f"> **有 {len(miss)}/{total_failed} 个失败用例不在推荐用例范围内。**")
    out.append("")

    # ================================================================
    #  Section 4: Detail table
    # ================================================================
    out.append("| 类别 | 数量 |")
    out.append("|---|---|")
    out.append(f"| 失败且命中推荐 | {len(hit)} |")
    out.append(f"| 失败但未命中推荐 | {len(miss)} |")
    out.append(f"| 推荐但未失败 | {len(untested)} |")
    out.append("")

    if hit:
        out.append("## 失败且命中推荐")
        out.append("")
        out.append("| # | Failed test |")
        out.append("|---|---|")
        for i, t in enumerate(hit, 1):
            out.append(f"| {i} | `{t}` |")
        out.append("")

    if miss:
        out.append("## 失败但未命中推荐")
        out.append("")
        out.append("> 可能原因：未覆盖的模块、环境问题、不稳定测试。")
        out.append("")
        for t in miss:
            out.append(f"- `{t}`")
        out.append("")

    if untested:
        out.append("## 推荐但未失败")
        out.append("")
        out.append("> 这些用例被推荐但本次未失败（已通过或未执行）。")
        out.append("")
        for t in untested:
            out.append(f"- `{t}`")
        out.append("")

    if not hit and not miss:
        out.append("## 未检测到失败")
        out.append("")

    out.append("---")
    out.append("*Generated by analyze_failure_report.py*")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Cross-reference CI test failures with hitest recommendations")
    parser.add_argument("--log-dir", required=True, help="Directory containing CI .txt log files")
    parser.add_argument("--hitest-file", required=True, help="Path to hitest recommended_pytest_paths.txt")
    parser.add_argument(
        "--output", default="failure_report.md", help="Output Markdown report path (default: failure_report.md)"
    )
    parser.add_argument(
        "--hitest-source",
        default="none",
        choices=["artifact", "committed", "none"],
        help="Where recommendations came from",
    )
    args = parser.parse_args()

    # For Windows console: force UTF-8 if possible
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 50)
    print("Step 1: Extract failed tests from CI logs")
    print("=" * 50)
    failed = extract_failed_from_logs(args.log_dir)
    print(f"Failed: {len(failed)}")

    print()
    print("=" * 50)
    print("Step 2: Read hitest recommendations")
    print("=" * 50)
    recommended = read_recommended(args.hitest_file)
    print(f"Recommended: {len(recommended)}")

    print()
    print("=" * 50)
    print("Step 3: Match")
    print("=" * 50)
    matched = match_failed_vs_recommended(failed, recommended)
    print(f"Hit (failed + recommended): {len(matched['hit'])}")
    print(f"Miss (failed, not recommended): {len(matched['miss'])}")
    print(f"Untested (recommended, no failure): {len(matched['untested'])}")

    report = generate_report(failed, recommended, matched, args.log_dir, args.hitest_source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print()
    print(f"Report => {output_path}")
    print()

    # Print report to stdout (safe fallback for Windows encoding)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
