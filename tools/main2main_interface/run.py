# SPDX-License-Identifier: Apache-2.0
"""Run a pinned source-only main2main scan with verified report reuse.

The analyzer is a separate checkout. This entry never installs vLLM, invokes
kickoff, modifies source, calls a model, or publishes to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from qa_markdown import render_qa

REPORT_FILES = (
    "main2main-range-report.json",
    "main2main-range-report.md",
    "main2main-introduced-breaks.csv",
    "main2main-all-findings.csv",
)


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def require_sha(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Expected a full lowercase 40-character commit SHA")
    return value


def clean_head(root: Path, expected: str) -> None:
    require_sha(expected)
    if git(root, "rev-parse", "HEAD") != expected:
        raise ValueError(f"Checkout SHA mismatch: {root}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Source checkout is dirty: {root}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(inputs: dict) -> str:
    raw = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return "main2main-interface-v1-" + hashlib.sha256(raw).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    roots = {name: getattr(args, name).resolve() for name in ("engine_root", "vllm_root", "ascend_root")}
    output = args.output.resolve()
    for root in roots.values():
        if output == root or root in output.parents:
            raise ValueError("Output must be outside all source checkouts")
    clean_head(roots["engine_root"], args.engine_sha)
    clean_head(roots["vllm_root"], args.new)
    clean_head(roots["ascend_root"], args.ascend_sha)
    require_sha(args.old)
    git(roots["vllm_root"], "cat-file", "-e", f"{args.old}^{{commit}}")
    git(roots["vllm_root"], "merge-base", "--is-ancestor", args.old, args.new)
    engine_tree = git(roots["engine_root"], "rev-parse", "HEAD:tools/vllm_interface_contracts")
    marker = roots["ascend_root"] / ".github/vllm-main-verified.commit"
    marker_value = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    if marker_value != args.old:
        raise ValueError("Resolved old SHA does not match the selected Ascend marker")
    resolution = {}
    if args.resolved_range:
        resolution = json.loads(args.resolved_range.read_text(encoding="utf-8"))
        for key, expected in (
            ("vllm_old_sha", args.old),
            ("vllm_new_sha", args.new),
            ("vllm_ascend_sha", args.ascend_sha),
        ):
            if resolution.get(key) != expected:
                raise ValueError(f"Resolved workflow range mismatch: {key}")
    identity = {
        "schema_version": 1,
        "engine_sha": args.engine_sha,
        "engine_tree": engine_tree,
        "runner_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "vllm_old_sha": args.old,
        "vllm_new_sha": args.new,
        "vllm_ascend_sha": args.ascend_sha,
        "scenario": "main2main",
        "profile": "exact-contracts",
        "external_sources": {},
        "index_workers": 1,
        "analysis_workers": 1,
    }
    metadata = {
        "inputs": identity,
        "cache_key": cache_key(identity),
        "roots": {name: str(root) for name, root in roots.items()},
        "baseline_marker": marker_value,
        "range_mode": "marker",
        "resolution": resolution,
        "qa": {"status": "disabled", "model_calls": 0},
    }
    json_write(output / "run-metadata.json", metadata)
    json_write(output / "run-status.json", {"scan": "pending", "qa": "disabled"})
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"cache_key={metadata['cache_key']}\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def verify_report(directory: Path, inputs: dict) -> dict:
    for name in REPORT_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete report: missing {name}")
    report = json.loads((directory / REPORT_FILES[0]).read_text(encoding="utf-8"))
    metadata = report["metadata"]
    for key in ("vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha", "scenario", "profile", "external_sources"):
        if metadata.get(key) != inputs[key]:
            raise ValueError(f"Report input mismatch: {key}")
    capabilities = metadata["analysis_plan"]["capabilities"]
    for capability in (
        "monkey_patch",
        "override",
        "direct_import",
        "direct_call",
        "inheritance_mro",
        "direct_attribute",
        "inherited_state",
    ):
        if capabilities[capability]["state"] != "analyzed":
            raise ValueError(f"Required main2main capability not analyzed: {capability}")
    if not isinstance(report.get("findings"), list) or not isinstance(report.get("summary"), dict):
        raise ValueError("Invalid report findings or summary")
    return report


def cached_report(directory: Path, inputs: dict) -> dict:
    manifest = json.loads((directory / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest["inputs"] != inputs or manifest["cache_key"] != cache_key(inputs):
        raise ValueError("Cache fingerprint mismatch")
    for name in REPORT_FILES:
        if (directory / name).is_symlink() or digest(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Cached report checksum mismatch: {name}")
    return verify_report(directory, inputs)


def scan(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    inputs = metadata["inputs"]
    roots = {name: Path(value) for name, value in metadata["roots"].items()}
    for name, key in (("engine_root", "engine_sha"), ("vllm_root", "vllm_new_sha"), ("ascend_root", "vllm_ascend_sha")):
        clean_head(roots[name], inputs[key])
    if inputs["vllm_old_sha"] == inputs["vllm_new_sha"]:
        message = (
            "# Main2Main QA 审阅材料\n\n无需升级：选定 Ascend marker 与 vLLM 目标一致。\n\n"
            f"vLLM：`{inputs['vllm_new_sha']}`\n\n"
            f"Ascend：`{inputs['vllm_ascend_sha']}`\n\n"
            "扫描未启动；QA 未调用，模型调用为 0。\n"
        )
        (output / "qa-review.md").write_text(message, encoding="utf-8")
        (output / "summary.md").write_text(message, encoding="utf-8")
        json_write(
            output / "run-status.json",
            {
                "scan": "skipped_no_changes",
                "qa": "disabled",
                "model_calls": 0,
                "source_unchanged": True,
                "qa_input": "qa-review.md",
            },
        )
        return
    cache = args.cache_dir.resolve()
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("Cache and output must be separate directories")
    for root in roots.values():
        if cache == root or root in cache.parents:
            raise ValueError("Cache must be outside source checkouts")
    start = time.monotonic()
    cache_status = "bypassed" if args.force_rescan else "miss"
    report = None
    if not args.force_rescan and cache.exists():
        try:
            report = cached_report(cache, inputs)
        except (OSError, ValueError, KeyError, TypeError) as error:
            cache_status = "invalid"
            metadata["cache_rejection"] = str(error)
        else:
            cache_status = "hit"
    destination = output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    if report is not None:
        for name in REPORT_FILES:
            shutil.copyfile(cache / name, destination / name)
    else:
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "tools.vllm_interface_contracts",
            "analyze-range",
            "--vllm-root",
            str(roots["vllm_root"]),
            "--ascend-root",
            str(roots["ascend_root"]),
            "--expect-ascend-sha",
            inputs["vllm_ascend_sha"],
            "--old",
            inputs["vllm_old_sha"],
            "--new",
            inputs["vllm_new_sha"],
            "--scenario",
            "main2main",
            "--profile",
            "exact-contracts",
            "--output-dir",
            str(destination),
            "--fail-on",
            "never",
            "--no-cache",
            "--analysis-workers",
            "1",
            "--index-workers",
            "1",
        ]
        # No persistent executable/pickle index is restored. Only verified JSON,
        # Markdown and CSV report files are cached across runs.
        with (output / "scan.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=roots["engine_root"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout_seconds,
            )
        report = verify_report(destination, inputs)
    for name, key in (("engine_root", "engine_sha"), ("vllm_root", "vllm_new_sha"), ("ascend_root", "vllm_ascend_sha")):
        clean_head(roots[name], inputs[key])
    # Copy into a dedicated sibling cache directory; never delete source paths.
    if cache.is_symlink():
        raise ValueError("Cache directory must not be a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FILES:
        target = cache / name
        if target.is_symlink():
            target.unlink()
        shutil.copyfile(destination / name, target)
    manifest = {
        "inputs": inputs,
        "cache_key": cache_key(inputs),
        "sha256": {name: digest(cache / name) for name in REPORT_FILES},
    }
    json_write(cache / "cache-manifest.json", manifest)
    metadata.update(
        cache_status=cache_status,
        elapsed_seconds=round(time.monotonic() - start, 3),
        capabilities=report["metadata"]["analysis_plan"]["capabilities"],
        phase_timings=report["metadata"].get("stage_timings_seconds", report["metadata"].get("timings_seconds", {})),
    )
    json_write(output / "run-metadata.json", metadata)
    json_write(
        output / "run-status.json",
        {
            "scan": "completed",
            "qa": "disabled",
            "model_calls": 0,
            "cache_status": cache_status,
            "summary": report["summary"],
            "source_unchanged": True,
            "qa_input": "qa-review.md",
        },
    )
    (output / "qa-review.md").write_text(
        render_qa(report, inputs, roots, metadata.get("resolution", {})), encoding="utf-8"
    )
    summary = (
        "# Main2Main interface scan\n\n"
        f"- Scan: completed; cache: **{cache_status}**\n"
        "- QA: **disabled**, model calls: **0**\n"
        "- Runner: CPU source analysis; source checkouts unchanged\n"
        f"- Range: `{inputs['vllm_old_sha']}` → `{inputs['vllm_new_sha']}`\n"
        f"- Ascend baseline: `{inputs['vllm_ascend_sha']}`\n"
        f"- Engine: `{inputs['engine_sha']}`\n"
        f"- Summary: `{json.dumps(report['summary'], ensure_ascii=False)}`\n\n"
        "QA input: qa-review.md (single Markdown). No QA verdict has been produced. "
        "Internal diagnostics remain separate from the QA input.\n"
    )
    (output / "summary.md").write_text(summary, encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"cache_status={cache_status}\n")
    print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("prepare")
    for name in ("engine-root", "vllm-root", "ascend-root"):
        setup.add_argument("--" + name, type=Path, required=True)
    for name in ("engine-sha", "old", "new", "ascend-sha"):
        setup.add_argument("--" + name, required=True)
    setup.add_argument("--resolved-range", type=Path)
    execute = subparsers.add_parser("scan")
    execute.add_argument("--cache-dir", type=Path, required=True)
    execute.add_argument("--force-rescan", action="store_true")
    execute.add_argument("--timeout-seconds", type=int, default=2400)
    for child in (setup, execute):
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else scan)(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        json_write(args.output / "run-status.json", {"scan": "failed", "qa": "disabled", "error": str(error)})
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
