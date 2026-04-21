#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Helpers for translating test commands into bisect workflow inputs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import regex as re

DEFAULT_ENV = {
    "kind": "",
    "runner": "linux-aarch64-a3-4",
    "image": "m.daocloud.io/quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11",
    "test_type": "e2e",
}
PATH_KIND_RULES = [
    (r"tests/e2e/310p/multicard/", "e2e_310p_4cards"),
    (r"tests/e2e/310p/singlecard/", "e2e_310p_singlecard"),
    (r"tests/e2e/multicard/4-cards/", "e2e_4cards"),
    (r"tests/e2e/multicard/2-cards/", "e2e_2cards"),
    (r"tests/e2e/singlecard/", "e2e_singlecard"),
    (r"tests/ut/", "ut"),
]

# Environment details come from the canonical unit/full workflows at runtime.
COMMON_E2E_SOURCE = {
    "test_type": "e2e",
    "workflow": ".github/workflows/_e2e_test.yaml",
    "prepare_steps": ["Config mirrors", "Install system dependencies"],
    "vllm_install_step": "Install vllm-project/vllm from source",
    "ascend_install_step": "Install vllm-project/vllm-ascend",
}
KIND_SOURCES = {
    "ut": {
        "test_type": "ut",
        "workflow": ".github/workflows/_unit_test.yaml",
        "job": "unit-test",
        "caller_workflow": ".github/workflows/pr_test_light.yaml",
        "caller_job": "ut",
        "prepare_steps": ["Install packages"],
        "vllm_install_step": "Install vllm-project/vllm from source",
        "ascend_install_step": "Install vllm-project/vllm-ascend",
        "test_step": "Run unit test",
    },
    "e2e_singlecard": {
        **COMMON_E2E_SOURCE,
        "job": "e2e-full",
        "caller_workflow": ".github/workflows/pr_test_full.yaml",
        "caller_job": "e2e-test",
        "test_step": "Run e2e test",
    },
    "e2e_2cards": {
        **COMMON_E2E_SOURCE,
        "job": "e2e-2-cards-full",
        "test_step": "Run vllm-project/vllm-ascend test (full)",
    },
    "e2e_4cards": {
        **COMMON_E2E_SOURCE,
        "job": "e2e-4-cards-full",
        "test_step": "Run vllm-project/vllm-ascend test for V1 Engine",
    },
    "e2e_310p_singlecard": {**COMMON_E2E_SOURCE, "job": "e2e_310p", "test_step": "Run vllm-project/vllm-ascend test"},
    "e2e_310p_4cards": {
        **COMMON_E2E_SOURCE,
        "job": "e2e_310p-4cards",
        "test_step": "Run vllm-project/vllm-ascend test",
    },
}

COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")
TEST_PATH_RE = re.compile(r"\b(tests/[-\w/]+\.py(?:::[\w_]+)*)")
_MANIFEST_CACHE: dict[str, dict] | None = None
_REPO_ROOT: Path | None = None


def _get_repo_root(cwd: Path | None = None) -> Path:
    global _REPO_ROOT
    if _REPO_ROOT is not None and cwd is None:
        return _REPO_ROOT
    candidates = []
    if env_root := os.environ.get("VLLM_BENCHMARKS_REPO_ROOT"):
        candidates.append(Path(env_root).expanduser().resolve())
    start = (cwd or Path.cwd()).resolve()
    candidates.extend([start, *start.parents])
    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / ".github" / "workflows" / "pr_test_light.yaml").is_file():
            if cwd is None:
                _REPO_ROOT = candidate
            return candidate
    raise RuntimeError(
        "Cannot detect repo root. Run from the repository, set VLLM_BENCHMARKS_REPO_ROOT, "
        "or keep bisect_helper.py under tools."
    )


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for workflow parsing paths") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_inputs(value, with_inputs: dict[str, object] | None):
    if isinstance(value, str) and with_inputs:
        match = re.fullmatch(r"\${{\s*inputs\.([A-Za-z0-9_]+)\s*}}", value)
        if match:
            return with_inputs.get(match.group(1), value)
    if isinstance(value, dict):
        return {k: _resolve_inputs(v, with_inputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_inputs(v, with_inputs) for v in value]
    return value


def _normalize_env(env: dict | None) -> dict[str, str]:
    return {} if not isinstance(env, dict) else {str(k): str(v) for k, v in env.items()}


def _format_run_with_env(run: str | None, env: dict[str, str] | None = None) -> str:
    if not run:
        return ""
    exports = [f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items()]
    return "\n".join(exports + [run.strip()]) if exports else run.strip()


def _get_job(workflow: dict, job_name: str) -> dict:
    job = workflow.get("jobs", {}).get(job_name)
    if not isinstance(job, dict):
        raise KeyError(f"job not found: {job_name}")
    return job


def _get_step(job: dict, step_name: str) -> dict:
    step = next(
        (step for step in job.get("steps", []) if isinstance(step, dict) and step.get("name") == step_name), None
    )
    if step is None:
        raise KeyError(f"step not found: {step_name}")
    return step


def _get_caller_inputs(workflow_path: str | None, job_name: str | None) -> dict[str, object]:
    if not workflow_path or not job_name:
        return {}
    job = _get_job(_load_yaml(_get_repo_root() / workflow_path), job_name)
    with_inputs = job.get("with", {})
    return with_inputs if isinstance(with_inputs, dict) else {}


def _extract_profile(kind: str, config: dict) -> dict:
    # Emit only the fields consumed by bisect_vllm.yaml.
    workflow = _load_yaml(_get_repo_root() / config["workflow"])
    caller_inputs = _get_caller_inputs(config.get("caller_workflow"), config.get("caller_job"))
    job = _get_job(workflow, config["job"])
    container = job.get("container", {}) if isinstance(job.get("container"), dict) else {}
    workflow_env = _normalize_env(workflow.get("env"))
    container_env = _normalize_env(_resolve_inputs(container.get("env", {}), caller_inputs))
    effective_container_env = {**workflow_env, **container_env}
    prepare_runs = [
        _format_run_with_env(_resolve_inputs(_get_step(job, step_name).get("run", ""), caller_inputs))
        for step_name in config.get("prepare_steps", [])
    ]
    vllm_install_step = _get_step(job, config["vllm_install_step"])
    ascend_install_step = _get_step(job, config["ascend_install_step"])
    test_step = _get_step(job, config["test_step"])
    return {
        "kind": kind,
        "test_type": config["test_type"],
        "runner": str(_resolve_inputs(job.get("runs-on", DEFAULT_ENV["runner"]), caller_inputs)),
        "image": str(_resolve_inputs(container.get("image", DEFAULT_ENV["image"]), caller_inputs)),
        "effective_container_env": effective_container_env,
        "sys_deps": "\n".join(run for run in prepare_runs if run),
        "vllm_install": _format_run_with_env(_resolve_inputs(vllm_install_step.get("run", ""), caller_inputs)),
        "ascend_install": _format_run_with_env(
            _resolve_inputs(ascend_install_step.get("run", ""), caller_inputs),
            _normalize_env(_resolve_inputs(ascend_install_step.get("env", {}), caller_inputs)),
        ),
        "runtime_env": _normalize_env(_resolve_inputs(test_step.get("env", {}), caller_inputs)),
    }


def load_runtime_env_manifest() -> dict[str, dict]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = {kind: _extract_profile(kind, cfg) for kind, cfg in KIND_SOURCES.items()}
    return _MANIFEST_CACHE


def _detect_kind(test_cmd: str) -> str:
    return next((kind for pattern, kind in PATH_KIND_RULES if re.search(pattern, test_cmd)), "")


def _resolve_env_for_test_cmd(test_cmd: str) -> dict:
    manifest = load_runtime_env_manifest()
    return manifest.get(_detect_kind(test_cmd), DEFAULT_ENV.copy())


def get_vllm_install_for_test_cmd(test_cmd: str) -> str:
    return str(_resolve_env_for_test_cmd(test_cmd).get("vllm_install", "")).strip()


def get_ascend_install_for_test_cmd(test_cmd: str) -> str:
    return str(_resolve_env_for_test_cmd(test_cmd).get("ascend_install", "")).strip()


def get_commit_from_yaml(yaml_path: str, ref: str | None = None) -> str | None:
    if ref:
        try:
            repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
            rel_path = os.path.relpath(yaml_path, repo_root)
            content = subprocess.check_output(
                ["git", "show", f"{ref}:{rel_path}"], text=True, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return None
    else:
        try:
            content = Path(yaml_path).read_text()
        except FileNotFoundError:
            return None
    match = re.search(r"vllm_version:\s*\[([^\]]+)\]", content)
    if not match:
        return None
    return next(
        (entry for entry in (e.strip().strip("'\"") for e in match.group(1).split(",")) if COMMIT_HASH_RE.match(entry)),
        None,
    )


def get_pkg_location(pkg_name: str) -> str | None:
    try:
        output = subprocess.check_output(["pip", "show", pkg_name], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    editable_loc = location = None
    for line in output.splitlines():
        if line.startswith("Editable project location:"):
            editable_loc = line.split(":", 1)[1].strip()
        elif line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
    return editable_loc or location


def generate_report(
    bad_commit: str,
    good_commit: str,
    first_bad: str,
    first_bad_info: str,
    test_cmd: str,
    total_steps: int,
    total_commits: int,
    skipped: list[str] | None = None,
    log_entries: list[dict] | None = None,
) -> str:
    lines = [
        "## Bisect Result",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| First bad commit | `{first_bad}` |",
        f"| Link | https://github.com/vllm-project/vllm/commit/{first_bad} |",
        f"| Good commit | `{good_commit}` |",
        f"| Bad commit | `{bad_commit}` |",
        f"| Range | {total_commits} commits, {total_steps} bisect steps |",
        f"| Test command | `{test_cmd}` |",
        "",
        "### First Bad Commit Details",
        "```",
        first_bad_info,
        "```",
    ]
    if skipped:
        lines.extend(["", "### Skipped Commits", ""] + [f"- `{commit}`" for commit in skipped])
    if log_entries:
        lines.extend(["", "### Bisect Log", "", "| Step | Commit | Result |", "|------|--------|--------|"])
        lines.extend(
            f"| {idx} | `{entry.get('commit', '?')[:12]}` | {entry.get('result', '?')} |"
            for idx, entry in enumerate(log_entries, 1)
        )
    lines.extend(["", "---", "*Generated by `tools/bisect_vllm.sh`*"])
    return "\n".join(lines)


def build_batch_matrix(test_cmds_str: str) -> dict:
    # Group by kind so distinct workflow profiles never collapse together.
    cmds: list[str] = []
    seen_cmds: set[str] = set()
    for raw_cmd in test_cmds_str.split(";"):
        cmd = raw_cmd.strip()
        if not cmd or cmd in seen_cmds:
            continue
        seen_cmds.add(cmd)
        cmds.append(cmd)
    if not cmds:
        return {"include": []}
    manifest = load_runtime_env_manifest()
    all_container_env_keys = sorted(
        {key for profile in manifest.values() for key in profile["effective_container_env"]}
    )
    grouped: dict[tuple[str, str, str, str], dict] = {}
    for cmd in cmds:
        env = _resolve_env_for_test_cmd(cmd)
        key = (env["kind"], env["runner"], env["image"], env["test_type"])
        if key not in grouped:
            grouped[key] = {**env, "test_cmds": [cmd]}
            continue
        grouped[key]["test_cmds"].append(cmd)
        grouped[key]["effective_container_env"].update(env.get("effective_container_env", {}))
        grouped[key]["runtime_env"].update(env.get("runtime_env", {}))
    include = []
    for (_, runner, image, test_type), env in grouped.items():
        entry = {
            "group": env["kind"].replace("_", "-"),
            "runner": runner,
            "image": image,
            "test_type": test_type,
            "test_cmds": ";".join(env["test_cmds"]),
            "sys_deps": env.get("sys_deps", "echo 'no sys_deps configured'"),
            "vllm_install": env.get("vllm_install", "echo 'no vllm_install configured'"),
            "ascend_install": env.get("ascend_install", "echo 'no ascend_install configured'"),
            "runtime_env": json.dumps(env.get("runtime_env", {})),
        }
        entry.update({f"cenv_{key}": env["effective_container_env"].get(key, "") for key in all_container_env_keys})
        include.append(entry)
    include.sort(key=lambda entry: entry["group"])
    return {"include": include}


def _coalesce_first_bad(group_results: list[dict]) -> tuple[str, str, str]:
    successes = [group for group in group_results if group.get("status") == "success" and group.get("first_bad_commit")]
    if not successes:
        return "failed", "", ""
    unique_commits = {group.get("first_bad_commit", "") for group in successes}
    if len(unique_commits) > 1:
        return "ambiguous", "", ""
    first_bad_commit = successes[0].get("first_bad_commit", "")
    first_bad_commit_url = successes[0].get("first_bad_commit_url", "")
    if len(successes) == len(group_results):
        return "success", first_bad_commit, first_bad_commit_url
    return "partial_success", first_bad_commit, first_bad_commit_url


def build_bisect_result_json(
    *,
    caller_type: str,
    caller_run_id: str,
    bisect_run_id: str,
    good_commit: str,
    bad_commit: str,
    test_cmd: str,
    group_results: list[dict],
    skipped_commits: list[str] | None = None,
    log_entries: list[dict] | None = None,
) -> dict:
    status, first_bad_commit, first_bad_commit_url = _coalesce_first_bad(group_results)
    return {
        "caller_type": caller_type,
        "caller_run_id": caller_run_id,
        "bisect_run_id": bisect_run_id,
        "status": status,
        "good_commit": good_commit,
        "bad_commit": bad_commit,
        "test_cmd": test_cmd,
        "first_bad_commit": first_bad_commit,
        "first_bad_commit_url": first_bad_commit_url,
        "total_steps": 0,
        "total_commits": 0,
        "skipped_commits": skipped_commits or [],
        "log_entries": log_entries or [],
        "group_results": group_results,
    }


def _parse_summary_file(summary_path: Path, *, group: str) -> dict:
    if not summary_path.is_file():
        return {"group": group, "status": "failed", "first_bad_commit": "", "first_bad_commit_url": ""}
    text = summary_path.read_text(encoding="utf-8")
    first_bad_match = re.search(r"\| First bad commit \| `([^`]+)` \|", text)
    url_match = re.search(r"\| Link \| ([^|]+) \|", text)
    if first_bad_match:
        return {
            "group": group,
            "status": "success",
            "first_bad_commit": first_bad_match.group(1).strip(),
            "first_bad_commit_url": url_match.group(1).strip() if url_match else "",
        }
    return {"group": group, "status": "failed", "first_bad_commit": "", "first_bad_commit_url": ""}


def collect_group_results(results_dir: Path) -> list[dict]:
    group_results = []
    for summary_path in sorted(results_dir.rglob("bisect_summary*.md")):
        match = re.fullmatch(r"bisect_summary_(.+)\.md", summary_path.name)
        if match:
            group = match.group(1)
        elif summary_path.parent.name.startswith("bisect-result-"):
            group = summary_path.parent.name.removeprefix("bisect-result-")
        else:
            group = summary_path.stem.removeprefix("bisect_summary").lstrip("_") or "default"
        group_results.append(_parse_summary_file(summary_path, group=group))
    return group_results


def cmd_batch_matrix(args):
    matrix = build_batch_matrix(args.test_cmds)
    matrix_json = json.dumps(matrix, separators=(",", ":"))
    if args.output_format == "github":
        if github_output := os.environ.get("GITHUB_OUTPUT"):
            with open(github_output, "a", encoding="utf-8") as file:
                file.write(f"matrix={matrix_json}\n")
        print(f"matrix={matrix_json}")
        total_cmds = sum(len(group["test_cmds"].split(";")) for group in matrix["include"])
        print(f"Total: {len(matrix['include'])} group(s) from {total_cmds} command(s)")
        return
    print(json.dumps(matrix, indent=2))


def cmd_get_commit(args):
    # Default yaml_path keeps the helper usable for local/manual invocation.
    yaml_path = args.yaml_path
    if not yaml_path:
        try:
            repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        except subprocess.CalledProcessError:
            print("ERROR: Cannot determine repo root", file=sys.stderr)
            sys.exit(1)
        yaml_path = os.path.join(repo_root, ".github/workflows/pr_test_light.yaml")
    if commit := get_commit_from_yaml(yaml_path, ref=args.ref):
        print(commit)
        return
    suffix = f" at ref {args.ref}" if args.ref else ""
    print(f"ERROR: Could not extract vllm commit from {yaml_path}{suffix}", file=sys.stderr)
    sys.exit(1)


def cmd_report(args):
    # The report command appends to a markdown file for artifact collection.
    skipped = args.skipped.split(",") if args.skipped else None
    try:
        log_entries = json.loads(Path(args.log_file).read_text()) if args.log_file else None
    except (FileNotFoundError, json.JSONDecodeError):
        log_entries = None
    first_bad_info = args.first_bad_info or ""
    if args.first_bad_info_file:
        try:
            first_bad_info = Path(args.first_bad_info_file).read_text().strip()
        except FileNotFoundError:
            first_bad_info = "N/A"
    report = generate_report(
        bad_commit=args.bad_commit,
        good_commit=args.good_commit,
        first_bad=args.first_bad,
        first_bad_info=first_bad_info,
        test_cmd=args.test_cmd,
        total_steps=args.total_steps,
        total_commits=args.total_commits,
        skipped=skipped,
        log_entries=log_entries,
    )
    print(report)
    summary_md_path = Path(args.summary_output or "/tmp/bisect_summary.md")
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_md_path, "a" if summary_md_path.exists() else "w", encoding="utf-8") as file:
        match = TEST_PATH_RE.search(args.test_cmd)
        file.write(f"\n# bisect {(match.group(1) if match else args.test_cmd)}\n\n{report}\n")
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as file:
            file.write(report + "\n")


def cmd_vllm_location(_args):
    if loc := get_pkg_location("vllm"):
        print(loc)
        return
    print("ERROR: vllm not installed or pip show failed", file=sys.stderr)
    sys.exit(1)


def cmd_vllm_install(args):
    if install_cmd := get_vllm_install_for_test_cmd(args.test_cmd):
        print(install_cmd)
        return
    print(f"ERROR: No vllm install command found for test command: {args.test_cmd}", file=sys.stderr)
    sys.exit(1)


def cmd_ascend_install(args):
    if install_cmd := get_ascend_install_for_test_cmd(args.test_cmd):
        print(install_cmd)
        return
    print(f"ERROR: No vllm-ascend install command found for test command: {args.test_cmd}", file=sys.stderr)
    sys.exit(1)


def cmd_result_json(args):
    results_dir = Path(args.results_dir)
    group_results = collect_group_results(results_dir) if results_dir.exists() else []
    result = build_bisect_result_json(
        caller_type=args.caller_type,
        caller_run_id=args.caller_run_id,
        bisect_run_id=args.bisect_run_id,
        good_commit=args.good_commit,
        bad_commit=args.bad_commit,
        test_cmd=args.test_cmd,
        group_results=group_results,
    )
    output = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(description="Helper for vllm bisect automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Used by bisect_vllm.yaml set-params.
    p_batch = subparsers.add_parser(
        "batch-matrix", help="Build a GitHub Actions matrix from semicolon-separated test commands"
    )
    p_batch.add_argument("--test-cmds", required=True, help="Semicolon-separated test commands")
    p_batch.add_argument("--output-format", choices=["json", "github"], default="github", help="Output format")
    p_batch.set_defaults(func=cmd_batch_matrix)

    # Used by bisect_vllm.sh when good/bad commits are not passed explicitly.
    p_commit = subparsers.add_parser("get-commit", help="Extract vllm commit from workflow yaml")
    p_commit.add_argument("--yaml-path", default="", help="Path to workflow yaml (default: pr_test_light.yaml)")
    p_commit.add_argument("--ref", default=None, help="Git ref to read from (e.g. origin/main)")
    p_commit.set_defaults(func=cmd_get_commit)

    # Used by bisect_vllm.sh to render the final markdown summary.
    p_report = subparsers.add_parser("report", help="Generate bisect result report")
    p_report.add_argument("--good-commit", required=True)
    p_report.add_argument("--bad-commit", required=True)
    p_report.add_argument("--first-bad", required=True)
    p_report.add_argument("--first-bad-info", default=None)
    p_report.add_argument("--first-bad-info-file", default=None)
    p_report.add_argument("--test-cmd", required=True)
    p_report.add_argument("--total-steps", type=int, required=True)
    p_report.add_argument("--total-commits", type=int, required=True)
    p_report.add_argument("--skipped", default=None)
    p_report.add_argument("--log-file", default=None)
    p_report.add_argument("--summary-output", default="/tmp/bisect_summary.md")
    p_report.set_defaults(func=cmd_report)

    # Used by bisect_vllm.sh to locate an editable/local vllm checkout.
    p_loc = subparsers.add_parser("vllm-location", help="Get vllm install location via pip show")
    p_loc.set_defaults(func=cmd_vllm_location)

    p_vllm_install = subparsers.add_parser("vllm-install", help="Get vllm install command for a test command")
    p_vllm_install.add_argument("--test-cmd", required=True)
    p_vllm_install.set_defaults(func=cmd_vllm_install)

    p_ascend_install = subparsers.add_parser(
        "ascend-install", help="Get vllm-ascend install command for a test command"
    )
    p_ascend_install.add_argument("--test-cmd", required=True)
    p_ascend_install.set_defaults(func=cmd_ascend_install)

    p_result = subparsers.add_parser("result-json", help="Build a machine-readable bisect result payload")
    p_result.add_argument("--caller-type", required=True)
    p_result.add_argument("--caller-run-id", required=True)
    p_result.add_argument("--bisect-run-id", required=True)
    p_result.add_argument("--good-commit", required=True)
    p_result.add_argument("--bad-commit", required=True)
    p_result.add_argument("--test-cmd", required=True)
    p_result.add_argument("--results-dir", required=True)
    p_result.add_argument("--summary-file", default="")
    p_result.add_argument("--output", default="")
    p_result.set_defaults(func=cmd_result_json)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
