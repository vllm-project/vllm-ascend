from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_bisect_test_cmd(summary: dict[str, Any]) -> str:
    failed_test_cases = summary.get("failed_test_cases") or []
    if failed_test_cases:
        return f"pytest -sv {failed_test_cases[0]}"

    failed_test_files = summary.get("failed_test_files") or []
    if failed_test_files:
        return f"pytest -sv {failed_test_files[0]}"

    raise ValueError("No failed tests available to build bisect command")


def build_bisect_request_id(*, run_id: int | str, round_index: int) -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"main2main-{run_id}-r{round_index}-{suffix}"


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    combine_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    if combine_output:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        if stdout_path is not None:
            stdout_path.write_text(completed.stdout, encoding="utf-8")
        if stderr_path is not None:
            stderr_path.write_text("", encoding="utf-8")
        return completed

    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else subprocess.PIPE
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else subprocess.PIPE
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        if stdout_path is not None:
            stdout_handle.close()
        if stderr_path is not None:
            stderr_handle.close()
    return completed


def _phase_artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "prompt": Path(f"{prefix}-prompt.txt"),
        "stdout": Path(f"{prefix}-result.json"),
        "stderr": Path(f"{prefix}-result.err"),
        "status": Path(f"{prefix}-status.txt"),
    }


def _suite_artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "log": Path(f"{prefix}.log"),
        "summary": prefix.parent / "main2main-failure-summary.json",
        "summary_stdout": Path(f"{prefix}-summary.out"),
        "summary_stderr": Path(f"{prefix}-summary.err"),
    }


def _bisect_artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "input": Path(f"{prefix}-input.json"),
        "dispatch_stdout": Path(f"{prefix}-dispatch.out"),
        "dispatch_stderr": Path(f"{prefix}-dispatch.err"),
        "run_json": Path(f"{prefix}-run.json"),
        "recent_runs": Path(f"{prefix}-runs.json"),
        "find_err": Path(f"{prefix}-find.err"),
        "complete_json": Path(f"{prefix}-complete.json"),
        "poll_err": Path(f"{prefix}-poll.err"),
        "download_stdout": Path(f"{prefix}-download.out"),
        "download_stderr": Path(f"{prefix}-download.err"),
        "output_dir": Path(f"{prefix}-output"),
    }


def _print_group(title: str, path: Path) -> None:
    print(f"::group::{title}")
    try:
        print(path.read_text(encoding="utf-8"), end="")
    except FileNotFoundError:
        pass
    print("::endgroup::")


def _print_group_if_nonempty(title: str, path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    _print_group(title, path)


def _validate_stream_json_result(path: Path) -> dict[str, Any]:
    final_result: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Claude stream-json event in {path}:{line_number}") from exc
            if event.get("type") == "result":
                final_result = event
    if final_result is None:
        raise ValueError(f"Claude stream-json output did not contain a final result event: {path}")
    return final_result


def _arg_or_env(value: str | None, env_name: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name, "")
    if env_value:
        return env_value
    raise ValueError(f"Missing required value: use CLI arg or set {env_name}")


def _path_arg_or_env(value: Path | None, env_name: str) -> Path:
    return Path(_arg_or_env(str(value) if value is not None else None, env_name))


def _arg_or_envs(value: str | None, *env_names: str) -> str:
    if value:
        return value
    for env_name in env_names:
        env_value = os.environ.get(env_name, "")
        if env_value:
            return env_value
    joined = ", ".join(env_names)
    raise ValueError(f"Missing required value: use CLI arg or set one of {joined}")


def _path_arg_or_envs(value: Path | None, *env_names: str) -> Path:
    raw_path = Path(_arg_or_envs(str(value) if value is not None else None, *env_names))
    if raw_path.is_absolute():
        return raw_path

    workspace = os.environ.get("GITHUB_WORKSPACE", "")
    if workspace:
        return Path(workspace) / raw_path
    return raw_path


def collect_commit_range(*, repo: Path, start_ref: str, end_ref: str) -> list[dict[str, str]]:
    log_output = _run_git(
        repo,
        "log",
        "--reverse",
        "--format=%H%x1f%s%x1f%b%x1e",
        f"{start_ref}..{end_ref}",
    )
    commits: list[dict[str, str]] = []
    for record in log_output.split("\x1e"):
        if not record.strip():
            continue
        record = record.rstrip("\n")
        parts = record.split("\x1f", 2)
        if len(parts) == 2:
            sha, subject = parts
            body = ""
        elif len(parts) == 3:
            sha, subject, body = parts
        else:
            raise ValueError(f"Unexpected git log record format: {record!r}")
        commits.append(
            {
                "sha": sha.strip(),
                "subject": subject.strip(),
                "body": body.strip(),
            }
        )
    return commits


def append_round_commits_markdown(
    *,
    output_path: Path,
    phase: str,
    round_index: int,
    commits: list[dict[str, str]],
) -> None:
    if not commits:
        return

    lines: list[str] = []
    if output_path.exists() and output_path.read_text(encoding="utf-8").strip():
        lines.append("")
        lines.append("")
    lines.append(f"### phase: {phase}")
    lines.append(f"round: {round_index}")
    lines.append("")
    lines.append("git_commits:")
    for commit in commits:
        lines.append(f"- sha: `{commit['sha']}`")
        lines.append(f"  subject: {commit['subject']}")
        lines.append("  body:")
        body = commit["body"] or "(empty)"
        for body_line in body.splitlines():
            lines.append(f"    {body_line}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_suite_and_summarize(
    *,
    work_repo_dir: Path,
    suite: str,
    artifact_prefix: Path,
) -> dict[str, Any]:
    paths = _suite_artifact_paths(artifact_prefix)
    log_path = paths["log"]
    summary_path = paths["summary"]
    summary_stdout_path = paths["summary_stdout"]
    summary_stderr_path = paths["summary_stderr"]
    # Keep the full suite log in one file so the workflow can print it directly.
    summary_path.unlink(missing_ok=True)
    summary_stdout_path.unlink(missing_ok=True)
    summary_stderr_path.unlink(missing_ok=True)
    completed = _run_command(
        [
            "python3",
            ".github/workflows/scripts/run_suite.py",
            "--suite",
            suite,
            "--continue-on-error",
        ],
        cwd=work_repo_dir,
        stdout_path=log_path,
        combine_output=True,
    )

    if completed.returncode != 0:
        summary_cmd = [
            "python3",
            ".github/workflows/scripts/ci_log_summary.py",
            "--log-file",
            str(log_path),
            "--format",
            "llm-json",
            "--output",
            str(summary_path),
        ]
        summary_result = _run_command(
            summary_cmd,
            cwd=work_repo_dir,
            stdout_path=summary_stdout_path,
            stderr_path=summary_stderr_path,
        )
        if summary_result.returncode != 0:
            stderr_text = ""
            if summary_stderr_path is not None and summary_stderr_path.exists():
                stderr_text = summary_stderr_path.read_text(encoding="utf-8").strip()
            raise RuntimeError(stderr_text or "ci_log_summary.py failed")

    return {
        "status": "success" if completed.returncode == 0 else "failure",
        "exit_code": completed.returncode,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "summary_stdout_path": str(summary_stdout_path),
        "summary_stderr_path": str(summary_stderr_path),
    }


def run_claude_phase(
    *,
    phase: str,
    round_index: int | None,
    work_repo_dir: Path,
    vllm_dir: str,
    old_commit: str,
    new_commit: str,
    session_id: str,
    model: str,
    allowed_tools: str,
    skill_path: Path,
    rounds_markdown_path: Path,
    artifact_prefix: Path,
    log_path: str | None = None,
    bisect_result_path: str | None = None,
) -> dict[str, Any]:
    paths = _phase_artifact_paths(artifact_prefix)
    prompt_path = paths["prompt"]
    stdout_path = paths["stdout"]
    stderr_path = paths["stderr"]
    status_path = paths["status"]
    # Each workflow phase still owns its control flow in YAML. This helper only
    # handles the repeated "render -> run Claude -> validate -> collect commits" sequence.
    if phase == "detect":
        prompt_text = render_detect_prompt(
            work_repo_dir=str(work_repo_dir),
            vllm_dir=vllm_dir,
            old_commit=old_commit,
            new_commit=new_commit,
        )
        effective_round = 0
    elif phase == "fix":
        if round_index is None or log_path is None:
            raise ValueError("phase=fix requires round_index and log_path")
        prompt_text = render_fix_prompt(
            work_repo_dir=str(work_repo_dir),
            vllm_dir=vllm_dir,
            old_commit=old_commit,
            new_commit=new_commit,
            round_index=round_index,
            log_path=log_path,
        )
        effective_round = round_index
    else:
        if round_index is None or log_path is None or bisect_result_path is None:
            raise ValueError("phase=bisect requires round_index, log_path, and bisect_result_path")
        prompt_text = render_bisect_fix_prompt(
            work_repo_dir=str(work_repo_dir),
            vllm_dir=vllm_dir,
            old_commit=old_commit,
            new_commit=new_commit,
            round_index=round_index,
            log_path=log_path,
            bisect_result_path=bisect_result_path,
        )
        effective_round = round_index

    prompt_path.write_text(prompt_text, encoding="utf-8")
    before_head = _run_git(work_repo_dir, "rev-parse", "HEAD").strip()
    session_args = ["--session-id", session_id] if phase == "detect" else ["--resume", session_id]
    completed = _run_command(
        [
            "claude",
            *session_args,
            "-p",
            prompt_text,
            "--model",
            model,
            "--append-system-prompt-file",
            str(skill_path),
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--allowedTools",
            allowed_tools,
        ],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)

    _validate_stream_json_result(stdout_path)

    status_output = _run_git(work_repo_dir, "status", "--porcelain").strip()
    if status_path is not None:
        status_path.write_text(status_output + ("\n" if status_output else ""), encoding="utf-8")
    if status_output:
        raise RuntimeError(f"Claude {phase} left uncommitted changes in {work_repo_dir}")

    after_head = _run_git(work_repo_dir, "rev-parse", "HEAD").strip()
    commits = collect_commit_range(repo=work_repo_dir, start_ref=before_head, end_ref=after_head)
    append_round_commits_markdown(
        output_path=rounds_markdown_path,
        phase=phase,
        round_index=effective_round,
        commits=commits,
    )

    return {
        "phase": phase,
        "round": effective_round,
        "commit_count": len(commits),
        "before_head": before_head,
        "after_head": after_head,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "status_path": str(status_path),
    }


def render_detect_prompt(*, work_repo_dir: str, vllm_dir: str, old_commit: str, new_commit: str) -> str:
    return (
        "\n".join(
            [
                "Use Main2Main skill to adapt vllm-ascend to the latest vLLM main branch.",
                "",
                "Context:",
                f"- Benchmark repo is checked out at ./{work_repo_dir}",
                f"- Upstream vLLM source is checked out at ./{vllm_dir}",
                f"- OLD_COMMIT={old_commit}",
                f"- NEW_COMMIT={new_commit}",
                "",
                "Requirements:",
                "- Analyze upstream diff and apply adaptation fixes",
                "- Update commit references if needed",
                "- Create a git commit if and only if you make valid code changes",
                '- When committing, always use: git commit -s -m "<message>"',
                "- Do not push",
                "- Do not create a PR",
            ]
        )
        + "\n"
    )


def render_fix_prompt(
    *,
    work_repo_dir: str,
    vllm_dir: str,
    old_commit: str,
    new_commit: str,
    round_index: int,
    log_path: str,
) -> str:
    return (
        "\n".join(
            [
                "Use Main2Main skill to fix the current main2main test failures.",
                f"test failures log is available at {log_path}",
                "",
                "Context:",
                f"- Benchmark repo is checked out at ./{work_repo_dir}",
                f"- Upstream vLLM source is checked out at ./{vllm_dir}",
                f"- OLD_COMMIT={old_commit}",
                f"- NEW_COMMIT={new_commit}",
                f"- Current round={round_index}",
                f"- Main2Main test log path={log_path}",
                "",
                "Requirements:",
                f"- Use {log_path} as the primary failure-analysis input",
                "- Fix code bugs in the benchmark repo",
                "- Modify code only",
                "- Create a git commit if and only if you make valid code changes",
                '- When committing, always use: git commit -s -m "<message>"',
                "- Do not push",
                "- Do not create a PR",
            ]
        )
        + "\n"
    )


def render_bisect_fix_prompt(
    *,
    work_repo_dir: str,
    vllm_dir: str,
    old_commit: str,
    new_commit: str,
    round_index: int,
    log_path: str,
    bisect_result_path: str,
) -> str:
    return (
        "\n".join(
            [
                "Use the main2main skill to fix the remaining main2main failures based on bisect results.",
                "The CI still failed after fix, so test failure log and bisect results are also provided.",
                "",
                "Context:",
                f"- Benchmark repo is checked out at ./{work_repo_dir}",
                f"- Upstream vLLM source is checked out at ./{vllm_dir}",
                f"- Main2Main test log path={log_path}",
                f"- Bisect result path={bisect_result_path}",
                f"- Round={round_index}",
                f"- OLD_COMMIT={old_commit}",
                f"- NEW_COMMIT={new_commit}",
                "",
                "Requirements:",
                "- Use the bisect result to produce a targeted fix",
                f"- Use {log_path} as the primary failure-analysis input",
                "- Modify code only",
                "- Create a git commit if and only if you make valid code changes",
                '- When committing, always use: git commit -s -m "<message>"',
                "- Do not push",
                "- Do not create a PR",
            ]
        )
        + "\n"
    )


def should_create_pr(commits: list[dict[str, str]]) -> bool:
    return bool(commits)


def render_pr_body(
    *,
    old_commit: str,
    new_commit: str,
    rounds_markdown: str,
) -> str:
    lines = [
        "Automated adaptation to upstream vLLM main branch changes.",
        f"Commit range: {old_commit}...{new_commit}",
        "",
    ]
    rounds_markdown = rounds_markdown.strip()
    if rounds_markdown:
        lines.append(rounds_markdown)
    return "\n".join(lines).rstrip() + "\n"


def render_manual_review_issue(
    *,
    pr_url: str,
    old_commit: str,
    new_commit: str,
    summary: dict[str, Any],
    bisect_summary: dict[str, Any] | None = None,
) -> str:
    code_bugs = summary.get("code_bugs") or []
    failed_test_files = summary.get("failed_test_files") or []
    failed_test_cases = summary.get("failed_test_cases") or []

    lines = [
        "## Summary",
        "",
        "main2main automation exhausted its fix and bisect budget.",
        "",
        "## Context",
        "",
        f"- Draft PR: {pr_url}",
        f"- Commit range: `{old_commit}`...`{new_commit}`",
        "",
        "## Remaining Failures",
        "",
        f"- Failed test files: `{len(failed_test_files)}`",
        f"- Failed test cases: `{len(failed_test_cases)}`",
        f"- Code bugs: `{len(code_bugs)}`",
    ]

    if code_bugs:
        lines.extend(["", "### Code Bugs", ""])
        for bug in code_bugs:
            error_type = bug.get("error_type", "UnknownError")
            error_message = bug.get("error_message", "")
            lines.append(f"- `{error_type}`: {error_message}")

    if bisect_summary:
        lines.extend(
            [
                "",
                "## Bisect Summary",
                "",
                f"- Status: `{bisect_summary.get('status', 'unknown')}`",
            ]
        )
        if bisect_summary.get("first_bad_commit"):
            lines.append(f"- First bad commit: `{bisect_summary['first_bad_commit']}`")
        if bisect_summary.get("first_bad_commit_url"):
            lines.append(f"- Commit URL: {bisect_summary['first_bad_commit_url']}")

    return "\n".join(lines).rstrip() + "\n"


def _run_gh_json(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _list_runs_json(*, repo: str, limit: int, fields: str, workflow_name: str | None = None) -> list[dict[str, Any]]:
    args = ["run", "list", "--repo", repo]
    if workflow_name:
        args.extend(["--workflow", workflow_name])
    args.extend(["--limit", str(limit), "--json", fields])
    return _run_gh_json(args)


def _run_title(run: dict[str, Any]) -> str:
    return (run.get("displayTitle") or run.get("name") or "").strip()


def find_bisect_run(
    *, repo: str, request_id: str, workflow_name: str = "dispatch_main2main_bisect.yaml"
) -> dict[str, Any]:
    fields = "databaseId,name,status,conclusion,headBranch,event,url,workflowDatabaseId"
    filtered_error = ""
    filtered_runs: list[dict[str, Any]] = []
    try:
        filtered_runs = _list_runs_json(repo=repo, workflow_name=workflow_name, limit=100, fields=fields)
    except subprocess.CalledProcessError as exc:
        filtered_error = (exc.stderr or "").strip()
    for run in filtered_runs:
        if request_id in _run_title(run):
            return run

    try:
        fallback_runs = _list_runs_json(repo=repo, workflow_name=None, limit=100, fields=fields)
    except subprocess.CalledProcessError as fallback_exc:
        fallback_error = (fallback_exc.stderr or "").strip()
        if filtered_error:
            raise ValueError(
                "gh run list failed with workflow filter"
                f" ({filtered_error or 'unknown error'}); fallback without workflow filter failed"
                f" ({fallback_error or fallback_exc})"
            ) from fallback_exc
        raise

    for run in fallback_runs:
        if request_id in _run_title(run):
            return run

    if filtered_error:
        raise ValueError(
            f"No bisect run found for request_id={request_id}; "
            f"workflow-filtered gh run list failed first: {filtered_error}"
        )
    raise ValueError(f"No bisect run found for request_id={request_id}")


def list_bisect_runs(
    *,
    repo: str,
    workflow_name: str = "dispatch_main2main_bisect.yaml",
    limit: int = 30,
) -> list[dict[str, Any]]:
    fields = "databaseId,name,status,conclusion,headBranch,event,url,createdAt,workflowDatabaseId"
    try:
        return _list_runs_json(repo=repo, workflow_name=workflow_name, limit=limit, fields=fields)
    except subprocess.CalledProcessError:
        return _list_runs_json(repo=repo, workflow_name=None, limit=limit, fields=fields)


def poll_bisect_run(
    *,
    repo: str,
    run_id: int,
    timeout_seconds: int = 1800,
    poll_interval_seconds: int = 15,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = _run_gh_json(
            [
                "run",
                "view",
                str(run_id),
                "--repo",
                repo,
                "--json",
                "databaseId,status,conclusion,name,url",
            ]
        )
        if run.get("status") == "completed":
            return run
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for bisect run {run_id}")


def run_bisect_round(
    *,
    work_repo_dir: Path,
    github_repo: str,
    github_run_id: str,
    round_index: int,
    old_commit: str,
    new_commit: str,
    log_path: Path,
    failure_summary_path: Path,
    artifact_prefix: Path,
) -> dict[str, Any]:
    paths = _bisect_artifact_paths(artifact_prefix)
    bisect_input_path = paths["input"]
    dispatch_stdout_path = paths["dispatch_stdout"]
    dispatch_stderr_path = paths["dispatch_stderr"]
    run_json_path = paths["run_json"]
    recent_runs_path = paths["recent_runs"]
    find_err_path = paths["find_err"]
    complete_json_path = paths["complete_json"]
    poll_err_path = paths["poll_err"]
    output_dir = paths["output_dir"]
    download_stdout_path = paths["download_stdout"]
    download_stderr_path = paths["download_stderr"]
    # Keep one complete bisect round in a single command so the workflow can
    # stay readable while still printing every intermediate artifact on failure.
    if not failure_summary_path.exists():
        raise FileNotFoundError(f"Expected failure summary at {failure_summary_path}")

    bisect_input_result = _run_command(
        [
            "python3",
            str(work_repo_dir / ".github/workflows/scripts/ci_log_summary.py"),
            "--log-file",
            str(log_path),
            "--format",
            "bisect-json",
            "--output",
            str(bisect_input_path),
        ],
    )
    if bisect_input_result.returncode != 0:
        raise subprocess.CalledProcessError(bisect_input_result.returncode, bisect_input_result.args)

    bisect_input = json.loads(bisect_input_path.read_text(encoding="utf-8"))
    test_cmd = bisect_input.get("test_cmd", "")
    if not test_cmd:
        raise ValueError(f"No bisect test command could be generated from {log_path}")

    request_id = build_bisect_request_id(run_id=github_run_id, round_index=round_index)
    dispatch_result = _run_command(
        [
            "gh",
            "workflow",
            "run",
            "dispatch_main2main_bisect.yaml",
            "--repo",
            github_repo,
            "-f",
            f"good_commit={old_commit}",
            "-f",
            f"bad_commit={new_commit}",
            "-f",
            f"test_cmd={test_cmd}",
            "-f",
            f"request_id={request_id}",
        ],
        stdout_path=dispatch_stdout_path,
        stderr_path=dispatch_stderr_path,
    )
    if dispatch_result.returncode != 0:
        raise subprocess.CalledProcessError(dispatch_result.returncode, dispatch_result.args)

    run_json_path.write_text("", encoding="utf-8")
    recent_runs_path.write_text("", encoding="utf-8")
    find_err_path.write_text("", encoding="utf-8")
    for _ in range(20):
        find_result = _run_command(
            [
                "python3",
                str(work_repo_dir / ".github/workflows/scripts/main2main_ci.py"),
                "find-bisect-run",
                "--repo",
                github_repo,
                "--request-id",
                request_id,
            ],
            stdout_path=run_json_path,
            stderr_path=find_err_path,
        )
        if find_result.returncode == 0 and run_json_path.read_text(encoding="utf-8").strip():
            break
        time.sleep(15)

    if not run_json_path.read_text(encoding="utf-8").strip():
        list_result = _run_command(
            [
                "python3",
                str(work_repo_dir / ".github/workflows/scripts/main2main_ci.py"),
                "list-bisect-runs",
                "--repo",
                github_repo,
                "--limit",
                "30",
            ],
            stdout_path=recent_runs_path,
        )
        if list_result.returncode != 0:
            recent_runs_path.write_text("", encoding="utf-8")
        raise ValueError(f"Unable to locate bisect run for request {request_id}")

    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    bisect_run_id = int(run_data["databaseId"])
    poll_timeout_minutes = int(os.environ.get("BISECT_POLL_TIMEOUT_MINUTES", "180"))
    poll_timeout_seconds = max(1, poll_timeout_minutes) * 60

    poll_result = _run_command(
        [
            "python3",
            str(work_repo_dir / ".github/workflows/scripts/main2main_ci.py"),
            "poll-bisect-run",
            "--repo",
            github_repo,
            "--run-id",
            str(bisect_run_id),
            "--timeout-seconds",
            str(poll_timeout_seconds),
        ],
        stdout_path=complete_json_path,
        stderr_path=poll_err_path,
    )
    if poll_result.returncode != 0:
        raise subprocess.CalledProcessError(poll_result.returncode, poll_result.args)

    if output_dir.exists():
        # Recreate the artifact directory so each round exposes only its own files.
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_result = _run_command(
        [
            "gh",
            "run",
            "download",
            str(bisect_run_id),
            "--repo",
            github_repo,
            "--name",
            f"bisect-summary-{request_id}",
            "--dir",
            str(output_dir),
        ],
        stdout_path=download_stdout_path,
        stderr_path=download_stderr_path,
    )
    if download_result.returncode != 0:
        raise subprocess.CalledProcessError(download_result.returncode, download_result.args)

    bisect_result_path = output_dir / "bisect_result.json"
    if not bisect_result_path.exists():
        raise FileNotFoundError(f"Missing bisect_result.json in {output_dir}")

    complete_data = json.loads(complete_json_path.read_text(encoding="utf-8"))
    return {
        "request_id": request_id,
        "test_cmd": test_cmd,
        "bisect_run_id": bisect_run_id,
        "run_url": run_data.get("url", ""),
        "complete_url": complete_data.get("url", ""),
        "dispatch_stdout_path": str(dispatch_stdout_path) if dispatch_stdout_path is not None else "",
        "dispatch_stderr_path": str(dispatch_stderr_path) if dispatch_stderr_path is not None else "",
        "run_json_path": str(run_json_path),
        "recent_runs_path": str(recent_runs_path),
        "find_err_path": str(find_err_path),
        "complete_json_path": str(complete_json_path),
        "poll_err_path": str(poll_err_path),
        "output_dir": str(output_dir),
        "bisect_result_path": str(bisect_result_path),
        "bisect_input_path": str(bisect_input_path),
        "download_stdout_path": str(download_stdout_path) if download_stdout_path is not None else "",
        "download_stderr_path": str(download_stderr_path) if download_stderr_path is not None else "",
    }


def print_bisect_round_logs(*, artifact_prefix: Path) -> None:
    paths = _bisect_artifact_paths(artifact_prefix)
    meta_path = Path(f"{artifact_prefix}-meta.json")

    if meta_path.exists():
        print(meta_path.read_text(encoding="utf-8"), end="")
    if paths["input"].exists():
        print(paths["input"].read_text(encoding="utf-8"), end="")

    round_label = artifact_prefix.name.replace("main2main-", "").replace("-", " ")
    _print_group_if_nonempty(f"{round_label} dispatch stdout", paths["dispatch_stdout"])
    _print_group_if_nonempty(f"{round_label} dispatch stderr", paths["dispatch_stderr"])

    print(f"::group::{round_label} wait for run")
    if paths["run_json"].exists():
        run_text = paths["run_json"].read_text(encoding="utf-8")
        print(run_text, end="")
        if run_text.strip():
            try:
                run_data = json.loads(run_text)
            except json.JSONDecodeError:
                run_data = {}
            run_url = run_data.get("url", "")
            if run_url:
                print(f"Found bisect run: {run_url}")
    if paths["complete_json"].exists():
        complete_text = paths["complete_json"].read_text(encoding="utf-8")
        print(complete_text, end="")
        if complete_text.strip():
            try:
                complete_data = json.loads(complete_text)
            except json.JSONDecodeError:
                complete_data = {}
            complete_url = complete_data.get("url", "")
            if complete_url:
                print(f"Completed bisect run: {complete_url}")
    _print_group_if_nonempty(f"{round_label} poll stderr", paths["poll_err"])
    _print_group_if_nonempty(f"{round_label} find stderr", paths["find_err"])
    _print_group_if_nonempty(f"{round_label} recent runs", paths["recent_runs"])
    print("::endgroup::")

    print(f"::group::{round_label} download result")
    _print_group_if_nonempty(f"{round_label} download stdout", paths["download_stdout"])
    _print_group_if_nonempty(f"{round_label} download stderr", paths["download_stderr"])
    bisect_result_path = paths["output_dir"] / "bisect_result.json"
    if bisect_result_path.exists():
        print(bisect_result_path.read_text(encoding="utf-8"), end="")
    print("::endgroup::")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper CLI for simplified main2main workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("extract-bisect-test-cmd")
    summary_parser.add_argument("--summary", type=Path, required=True)

    request_id_parser = subparsers.add_parser("build-request-id")
    request_id_parser.add_argument("--run-id", required=True)
    request_id_parser.add_argument("--round-index", type=int, required=True)

    find_run_parser = subparsers.add_parser("find-bisect-run")
    find_run_parser.add_argument("--repo", required=True)
    find_run_parser.add_argument("--request-id", required=True)
    find_run_parser.add_argument("--workflow-name", default="dispatch_main2main_bisect.yaml")

    list_runs_parser = subparsers.add_parser("list-bisect-runs")
    list_runs_parser.add_argument("--repo", required=True)
    list_runs_parser.add_argument("--workflow-name", default="dispatch_main2main_bisect.yaml")
    list_runs_parser.add_argument("--limit", type=int, default=30)

    poll_parser = subparsers.add_parser("poll-bisect-run")
    poll_parser.add_argument("--repo", required=True)
    poll_parser.add_argument("--run-id", type=int, required=True)
    poll_parser.add_argument("--timeout-seconds", type=int, default=1800)
    poll_parser.add_argument("--poll-interval-seconds", type=int, default=15)

    commits_parser = subparsers.add_parser("collect-commit-range")
    commits_parser.add_argument("--repo", type=Path, required=True)
    commits_parser.add_argument("--start-ref", required=True)
    commits_parser.add_argument("--end-ref", required=True)

    append_round_parser = subparsers.add_parser("append-round-commits-markdown")
    append_round_parser.add_argument("--repo", type=Path, required=True)
    append_round_parser.add_argument("--output", type=Path, required=True)
    append_round_parser.add_argument("--phase", required=True, choices=["detect", "fix", "bisect"])
    append_round_parser.add_argument("--round", type=int, required=True)
    append_round_parser.add_argument("--start-ref", required=True)
    append_round_parser.add_argument("--end-ref", required=True)

    prompt_parser = subparsers.add_parser("render-prompt")
    prompt_parser.add_argument("--phase", required=True, choices=["detect", "fix", "bisect"])
    prompt_parser.add_argument("--output", type=Path, required=True)
    prompt_parser.add_argument("--work-repo-dir", required=True)
    prompt_parser.add_argument("--vllm-dir", required=True)
    prompt_parser.add_argument("--old-commit", required=True)
    prompt_parser.add_argument("--new-commit", required=True)
    prompt_parser.add_argument("--round", type=int)
    prompt_parser.add_argument("--log-path")
    prompt_parser.add_argument("--bisect-result-path")

    claude_phase_parser = subparsers.add_parser("run-claude-phase")
    claude_phase_parser.add_argument("--phase", required=True, choices=["detect", "fix", "bisect"])
    claude_phase_parser.add_argument("--work-repo-dir", type=Path)
    claude_phase_parser.add_argument("--vllm-dir")
    claude_phase_parser.add_argument("--old-commit")
    claude_phase_parser.add_argument("--new-commit")
    claude_phase_parser.add_argument("--session-id")
    claude_phase_parser.add_argument("--model")
    claude_phase_parser.add_argument("--allowed-tools")
    claude_phase_parser.add_argument("--skill-path", type=Path)
    claude_phase_parser.add_argument("--rounds-markdown-path", type=Path)
    claude_phase_parser.add_argument("--artifact-prefix", type=Path, required=True)
    claude_phase_parser.add_argument("--round", type=int)
    claude_phase_parser.add_argument("--log-path")
    claude_phase_parser.add_argument("--bisect-result-path")

    suite_parser = subparsers.add_parser("run-suite-and-summarize")
    suite_parser.add_argument("--work-repo-dir", type=Path)
    suite_parser.add_argument("--suite")
    suite_parser.add_argument("--artifact-prefix", type=Path, required=True)

    bisect_round_parser = subparsers.add_parser("run-bisect-round")
    bisect_round_parser.add_argument("--work-repo-dir", type=Path)
    bisect_round_parser.add_argument("--github-repo")
    bisect_round_parser.add_argument("--github-run-id")
    bisect_round_parser.add_argument("--round", type=int, required=True)
    bisect_round_parser.add_argument("--old-commit")
    bisect_round_parser.add_argument("--new-commit")
    bisect_round_parser.add_argument("--log-path", type=Path, required=True)
    bisect_round_parser.add_argument("--failure-summary-path", type=Path, required=True)
    bisect_round_parser.add_argument("--artifact-prefix", type=Path, required=True)

    print_bisect_logs_parser = subparsers.add_parser("print-bisect-round-logs")
    print_bisect_logs_parser.add_argument("--artifact-prefix", type=Path, required=True)

    pr_parser = subparsers.add_parser("render-pr-body")
    pr_parser.add_argument("--old-commit", required=True)
    pr_parser.add_argument("--new-commit", required=True)
    pr_parser.add_argument("--rounds-md", type=Path, required=True)

    issue_parser = subparsers.add_parser("render-manual-review-issue")
    issue_parser.add_argument("--pr-url", required=True)
    issue_parser.add_argument("--old-commit", required=True)
    issue_parser.add_argument("--new-commit", required=True)
    issue_parser.add_argument("--summary-json", type=Path, required=True)
    issue_parser.add_argument("--bisect-json", type=Path)

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "extract-bisect-test-cmd":
        print(extract_bisect_test_cmd(load_summary(args.summary)))
        return

    if args.command == "build-request-id":
        print(build_bisect_request_id(run_id=args.run_id, round_index=args.round_index))
        return

    if args.command == "find-bisect-run":
        print(json.dumps(find_bisect_run(repo=args.repo, request_id=args.request_id, workflow_name=args.workflow_name)))
        return

    if args.command == "list-bisect-runs":
        print(
            json.dumps(
                list_bisect_runs(repo=args.repo, workflow_name=args.workflow_name, limit=args.limit),
                ensure_ascii=False,
            )
        )
        return

    if args.command == "poll-bisect-run":
        print(
            json.dumps(
                poll_bisect_run(
                    repo=args.repo,
                    run_id=args.run_id,
                    timeout_seconds=args.timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                )
            )
        )
        return

    if args.command == "collect-commit-range":
        print(
            json.dumps(
                collect_commit_range(repo=args.repo, start_ref=args.start_ref, end_ref=args.end_ref),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "append-round-commits-markdown":
        commits = collect_commit_range(repo=args.repo, start_ref=args.start_ref, end_ref=args.end_ref)
        append_round_commits_markdown(
            output_path=args.output,
            phase=args.phase,
            round_index=args.round,
            commits=commits,
        )
        print(json.dumps({"count": len(commits)}, ensure_ascii=False))
        return

    if args.command == "render-prompt":
        if args.phase == "detect":
            text = render_detect_prompt(
                work_repo_dir=args.work_repo_dir,
                vllm_dir=args.vllm_dir,
                old_commit=args.old_commit,
                new_commit=args.new_commit,
            )
        elif args.phase == "fix":
            if args.round is None:
                raise ValueError("--round is required for phase=fix")
            if not args.log_path:
                raise ValueError("--log-path is required for phase=fix")
            text = render_fix_prompt(
                work_repo_dir=args.work_repo_dir,
                vllm_dir=args.vllm_dir,
                old_commit=args.old_commit,
                new_commit=args.new_commit,
                round_index=args.round,
                log_path=args.log_path,
            )
        else:
            if args.round is None:
                raise ValueError("--round is required for phase=bisect")
            if not args.log_path:
                raise ValueError("--log-path is required for phase=bisect")
            if not args.bisect_result_path:
                raise ValueError("--bisect-result-path is required for phase=bisect")
            text = render_bisect_fix_prompt(
                work_repo_dir=args.work_repo_dir,
                vllm_dir=args.vllm_dir,
                old_commit=args.old_commit,
                new_commit=args.new_commit,
                round_index=args.round,
                log_path=args.log_path,
                bisect_result_path=args.bisect_result_path,
            )
        args.output.write_text(text, encoding="utf-8")
        return

    if args.command == "run-claude-phase":
        print(
            json.dumps(
                run_claude_phase(
                    phase=args.phase,
                    round_index=args.round,
                    work_repo_dir=_path_arg_or_env(args.work_repo_dir, "WORK_REPO_DIR"),
                    vllm_dir=_arg_or_env(args.vllm_dir, "VLLM_DIR"),
                    old_commit=_arg_or_env(args.old_commit, "MAIN2MAIN_OLD_COMMIT"),
                    new_commit=_arg_or_env(args.new_commit, "MAIN2MAIN_NEW_COMMIT"),
                    session_id=_arg_or_env(args.session_id, "MAIN2MAIN_SESSION_ID"),
                    model=_arg_or_env(args.model, "MAIN2MAIN_MODEL"),
                    allowed_tools=_arg_or_env(args.allowed_tools, "CLAUDE_ALLOWED_TOOLS"),
                    skill_path=_path_arg_or_env(args.skill_path, "MAIN2MAIN_SKILL_PATH"),
                    rounds_markdown_path=_path_arg_or_env(args.rounds_markdown_path, "MAIN2MAIN_ROUNDS_MARKDOWN_PATH"),
                    artifact_prefix=args.artifact_prefix,
                    log_path=args.log_path,
                    bisect_result_path=args.bisect_result_path,
                ),
                ensure_ascii=False,
            )
        )
        return

    if args.command == "run-suite-and-summarize":
        print(
            json.dumps(
                run_suite_and_summarize(
                    work_repo_dir=_path_arg_or_env(args.work_repo_dir, "WORK_REPO_DIR"),
                    suite=_arg_or_env(args.suite, "MAIN2MAIN_SUITE"),
                    artifact_prefix=args.artifact_prefix,
                ),
                ensure_ascii=False,
            )
        )
        return

    if args.command == "run-bisect-round":
        print(
            json.dumps(
                run_bisect_round(
                    work_repo_dir=_path_arg_or_env(args.work_repo_dir, "WORK_REPO_DIR"),
                    github_repo=_arg_or_env(args.github_repo, "GITHUB_REPOSITORY"),
                    github_run_id=_arg_or_env(args.github_run_id, "GITHUB_RUN_ID"),
                    round_index=args.round,
                    old_commit=_arg_or_env(args.old_commit, "MAIN2MAIN_OLD_COMMIT"),
                    new_commit=_arg_or_env(args.new_commit, "MAIN2MAIN_NEW_COMMIT"),
                    log_path=args.log_path,
                    failure_summary_path=args.failure_summary_path,
                    artifact_prefix=args.artifact_prefix,
                ),
                ensure_ascii=False,
            )
        )
        return

    if args.command == "print-bisect-round-logs":
        print_bisect_round_logs(artifact_prefix=args.artifact_prefix)
        return

    if args.command == "render-pr-body":
        rounds_markdown = args.rounds_md.read_text(encoding="utf-8") if args.rounds_md.exists() else ""
        print(
            render_pr_body(
                old_commit=args.old_commit,
                new_commit=args.new_commit,
                rounds_markdown=rounds_markdown,
            ),
            end="",
        )
        return

    if args.command == "render-manual-review-issue":
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        bisect_summary = None
        if args.bisect_json is not None:
            bisect_summary = json.loads(args.bisect_json.read_text(encoding="utf-8"))
        print(
            render_manual_review_issue(
                pr_url=args.pr_url,
                old_commit=args.old_commit,
                new_commit=args.new_commit,
                summary=summary,
                bisect_summary=bisect_summary,
            ),
            end="",
        )
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
