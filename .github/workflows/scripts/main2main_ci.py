from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

STATE_MARKER = "main2main-state:v1"
REGISTER_MARKER = "main2main-register"

_COMMIT_RANGE_RE = re.compile(
    r"^\*\*Commit range:\*\* `([0-9a-f]{40})`\.\.\.`([0-9a-f]{40})`$",
    re.MULTILINE,
)
_REGISTRATION_COMMENT_RE = re.compile(
    r"<!-- main2main-register\s*"
    r"pr_number=(\d+)\s*"
    r"branch=([^\n]+)\s*"
    r"head_sha=([0-9a-f]{40})\s*"
    r"old_commit=([0-9a-f]{40})\s*"
    r"new_commit=([0-9a-f]{40})\s*"
    r"phase=(2|3|done)\s*"
    r"-->",
    re.MULTILINE,
)
_STATE_COMMENT_RE = re.compile(r"<!-- main2main-state:v1\s*(\{.*?\})\s*-->", re.DOTALL)
_RUN_ID_FROM_DETAILS_URL_RE = re.compile(r"/actions/runs/(\d+)(?:/job/\d+)?")
_PHASE_SECTION_RE_TEMPLATE = r"(?:\n)?### {heading}\n(?:.*?)(?=\n### |\Z)"
DEFAULT_BISECT_TEST_CMD = (
    "pytest -sv tests/e2e/singlecard/test_aclgraph_accuracy.py; "
    "pytest -sv tests/e2e/multicard/4-cards/long_sequence/test_accuracy.py"
)

_FAILURE_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "stale",
    "startup_failure",
    "timed_out",
}
_CONFLICT_MERGEABLES = {"CONFLICTING"}
_CONFLICT_MERGE_STATE_STATUSES = {"DIRTY", "CONFLICTING"}


@dataclass(frozen=True)
class PrMetadata:
    old_commit: str
    new_commit: str


@dataclass(frozen=True)
class RegistrationMetadata:
    pr_number: int
    branch: str
    head_sha: str
    old_commit: str
    new_commit: str
    phase: str


@dataclass(frozen=True)
class Main2MainState:
    pr_number: int
    branch: str
    head_sha: str
    old_commit: str
    new_commit: str
    phase: str
    status: str
    dispatch_token: str = ""
    e2e_run_id: str = ""
    fix_run_id: str = ""
    bisect_run_id: str = ""
    terminal_reason: str = ""
    workflow_error_count: int = 0
    last_transition: str = ""
    updated_at: str = ""
    updated_by: str = ""


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ReconcileDecision:
    action: str
    reason: str = ""
    terminal_reason: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class FixupOutcome:
    result: str
    phase: str


@dataclass(frozen=True)
class MarkerComment:
    id: int
    body: str


@dataclass(frozen=True)
class PhaseContext:
    pr: dict[str, Any]
    state: Main2MainState
    registration: RegistrationMetadata
    state_comment_id: int
    register_comment_id: int
    branch: str
    head_sha: str


def parse_pr_metadata(body: str) -> PrMetadata:
    commit_match = _COMMIT_RANGE_RE.search(body)
    if commit_match is None:
        raise ValueError("PR body is missing main2main metadata")
    return PrMetadata(
        old_commit=commit_match.group(1),
        new_commit=commit_match.group(2),
    )


def parse_registration_comment(body: str) -> RegistrationMetadata:
    match = _REGISTRATION_COMMENT_RE.search(body)
    if match is None:
        raise ValueError("registration comment is missing main2main metadata")
    return RegistrationMetadata(
        pr_number=int(match.group(1)),
        branch=match.group(2),
        head_sha=match.group(3),
        old_commit=match.group(4),
        new_commit=match.group(5),
        phase=match.group(6),
    )


def render_registration_comment(metadata: RegistrationMetadata) -> str:
    return (
        "<!-- main2main-register\n"
        f"pr_number={metadata.pr_number}\n"
        f"branch={metadata.branch}\n"
        f"head_sha={metadata.head_sha}\n"
        f"old_commit={metadata.old_commit}\n"
        f"new_commit={metadata.new_commit}\n"
        f"phase={metadata.phase}\n"
        "-->"
    )


def _normalize_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "pr_number": int(payload["pr_number"]),
        "branch": payload["branch"],
        "head_sha": payload["head_sha"],
        "old_commit": payload["old_commit"],
        "new_commit": payload["new_commit"],
        "phase": payload["phase"],
        "status": payload["status"],
        "dispatch_token": str(payload.get("dispatch_token", "")),
        "e2e_run_id": str(payload.get("e2e_run_id", "")),
        "fix_run_id": str(payload.get("fix_run_id", "")),
        "bisect_run_id": str(payload.get("bisect_run_id", "")),
        "terminal_reason": str(payload.get("terminal_reason", "")),
        "workflow_error_count": int(payload.get("workflow_error_count", 0)),
        "last_transition": str(payload.get("last_transition", "")),
        "updated_at": str(payload.get("updated_at", "")),
        "updated_by": str(payload.get("updated_by", "")),
    }
    return normalized


def parse_state_comment(body: str) -> Main2MainState:
    match = _STATE_COMMENT_RE.search(body)
    if match is None:
        raise ValueError("state comment is missing main2main metadata")
    payload = json.loads(match.group(1))
    return Main2MainState(**_normalize_state_payload(payload))


def render_state_comment(state: Main2MainState) -> str:
    payload = json.dumps(asdict(state), ensure_ascii=True, indent=2, sort_keys=True)
    return f"<!-- {STATE_MARKER}\n{payload}\n-->"


def init_state_from_registration(
    metadata: RegistrationMetadata,
    *,
    dispatch_token: str,
    updated_at: str = "",
    updated_by: str = "",
) -> Main2MainState:
    return Main2MainState(
        pr_number=metadata.pr_number,
        branch=metadata.branch,
        head_sha=metadata.head_sha,
        old_commit=metadata.old_commit,
        new_commit=metadata.new_commit,
        phase=metadata.phase,
        status="waiting_e2e",
        dispatch_token=dispatch_token,
        last_transition="register->waiting_e2e",
        updated_at=updated_at,
        updated_by=updated_by,
    )


def mint_dispatch_token() -> str:
    return uuid.uuid4().hex


def upsert_pr_phase_section(body: str, *, heading: str, content_lines: list[str]) -> str:
    section = "\n".join([f"### {heading}", *content_lines]).rstrip()
    pattern = re.compile(_PHASE_SECTION_RE_TEMPLATE.format(heading=re.escape(heading)), re.DOTALL)
    existing_body = body.rstrip()
    if pattern.search(existing_body):
        updated = pattern.sub("\n" + section, existing_body, count=1)
        return updated.strip() + "\n"
    joiner = "\n\n" if existing_body else ""
    return f"{existing_body}{joiner}{section}\n"


def check_state_guard(
    state: Main2MainState,
    *,
    expected_phase: str | None = None,
    expected_status: str | None = None,
    dispatch_token: str | None = None,
) -> GuardResult:
    if expected_phase and state.phase != expected_phase:
        return GuardResult(False, f"phase mismatch: expected {expected_phase}, got {state.phase}")
    if expected_status and state.status != expected_status:
        return GuardResult(False, f"status mismatch: expected {expected_status}, got {state.status}")
    if dispatch_token is not None and state.dispatch_token != dispatch_token:
        return GuardResult(False, "dispatch token mismatch")
    return GuardResult(True)


def check_pr_consistency(
    state: Main2MainState,
    *,
    branch: str,
    head_sha: str,
) -> GuardResult:
    if state.branch != branch:
        return GuardResult(False, f"branch mismatch: expected {state.branch}, got {branch}")
    if state.head_sha != head_sha:
        return GuardResult(False, f"head_sha mismatch: expected {state.head_sha}, got {head_sha}")
    return GuardResult(True)


def check_registration_consistency(
    state: Main2MainState,
    registration: RegistrationMetadata,
) -> GuardResult:
    if state.branch != registration.branch:
        return GuardResult(False, f"registration branch mismatch: expected {state.branch}, got {registration.branch}")
    if state.head_sha != registration.head_sha:
        return GuardResult(False, f"registration head_sha mismatch: expected {state.head_sha}, got {registration.head_sha}")
    if state.old_commit != registration.old_commit or state.new_commit != registration.new_commit:
        return GuardResult(False, "registration commit range mismatch")
    if state.phase != registration.phase:
        return GuardResult(False, f"registration phase mismatch: expected {state.phase}, got {registration.phase}")
    return GuardResult(True)


def _sort_run_key(run: dict[str, Any]) -> tuple[str, int]:
    return (str(run.get("createdAt") or run.get("updatedAt") or ""), int(run.get("databaseId") or 0))


def select_matching_e2e_run(runs: list[dict[str, Any]], *, head_sha: str) -> dict[str, Any] | None:
    matching = [run for run in runs if run.get("headSha") == head_sha]
    if not matching:
        return None
    return max(matching, key=_sort_run_key)


def extract_run_id_from_details_url(url: str) -> str:
    match = _RUN_ID_FROM_DETAILS_URL_RE.search(url or "")
    if match is None:
        return ""
    return match.group(1)


def resolve_e2e_run_id_from_status_checks(status_checks: list[dict[str, Any]]) -> str:
    run_ids = [
        extract_run_id_from_details_url(str(check.get("detailsUrl") or ""))
        for check in status_checks
        if check.get("workflowName") == "E2E-Full"
    ]
    run_ids = [run_id for run_id in run_ids if run_id]
    if not run_ids:
        return ""
    return max(run_ids, key=int)


def select_latest_marker_comment(comments: list[dict[str, Any]], marker: str) -> MarkerComment | None:
    matches = [item for item in comments if marker in str(item.get("body") or "")]
    if not matches:
        return None
    latest = max(matches, key=lambda item: int(item.get("id") or 0))
    return MarkerComment(id=int(latest["id"]), body=str(latest.get("body") or ""))


def select_bisect_run_id(runs: list[dict[str, Any]], *, caller_run_id: str, dispatch_token: str) -> str:
    caller = f"caller-{caller_run_id}"
    token = f"token-{dispatch_token}"
    for item in sorted(runs, key=_sort_run_key, reverse=True):
        display_title = str(item.get("displayTitle") or "")
        if caller in display_title and token in display_title:
            run_id = item.get("databaseId")
            return str(run_id or "")
    return ""


def load_phase_context(
    pr: dict[str, Any],
    comments: list[dict[str, Any]],
    *,
    expected_phase: str | None = None,
    expected_status: str | None = None,
    allowed_statuses: list[str] | None = None,
    dispatch_token: str | None = None,
) -> PhaseContext:
    state_comment = select_latest_marker_comment(comments, STATE_MARKER)
    if state_comment is None:
        raise ValueError("missing main2main-state comment")
    register_comment = select_latest_marker_comment(comments, REGISTER_MARKER)
    if register_comment is None:
        raise ValueError("missing main2main-register comment")

    state = parse_state_comment(state_comment.body)
    registration = parse_registration_comment(register_comment.body)
    branch = str(pr["headRefName"])
    head_sha = str(pr["headRefOid"])

    guard = check_state_guard(
        state,
        expected_phase=expected_phase,
        expected_status=expected_status,
        dispatch_token=dispatch_token,
    )
    if not guard.ok:
        raise ValueError(guard.reason)

    if allowed_statuses and state.status not in allowed_statuses:
        raise ValueError(f"unexpected status: {state.status}")

    consistency = check_pr_consistency(state, branch=branch, head_sha=head_sha)
    if not consistency.ok:
        raise ValueError(consistency.reason)

    registration_consistency = check_registration_consistency(state, registration)
    if not registration_consistency.ok:
        raise ValueError(registration_consistency.reason)

    return PhaseContext(
        pr=pr,
        state=state,
        registration=registration,
        state_comment_id=state_comment.id,
        register_comment_id=register_comment.id,
        branch=branch,
        head_sha=head_sha,
    )


def normalize_conclusion(conclusion: str) -> str:
    if conclusion == "success":
        return "success"
    if conclusion in _FAILURE_CONCLUSIONS:
        return "failure"
    return "skip"


def is_merge_conflict(*, merge_state_status: str | None = None, mergeable: str | None = None) -> bool:
    if mergeable and mergeable.upper() in _CONFLICT_MERGEABLES:
        return True
    if merge_state_status and merge_state_status.upper() in _CONFLICT_MERGE_STATE_STATUSES:
        return True
    return False


def decide_reconcile_action(
    state: Main2MainState,
    *,
    e2e_run: dict[str, Any] | None = None,
    merge_state_status: str | None = None,
    mergeable: str | None = None,
    bisect_finished: bool = False,
    finalize_missing: bool = False,
) -> ReconcileDecision:
    if is_merge_conflict(merge_state_status=merge_state_status, mergeable=mergeable):
        return ReconcileDecision(
            action="dispatch_manual_review",
            terminal_reason="merge_conflict",
            reason="merge conflict blocks further automation",
        )

    if state.status == "waiting_bisect":
        if bisect_finished and finalize_missing:
            return ReconcileDecision(
                action="dispatch_fix_phase3_finalize",
                reason="bisect finished and finalize callback needs recovery",
                run_id=state.bisect_run_id,
            )
        return ReconcileDecision(action="wait", reason="bisect still in progress or finalize already handled")

    if state.status != "waiting_e2e":
        return ReconcileDecision(action="ignore", reason=f"state {state.status} is not handled by reconcile")

    if e2e_run is None:
        return ReconcileDecision(action="wait", reason="matching E2E run not found yet")

    if e2e_run.get("headSha") != state.head_sha:
        return ReconcileDecision(action="wait", reason="stale E2E run does not match current head_sha")

    if e2e_run.get("status") != "completed":
        return ReconcileDecision(action="wait", reason="matching E2E run is still in progress")

    run_id = str(e2e_run.get("databaseId") or "")
    normalized = normalize_conclusion(str(e2e_run.get("conclusion") or ""))
    if normalized == "success":
        return ReconcileDecision(
            action="dispatch_make_ready",
            reason="latest matching E2E run succeeded",
            run_id=run_id,
        )
    if normalized == "failure":
        if state.phase == "2":
            return ReconcileDecision(
                action="dispatch_fix_phase2",
                reason="phase 2 requires another automated fix attempt",
                run_id=run_id,
            )
        if state.phase == "3":
            return ReconcileDecision(
                action="dispatch_fix_phase3_prepare",
                reason="phase 3 requires bisect-guided automated repair",
                run_id=run_id,
            )
        if state.phase == "done":
            return ReconcileDecision(
                action="dispatch_manual_review",
                terminal_reason="done_failure",
                reason="all automated phases are exhausted",
                run_id=run_id,
            )
    return ReconcileDecision(action="ignore", reason=f"unsupported conclusion: {e2e_run.get('conclusion')}")


def apply_fixup_result(state: Main2MainState, *, new_head_sha: str) -> Main2MainState:
    next_phase = "3" if state.phase == "2" else "done"
    return replace(state, head_sha=new_head_sha, phase=next_phase, status="waiting_e2e", e2e_run_id="")


def apply_no_change_fixup_result(state: Main2MainState) -> Main2MainState:
    if state.phase == "2":
        return replace(state, phase="3", status="waiting_e2e")
    return replace(state, phase="done", status="manual_review_pending")


def prepare_bisect_payload(
    state: Main2MainState,
    *,
    ci_analysis: dict[str, Any] | None = None,
    fallback_test_cmd: str = DEFAULT_BISECT_TEST_CMD,
) -> dict[str, str]:
    analysis = ci_analysis or {}
    test_cmd = str(analysis.get("test_cmd") or "").strip() or fallback_test_cmd
    return {
        "e2e_run_id": state.e2e_run_id,
        "old_commit": state.old_commit,
        "new_commit": state.new_commit,
        "test_cmd": test_cmd,
    }


def prepare_fixing_state(
    state: Main2MainState,
    *,
    fix_run_id: str,
    last_transition: str,
    updated_by: str,
) -> Main2MainState:
    return replace(
        state,
        status="fixing",
        fix_run_id=fix_run_id,
        last_transition=last_transition,
        updated_by=updated_by,
    )


def parse_fixup_job_output(output: str, *, phase: str) -> FixupOutcome:
    if "No changes after phase" in output:
        return FixupOutcome(result="no_changes", phase=phase)
    if "fixes pushed" in output:
        return FixupOutcome(result="changes_pushed", phase=phase)
    raise ValueError("unable to determine fixup outcome from job output")


def _write_json(data: Any) -> None:
    json.dump(data, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_text(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def _write_json_file(path: str, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=True, indent=2))


def _registration_payload_from_state(state: Main2MainState) -> dict[str, Any]:
    return {
        "pr_number": state.pr_number,
        "branch": state.branch,
        "head_sha": state.head_sha,
        "old_commit": state.old_commit,
        "new_commit": state.new_commit,
        "phase": state.phase,
    }


def _run_command(args: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(args)}")
    return result.stdout


def _gh(args: list[str], *, input_text: str | None = None) -> str:
    return _run_command(["gh", *args], input_text=input_text)


def _gh_json(args: list[str]) -> Any:
    output = _gh(args)
    if not output.strip():
        return None
    return json.loads(output)


def _gh_api_json(endpoint: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    args = ["api", endpoint]
    if method != "GET":
        args.extend(["--method", method])
    input_text = None
    if payload is not None:
        args.extend(["--input", "-"])
        input_text = json.dumps(payload)
    output = _gh(args, input_text=input_text)
    if not output.strip():
        return None
    return json.loads(output)


def _patch_state_comment(repo: str, comment_id: int, state: Main2MainState) -> None:
    _gh_api_json(
        f"repos/{repo}/issues/comments/{comment_id}",
        method="PATCH",
        payload={"body": render_state_comment(state)},
    )


def _command_mint_dispatch_token(_args: argparse.Namespace) -> int:
    sys.stdout.write(f"{mint_dispatch_token()}\n")
    return 0


def _command_state_read(args: argparse.Namespace) -> int:
    state = parse_state_comment(_load_text(args.comment_file))
    _write_json(asdict(state))
    return 0


def _command_state_write(args: argparse.Namespace) -> int:
    payload = _load_json(args.json_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    sys.stdout.write(render_state_comment(state))
    sys.stdout.write("\n")
    return 0


def _command_registration_read(args: argparse.Namespace) -> int:
    metadata = parse_registration_comment(_load_text(args.comment_file))
    _write_json(asdict(metadata))
    return 0


def _command_registration_write(args: argparse.Namespace) -> int:
    payload = _load_json(args.json_file)
    metadata = RegistrationMetadata(**payload)
    sys.stdout.write(render_registration_comment(metadata))
    sys.stdout.write("\n")
    return 0


def _command_state_init_from_register(args: argparse.Namespace) -> int:
    metadata = parse_registration_comment(_load_text(args.comment_file))
    state = init_state_from_registration(
        metadata,
        dispatch_token=args.dispatch_token,
        updated_at=args.updated_at,
        updated_by=args.updated_by,
    )
    _write_json(asdict(state))
    return 0


def _command_guard_check(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    result = check_state_guard(
        state,
        expected_phase=args.expected_phase,
        expected_status=args.expected_status,
        dispatch_token=args.dispatch_token,
    )
    _write_json(asdict(result))
    return 0 if result.ok else 1


def _command_pr_consistency_check(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    result = check_pr_consistency(
        state,
        branch=args.branch,
        head_sha=args.head_sha,
    )
    _write_json(asdict(result))
    return 0 if result.ok else 1


def _command_registration_consistency_check(args: argparse.Namespace) -> int:
    state_payload = _load_json(args.state_file)
    registration_payload = _load_json(args.registration_file)
    state = Main2MainState(**_normalize_state_payload(state_payload))
    registration = RegistrationMetadata(**registration_payload)
    result = check_registration_consistency(state, registration)
    _write_json(asdict(result))
    return 0 if result.ok else 1


def _command_reconcile_decision(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    e2e_run = None
    if args.e2e_run_file:
        e2e_run = _load_json(args.e2e_run_file)
    decision = decide_reconcile_action(
        state,
        e2e_run=e2e_run,
        merge_state_status=args.merge_state_status,
        mergeable=args.mergeable,
        bisect_finished=args.bisect_finished,
        finalize_missing=args.finalize_missing,
    )
    _write_json(asdict(decision))
    return 0


def _command_select_e2e_run(args: argparse.Namespace) -> int:
    runs = _load_json(args.runs_file)
    run = select_matching_e2e_run(runs, head_sha=args.head_sha)
    if run is None:
        return 1
    _write_json(run)
    return 0


def _command_apply_fix_result(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    if args.result == "changes_pushed":
        if not args.new_head_sha:
            raise SystemExit("--new-head-sha is required for changes_pushed")
        updated = apply_fixup_result(state, new_head_sha=args.new_head_sha)
    else:
        updated = apply_no_change_fixup_result(state)
    _write_json(asdict(updated))
    return 0


def _command_json_get(args: argparse.Namespace) -> int:
    payload = _load_json(args.json_file)
    value: Any = payload
    for part in args.field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"missing field: {args.field}")
        value = value[part]
    sys.stdout.write(str(value))
    sys.stdout.write("\n")
    return 0


def _command_upsert_pr_phase_section(args: argparse.Namespace) -> int:
    body = _load_text(args.body_file)
    content_lines = _load_text(args.content_file).splitlines()
    updated = upsert_pr_phase_section(body, heading=args.heading, content_lines=content_lines)
    _write_text(args.output_file, updated)
    return 0


def _command_prepare_detect_artifacts(args: argparse.Namespace) -> int:
    register_payload = {
        "pr_number": int(args.pr_number),
        "branch": args.branch,
        "head_sha": args.head_sha,
        "old_commit": args.old_commit,
        "new_commit": args.new_commit,
        "phase": "2",
    }
    state = Main2MainState(
        pr_number=int(args.pr_number),
        branch=args.branch,
        head_sha=args.head_sha,
        old_commit=args.old_commit,
        new_commit=args.new_commit,
        phase="2",
        status="waiting_e2e",
        dispatch_token=args.dispatch_token,
        last_transition="detect->waiting_e2e",
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(state))
    _write_json_file(args.register_json_out, register_payload)
    _write_text(args.state_comment_out, render_state_comment(state) + "\n")
    _write_text(
        args.register_comment_out,
        render_registration_comment(RegistrationMetadata(**register_payload)) + "\n",
    )
    return 0


def _command_prepare_fix_transition(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    if args.result == "changes_pushed":
        if not args.new_head_sha:
            raise SystemExit("--new-head-sha is required for changes_pushed")
        next_state = apply_fixup_result(state, new_head_sha=args.new_head_sha)
    else:
        next_state = apply_no_change_fixup_result(state)
    next_state = replace(
        next_state,
        fix_run_id=args.fix_run_id,
        dispatch_token="" if args.clear_dispatch_token else next_state.dispatch_token,
        workflow_error_count=0,
        last_transition=args.last_transition,
        updated_by=args.updated_by,
    )
    register_payload = _registration_payload_from_state(next_state)
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_json_file(args.register_json_out, register_payload)
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    _write_text(
        args.register_comment_out,
        render_registration_comment(RegistrationMetadata(**register_payload)) + "\n",
    )
    return 0


def _command_prepare_bisect_payload(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    ci_analysis = _load_json(args.ci_analysis_file)
    bisect_payload = prepare_bisect_payload(state, ci_analysis=ci_analysis)
    if args.payload_json_out:
        _write_json_file(args.payload_json_out, bisect_payload)
    else:
        _write_json(bisect_payload)
    return 0


def _command_prepare_fixing_state(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    next_state = prepare_fixing_state(
        state,
        fix_run_id=args.fix_run_id,
        last_transition=args.last_transition,
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    return 0


def _command_prepare_waiting_bisect(args: argparse.Namespace) -> int:
    if not str(args.bisect_run_id or "").strip():
        raise SystemExit("--bisect-run-id is required")
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    next_state = replace(
        state,
        status="waiting_bisect",
        bisect_run_id=args.bisect_run_id,
        fix_run_id=args.fix_run_id,
        workflow_error_count=0,
        last_transition=args.last_transition,
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    return 0


def _command_prepare_manual_review_pending(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    next_state = replace(
        state,
        phase="done",
        status="manual_review_pending",
        terminal_reason=args.terminal_reason,
        fix_run_id=args.fix_run_id,
        workflow_error_count=0,
        last_transition=args.last_transition,
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_json_file(args.register_json_out, _registration_payload_from_state(next_state))
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    _write_text(
        args.register_comment_out,
        render_registration_comment(RegistrationMetadata(**_registration_payload_from_state(next_state))) + "\n",
    )
    return 0


def _command_prepare_workflow_error_action(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    next_count = state.workflow_error_count + 1
    action = "retry" if next_count <= args.max_retries else "manual_review"
    next_state = replace(
        state,
        dispatch_token=args.next_dispatch_token,
        terminal_reason=args.terminal_reason if action == "manual_review" else state.terminal_reason,
        workflow_error_count=next_count,
        last_transition=args.retry_transition if action == "retry" else args.terminal_transition,
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    _write_json({"action": action, "workflow_error_count": next_count, "dispatch_token": next_state.dispatch_token})
    return 0


def _command_prepare_workflow_error_recovery(args: argparse.Namespace) -> int:
    payload = _load_json(args.state_file)
    state = Main2MainState(**_normalize_state_payload(payload))
    next_token = mint_dispatch_token()
    next_count = state.workflow_error_count + 1
    action = "retry" if next_count <= args.max_retries else "manual_review"
    next_state = replace(
        state,
        dispatch_token=next_token,
        terminal_reason=args.terminal_reason if action == "manual_review" else state.terminal_reason,
        workflow_error_count=next_count,
        last_transition=args.retry_transition if action == "retry" else args.terminal_transition,
        updated_by=args.updated_by,
    )
    _write_json_file(args.state_json_out, asdict(next_state))
    _write_text(args.state_comment_out, render_state_comment(next_state) + "\n")
    _write_json({"action": action, "workflow_error_count": next_count, "dispatch_token": next_state.dispatch_token})
    return 0


def _command_extract_pr_comments(args: argparse.Namespace) -> int:
    comments = _load_json(args.comments_file)
    state_comment = select_latest_marker_comment(comments, STATE_MARKER)
    register_comment = select_latest_marker_comment(comments, REGISTER_MARKER)

    if args.state_comment_out:
        if state_comment is None:
            raise SystemExit("missing main2main-state comment")
        _write_text(args.state_comment_out, state_comment.body)
    if args.state_id_out:
        if state_comment is None:
            raise SystemExit("missing main2main-state comment")
        _write_text(args.state_id_out, str(state_comment.id))
    if args.register_comment_out:
        if register_comment is None:
            raise SystemExit("missing main2main-register comment")
        _write_text(args.register_comment_out, register_comment.body)
    if args.register_id_out:
        if register_comment is None:
            raise SystemExit("missing main2main-register comment")
        _write_text(args.register_id_out, str(register_comment.id))
    return 0


def _command_select_bisect_run_id(args: argparse.Namespace) -> int:
    runs = _load_json(args.runs_file)
    run_id = select_bisect_run_id(
        runs,
        caller_run_id=args.caller_run_id,
        dispatch_token=args.dispatch_token,
    )
    sys.stdout.write(run_id)
    sys.stdout.write("\n")
    return 0 if run_id else 1


def _command_load_phase_context(args: argparse.Namespace) -> int:
    pr = _gh_json(
        [
            "pr",
            "view",
            args.pr_number,
            "--repo",
            args.repo,
            "--json",
            "number,headRefName,headRefOid,body,url",
        ]
    )
    comments = _gh_api_json(f"repos/{args.repo}/issues/{args.pr_number}/comments")
    try:
        context = load_phase_context(
            pr,
            comments,
            expected_phase=args.expected_phase,
            expected_status=args.expected_status,
            allowed_statuses=args.allowed_statuses or None,
            dispatch_token=args.dispatch_token,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.pr_json_out:
        _write_json_file(args.pr_json_out, context.pr)
    if args.state_json_out:
        _write_json_file(args.state_json_out, asdict(context.state))
    if args.registration_json_out:
        _write_json_file(args.registration_json_out, asdict(context.registration))
    if args.state_id_out:
        _write_text(args.state_id_out, str(context.state_comment_id))
    if args.register_id_out:
        _write_text(args.register_id_out, str(context.register_comment_id))
    if args.context_json_out:
        _write_json_file(
            args.context_json_out,
            {
                "pr_number": context.state.pr_number,
                "branch": context.branch,
                "head_sha": context.head_sha,
                "state_comment_id": context.state_comment_id,
                "register_comment_id": context.register_comment_id,
                "pr_url": context.pr.get("url", ""),
            },
        )
    return 0


def _reconcile_pr(repo: str, pr_number: str) -> dict[str, Any]:
    pr = _gh_json(
        [
            "pr",
            "view",
            pr_number,
            "--repo",
            repo,
            "--json",
            "number,headRefName,headRefOid,body,mergeable,mergeStateStatus,url,statusCheckRollup",
        ]
    )
    comments = _gh_api_json(f"repos/{repo}/issues/{pr_number}/comments")
    register_comment = select_latest_marker_comment(comments, REGISTER_MARKER)
    state_comment = select_latest_marker_comment(comments, STATE_MARKER)

    if state_comment is None:
        token = mint_dispatch_token()
        if register_comment is not None:
            registration = parse_registration_comment(register_comment.body)
            state = init_state_from_registration(
                registration,
                dispatch_token=token,
                updated_by="schedule_main2main_reconcile.yaml/bootstrap",
            )
        else:
            metadata = parse_pr_metadata(pr.get("body") or "")
            state = Main2MainState(
                pr_number=int(pr["number"]),
                branch=pr["headRefName"],
                head_sha=pr["headRefOid"],
                old_commit=metadata.old_commit,
                new_commit=metadata.new_commit,
                phase="2",
                status="waiting_e2e",
                dispatch_token=token,
                last_transition="reconcile/recover->waiting_e2e",
                updated_by="schedule_main2main_reconcile.yaml/recover",
            )
        created = _gh_api_json(
            f"repos/{repo}/issues/{pr_number}/comments",
            method="POST",
            payload={"body": render_state_comment(state)},
        )
        state_comment_id = int(created["id"])
    else:
        state = parse_state_comment(state_comment.body)
        state_comment_id = state_comment.id

    consistency = check_pr_consistency(
        state,
        branch=pr["headRefName"],
        head_sha=pr["headRefOid"],
    )
    if not consistency.ok:
        return {"action": "skip", "reason": consistency.reason}
    if register_comment is not None:
        registration = parse_registration_comment(register_comment.body)
        registration_consistency = check_registration_consistency(state, registration)
        if not registration_consistency.ok:
            return {"action": "skip", "reason": registration_consistency.reason}
    else:
        _gh_api_json(
            f"repos/{repo}/issues/{pr_number}/comments",
            method="POST",
            payload={
                "body": render_registration_comment(
                    RegistrationMetadata(**_registration_payload_from_state(state))
                )
            },
        )

    if state.status == "waiting_e2e":
        resolved_e2e_run_id = resolve_e2e_run_id_from_status_checks(pr.get("statusCheckRollup") or [])
        e2e_run_id = resolved_e2e_run_id or state.e2e_run_id
        matched_run = None
        if e2e_run_id:
            try:
                matched_run = _gh_json(
                    [
                        "run",
                        "view",
                        e2e_run_id,
                        "--repo",
                        repo,
                        "--json",
                        "databaseId,headSha,status,conclusion,createdAt,url",
                    ]
                )
            except RuntimeError:
                matched_run = None
        if matched_run is None and not e2e_run_id:
            runs = _gh_json(
                [
                    "run",
                    "list",
                    "--repo",
                    repo,
                    "--workflow",
                    "pr_test_full.yaml",
                    "-L",
                    "50",
                    "--json",
                    "databaseId,headSha,status,conclusion,createdAt,url",
                ]
            )
            matched_run = select_matching_e2e_run(runs or [], head_sha=state.head_sha)
            if matched_run is not None:
                e2e_run_id = str(matched_run.get("databaseId") or "")
        if e2e_run_id and state.e2e_run_id != e2e_run_id:
            state = replace(state, e2e_run_id=e2e_run_id, updated_by="schedule_main2main_reconcile.yaml/resolve_e2e")
            _patch_state_comment(repo, state_comment_id, state)
        decision = decide_reconcile_action(
            state,
            e2e_run=matched_run,
            merge_state_status=pr.get("mergeStateStatus"),
            mergeable=pr.get("mergeable"),
        )
    elif state.status == "waiting_bisect":
        bisect_finished = False
        if state.bisect_run_id:
            try:
                bisect_meta = _gh_json(
                    ["run", "view", state.bisect_run_id, "--repo", repo, "--json", "status,conclusion"]
                )
            except RuntimeError:
                bisect_meta = {}
            bisect_finished = (bisect_meta or {}).get("status") == "completed"
        decision = decide_reconcile_action(
            state,
            merge_state_status=pr.get("mergeStateStatus"),
            mergeable=pr.get("mergeable"),
            bisect_finished=bisect_finished,
            finalize_missing=bisect_finished and state.last_transition != "reconcile->fix_phase3_finalize",
        )
    else:
        decision = ReconcileDecision(action="ignore", reason=f"state {state.status} is not handled by reconcile")

    if decision.action in {"wait", "ignore"}:
        return {"action": decision.action, "reason": decision.reason}

    token = mint_dispatch_token()
    next_state = replace(state, dispatch_token=token, updated_by="schedule_main2main_reconcile.yaml")
    if decision.run_id:
        next_state = replace(next_state, e2e_run_id=decision.run_id)
    if decision.action == "dispatch_fix_phase2":
        next_state = replace(
            next_state,
            status="fixing",
            phase="2",
            workflow_error_count=0,
            last_transition="reconcile->fix_phase2",
        )
    elif decision.action == "dispatch_fix_phase3_prepare":
        next_state = replace(
            next_state,
            status="fixing",
            phase="3",
            workflow_error_count=0,
            last_transition="reconcile->fix_phase3_prepare",
        )
    elif decision.action == "dispatch_manual_review":
        next_state = replace(
            next_state,
            terminal_reason=decision.terminal_reason,
            workflow_error_count=0,
            last_transition="reconcile->manual_review",
        )
    elif decision.action == "dispatch_make_ready":
        next_state = replace(next_state, workflow_error_count=0, last_transition="reconcile->make_ready")
    elif decision.action == "dispatch_fix_phase3_finalize":
        next_state = replace(next_state, workflow_error_count=0, last_transition="reconcile->fix_phase3_finalize")

    _gh_api_json(
        f"repos/{repo}/issues/comments/{state_comment_id}",
        method="PATCH",
        payload={"body": render_state_comment(next_state)},
    )

    if decision.action == "dispatch_fix_phase2":
        _gh(
            [
                "workflow",
                "run",
                "schedule_main2main_auto.yaml",
                "--repo",
                repo,
                "-f",
                "mode=fix_phase2",
                "-f",
                f"pr_number={pr_number}",
                "-f",
                f"dispatch_token={token}",
            ]
        )
    elif decision.action == "dispatch_fix_phase3_prepare":
        _gh(
            [
                "workflow",
                "run",
                "schedule_main2main_auto.yaml",
                "--repo",
                repo,
                "-f",
                "mode=fix_phase3_prepare",
                "-f",
                f"pr_number={pr_number}",
                "-f",
                f"dispatch_token={token}",
            ]
        )
    elif decision.action == "dispatch_fix_phase3_finalize":
        _gh(
            [
                "workflow",
                "run",
                "schedule_main2main_auto.yaml",
                "--repo",
                repo,
                "-f",
                "mode=fix_phase3_finalize",
                "-f",
                f"pr_number={pr_number}",
                "-f",
                f"dispatch_token={token}",
                "-f",
                f"bisect_run_id={next_state.bisect_run_id}",
            ]
        )
    elif decision.action == "dispatch_make_ready":
        _gh(
            [
                "workflow",
                "run",
                "dispatch_main2main_terminal.yaml",
                "--repo",
                repo,
                "-f",
                "action=make_ready",
                "-f",
                f"pr_number={pr_number}",
                "-f",
                f"dispatch_token={token}",
            ]
        )
    elif decision.action == "dispatch_manual_review":
        _gh(
            [
                "workflow",
                "run",
                "dispatch_main2main_terminal.yaml",
                "--repo",
                repo,
                "-f",
                "action=manual_review",
                "-f",
                f"pr_number={pr_number}",
                "-f",
                f"dispatch_token={token}",
                "-f",
                f"terminal_reason={decision.terminal_reason}",
            ]
        )

    return {"action": decision.action, "reason": decision.reason, "run_id": decision.run_id}


def _command_reconcile_pr(args: argparse.Namespace) -> int:
    result = _reconcile_pr(args.repo, args.pr_number)
    _write_json(result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workflow-native main2main CI helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mint = subparsers.add_parser("mint-dispatch-token")
    mint.set_defaults(func=_command_mint_dispatch_token)

    state_read = subparsers.add_parser("state-read")
    state_read.add_argument("--comment-file", required=True)
    state_read.set_defaults(func=_command_state_read)

    state_write = subparsers.add_parser("state-write")
    state_write.add_argument("--json-file", required=True)
    state_write.set_defaults(func=_command_state_write)

    register_read = subparsers.add_parser("registration-read")
    register_read.add_argument("--comment-file", required=True)
    register_read.set_defaults(func=_command_registration_read)

    register_write = subparsers.add_parser("registration-write")
    register_write.add_argument("--json-file", required=True)
    register_write.set_defaults(func=_command_registration_write)

    init_state = subparsers.add_parser("state-init-from-register")
    init_state.add_argument("--comment-file", required=True)
    init_state.add_argument("--dispatch-token", required=True)
    init_state.add_argument("--updated-at", default="")
    init_state.add_argument("--updated-by", default="")
    init_state.set_defaults(func=_command_state_init_from_register)

    guard = subparsers.add_parser("guard-check")
    guard.add_argument("--state-file", required=True)
    guard.add_argument("--expected-phase", default=None)
    guard.add_argument("--expected-status", default=None)
    guard.add_argument("--dispatch-token", default=None)
    guard.set_defaults(func=_command_guard_check)

    consistency = subparsers.add_parser("pr-consistency-check")
    consistency.add_argument("--state-file", required=True)
    consistency.add_argument("--branch", required=True)
    consistency.add_argument("--head-sha", required=True)
    consistency.set_defaults(func=_command_pr_consistency_check)

    registration_consistency = subparsers.add_parser("registration-consistency-check")
    registration_consistency.add_argument("--state-file", required=True)
    registration_consistency.add_argument("--registration-file", required=True)
    registration_consistency.set_defaults(func=_command_registration_consistency_check)

    reconcile = subparsers.add_parser("reconcile-decision")
    reconcile.add_argument("--state-file", required=True)
    reconcile.add_argument("--e2e-run-file", default="")
    reconcile.add_argument("--merge-state-status", default=None)
    reconcile.add_argument("--mergeable", default=None)
    reconcile.add_argument("--bisect-finished", action="store_true")
    reconcile.add_argument("--finalize-missing", action="store_true")
    reconcile.set_defaults(func=_command_reconcile_decision)

    reconcile_pr = subparsers.add_parser("reconcile-pr")
    reconcile_pr.add_argument("--repo", required=True)
    reconcile_pr.add_argument("--pr-number", required=True)
    reconcile_pr.set_defaults(func=_command_reconcile_pr)

    select_run = subparsers.add_parser("select-e2e-run")
    select_run.add_argument("--runs-file", required=True)
    select_run.add_argument("--head-sha", required=True)
    select_run.set_defaults(func=_command_select_e2e_run)

    apply_fix = subparsers.add_parser("apply-fix-result")
    apply_fix.add_argument("--state-file", required=True)
    apply_fix.add_argument("--result", required=True, choices=["changes_pushed", "no_changes"])
    apply_fix.add_argument("--new-head-sha", default="")
    apply_fix.set_defaults(func=_command_apply_fix_result)

    json_get = subparsers.add_parser("json-get")
    json_get.add_argument("--json-file", required=True)
    json_get.add_argument("--field", required=True)
    json_get.set_defaults(func=_command_json_get)

    upsert_section = subparsers.add_parser("upsert-pr-phase-section")
    upsert_section.add_argument("--body-file", required=True)
    upsert_section.add_argument("--heading", required=True)
    upsert_section.add_argument("--content-file", required=True)
    upsert_section.add_argument("--output-file", required=True)
    upsert_section.set_defaults(func=_command_upsert_pr_phase_section)

    detect_artifacts = subparsers.add_parser("prepare-detect-artifacts")
    detect_artifacts.add_argument("--pr-number", required=True)
    detect_artifacts.add_argument("--branch", required=True)
    detect_artifacts.add_argument("--head-sha", required=True)
    detect_artifacts.add_argument("--old-commit", required=True)
    detect_artifacts.add_argument("--new-commit", required=True)
    detect_artifacts.add_argument("--dispatch-token", required=True)
    detect_artifacts.add_argument("--updated-by", default="schedule_main2main_auto.yaml/detect")
    detect_artifacts.add_argument("--state-json-out", required=True)
    detect_artifacts.add_argument("--register-json-out", required=True)
    detect_artifacts.add_argument("--state-comment-out", required=True)
    detect_artifacts.add_argument("--register-comment-out", required=True)
    detect_artifacts.set_defaults(func=_command_prepare_detect_artifacts)

    fix_transition = subparsers.add_parser("prepare-fix-transition")
    fix_transition.add_argument("--state-file", required=True)
    fix_transition.add_argument("--result", required=True, choices=["changes_pushed", "no_changes"])
    fix_transition.add_argument("--new-head-sha", default="")
    fix_transition.add_argument("--fix-run-id", required=True)
    fix_transition.add_argument("--last-transition", required=True)
    fix_transition.add_argument("--updated-by", required=True)
    fix_transition.add_argument("--state-json-out", required=True)
    fix_transition.add_argument("--register-json-out", required=True)
    fix_transition.add_argument("--state-comment-out", required=True)
    fix_transition.add_argument("--register-comment-out", required=True)
    fix_transition.add_argument("--clear-dispatch-token", action="store_true")
    fix_transition.set_defaults(func=_command_prepare_fix_transition)

    bisect_payload = subparsers.add_parser("prepare-bisect-payload")
    bisect_payload.add_argument("--state-file", required=True)
    bisect_payload.add_argument("--ci-analysis-file", required=True)
    bisect_payload.add_argument("--payload-json-out", required=False)
    bisect_payload.set_defaults(func=_command_prepare_bisect_payload)

    fixing_state = subparsers.add_parser("prepare-fixing-state")
    fixing_state.add_argument("--state-file", required=True)
    fixing_state.add_argument("--fix-run-id", required=True)
    fixing_state.add_argument("--last-transition", required=True)
    fixing_state.add_argument("--updated-by", required=True)
    fixing_state.add_argument("--state-json-out", required=True)
    fixing_state.add_argument("--state-comment-out", required=True)
    fixing_state.set_defaults(func=_command_prepare_fixing_state)

    waiting_bisect = subparsers.add_parser("prepare-waiting-bisect")
    waiting_bisect.add_argument("--state-file", required=True)
    waiting_bisect.add_argument("--bisect-run-id", required=True)
    waiting_bisect.add_argument("--fix-run-id", required=True)
    waiting_bisect.add_argument("--last-transition", required=True)
    waiting_bisect.add_argument("--updated-by", required=True)
    waiting_bisect.add_argument("--state-json-out", required=True)
    waiting_bisect.add_argument("--state-comment-out", required=True)
    waiting_bisect.set_defaults(func=_command_prepare_waiting_bisect)

    manual_review_pending = subparsers.add_parser("prepare-manual-review-pending")
    manual_review_pending.add_argument("--state-file", required=True)
    manual_review_pending.add_argument("--terminal-reason", required=True)
    manual_review_pending.add_argument("--fix-run-id", required=True)
    manual_review_pending.add_argument("--last-transition", required=True)
    manual_review_pending.add_argument("--updated-by", required=True)
    manual_review_pending.add_argument("--state-json-out", required=True)
    manual_review_pending.add_argument("--register-json-out", required=True)
    manual_review_pending.add_argument("--state-comment-out", required=True)
    manual_review_pending.add_argument("--register-comment-out", required=True)
    manual_review_pending.set_defaults(func=_command_prepare_manual_review_pending)

    workflow_error = subparsers.add_parser("prepare-workflow-error-action")
    workflow_error.add_argument("--state-file", required=True)
    workflow_error.add_argument("--next-dispatch-token", required=True)
    workflow_error.add_argument("--max-retries", type=int, default=1)
    workflow_error.add_argument("--terminal-reason", default="workflow_error")
    workflow_error.add_argument("--retry-transition", required=True)
    workflow_error.add_argument("--terminal-transition", required=True)
    workflow_error.add_argument("--updated-by", required=True)
    workflow_error.add_argument("--state-json-out", required=True)
    workflow_error.add_argument("--state-comment-out", required=True)
    workflow_error.set_defaults(func=_command_prepare_workflow_error_action)

    workflow_error_recovery = subparsers.add_parser("prepare-workflow-error-recovery")
    workflow_error_recovery.add_argument("--state-file", required=True)
    workflow_error_recovery.add_argument("--max-retries", type=int, default=1)
    workflow_error_recovery.add_argument("--terminal-reason", default="workflow_error")
    workflow_error_recovery.add_argument("--retry-transition", required=True)
    workflow_error_recovery.add_argument("--terminal-transition", required=True)
    workflow_error_recovery.add_argument("--updated-by", required=True)
    workflow_error_recovery.add_argument("--state-json-out", required=True)
    workflow_error_recovery.add_argument("--state-comment-out", required=True)
    workflow_error_recovery.set_defaults(func=_command_prepare_workflow_error_recovery)

    extract_comments = subparsers.add_parser("extract-pr-comments")
    extract_comments.add_argument("--comments-file", required=True)
    extract_comments.add_argument("--state-comment-out", default="")
    extract_comments.add_argument("--state-id-out", default="")
    extract_comments.add_argument("--register-comment-out", default="")
    extract_comments.add_argument("--register-id-out", default="")
    extract_comments.set_defaults(func=_command_extract_pr_comments)

    select_bisect = subparsers.add_parser("select-bisect-run-id")
    select_bisect.add_argument("--runs-file", required=True)
    select_bisect.add_argument("--caller-run-id", required=True)
    select_bisect.add_argument("--dispatch-token", required=True)
    select_bisect.set_defaults(func=_command_select_bisect_run_id)

    load_phase = subparsers.add_parser("load-phase-context")
    load_phase.add_argument("--repo", required=True)
    load_phase.add_argument("--pr-number", required=True)
    load_phase.add_argument("--expected-phase", default=None)
    load_phase.add_argument("--expected-status", default=None)
    load_phase.add_argument("--allowed-statuses", nargs="*", default=[])
    load_phase.add_argument("--dispatch-token", default=None)
    load_phase.add_argument("--pr-json-out", default="")
    load_phase.add_argument("--state-json-out", default="")
    load_phase.add_argument("--registration-json-out", default="")
    load_phase.add_argument("--state-id-out", default="")
    load_phase.add_argument("--register-id-out", default="")
    load_phase.add_argument("--context-json-out", default="")
    load_phase.set_defaults(func=_command_load_phase_context)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
