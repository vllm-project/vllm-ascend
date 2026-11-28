#!/usr/bin/env python3
"""Build the versioned plan without importing fork Python in control jobs."""

import hashlib
import json
import os
import re
from pathlib import Path

from blue_catalog import BLUE_CASES, BLUE_DEFAULTS, BLUE_SMOKE_SUITE

REPO = "cann-framework/vllm-ascend"
WORKFLOW = "lite-ci-dispatcher-smoke.yml"



def selectors(event, payload, value):
    if event == "pull_request_target":
        pr = payload["pull_request"]
        base, head = pr["base"], pr["head"]
        if (
            base["repo"]["full_name"] != REPO
            or not base.get("ref")
            or not head.get("repo", {}).get("full_name")
            or pr["state"] != "open"
        ):
            raise ValueError("invalid PR source or target repository")
        sha = head["sha"]
    elif event == "workflow_dispatch":
        if not os.environ.get("GITHUB_REF", "").startswith("refs/heads/"):
            raise ValueError("manual run requires a branch")
        sha = os.environ.get("CI_COMMIT") or os.environ["GITHUB_SHA"]
    else:
        raise ValueError("unsupported GitHub event")
    if value and value not in BLUE_SMOKE_SUITE:
        raise ValueError("unknown dispatch Smoke case")
    ids = [value] if value else list(BLUE_SMOKE_SUITE)
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("invalid source SHA")
    if not ids or len(ids) > 256 or any(x not in BLUE_CASES for x in ids):
        raise ValueError("unknown smoke case")
    return sha, ids


def create_plan(event, payload, value, run_id, attempt):
    sha, ids = selectors(event, payload, value)
    defaults = BLUE_DEFAULTS
    items = []
    for i, test_id in enumerate(ids):
        test = BLUE_CASES[test_id]
        nodes, ngpu, timeout = test.nnodes, test.ngpu, test.timeout
        label, kind = test.test_descr, test.ci_kind
        overrides = test.ci_overrides or {}
        if not isinstance(overrides, dict) or set(overrides) - {"worker", "target_pool", "docker"}:
            raise ValueError("unknown case override")
        if kind == "worker_probe" and set(overrides) & {"target_pool", "docker"}:
            raise ValueError("probe cannot request resources")
        config = {**defaults, **overrides}
        items.append(
            {
                "case_id": f"case_{i:03d}_{test_id}",
                "order": i,
                "test_id": test_id,
                "label": label,
                "kind": kind,
                "worker": config["worker"],
                "target_pool": config["target_pool"] if kind != "worker_probe" else None,
                "resources": {"nodes": nodes, "devices_per_node": ngpu},
                "docker": config["docker"] if kind != "worker_probe" else None,
                "execution_timeout_seconds": timeout,
            }
        )
    plan = {
        "schema": 3,
        "repo": REPO,
        "workflow": WORKFLOW,
        "run_id": run_id,
        "attempt": attempt,
        "sha": sha,
        "defaults": defaults,
        "cases": items,
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return plan


def main():
    if os.environ["GITHUB_REPOSITORY"] != REPO:
        raise ValueError("untrusted repository")
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    effective_event = os.environ["GITHUB_EVENT_NAME"]
    source_pr = os.environ.get("CI_SOURCE_PR", "")
    commit = os.environ.get("CI_COMMIT", "")
    resolve_only = os.environ.get("CI_RESOLVE_ONLY") == "1"
    test_case = "" if resolve_only else os.environ.get("CI_TEST_CASE", "")
    if source_pr and commit:
        raise ValueError("choose either a PR or a commit")
    if source_pr:
        if effective_event != "workflow_dispatch" or not re.fullmatch(r"[1-9][0-9]{0,8}", source_pr):
            raise ValueError("invalid native PR recheck")
        from native_source import current_pr

        pr, _, _ = current_pr(int(source_pr))
        payload = {"pull_request": pr}
        effective_event = "pull_request_target"
    plan = create_plan(
        effective_event,
        payload,
        test_case,
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
    )
    out = Path(".ci-request")
    out.mkdir(exist_ok=True)
    from native_source import prepare_source

    source = prepare_source(plan, payload)
    if resolve_only:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("source_sha=" + source["source_sha"] + "\n")
        return
    if source["source_sha"] != os.environ.get("CI_CHECKED_SOURCE_SHA"):
        raise ValueError("tested source changed while preparing the plan")
    out.joinpath("source.json").write_text(json.dumps(source, sort_keys=True) + "\n")
    (out / "ci-request.json").write_text(json.dumps(plan, separators=(",", ":"), ensure_ascii=False) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("native_source=true\n")
        output.write("matrix=" + json.dumps({"include": plan["cases"]}, separators=(",", ":")) + "\n")
        output.write("sha=" + plan["sha"] + "\n")
        output.write("source_sha=" + source["source_sha"] + "\n")
        output.write("plan_digest=" + plan["plan_digest"] + "\n")


if __name__ == "__main__":
    main()
