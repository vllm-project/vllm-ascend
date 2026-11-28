"""Publish a version-bound test status, not an atomic merge authorization."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from blue_catalog import BLUE_SMOKE_SUITE
from native_source import REPO, api, assert_current


def context(pr_number):
    if type(pr_number) is not int or pr_number < 1:
        raise ValueError("invalid PR number")
    return f"Lite-Actions-Blue-Merge-PR-{pr_number}"


def description(base_sha, test_ids, base_ref):
    spec = hashlib.sha256(
        json.dumps([base_ref, test_ids], separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"Versioned test B={base_sha} S={spec}; not atomic merge permission"


def post(path, payload):
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("invalid repository write path")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        "https://api.github.com/repos/" + REPO + path,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.build_opener(NoRedirect()).open(
        request, timeout=25
    ) as response:
        content = response.read(1024**2 + 1)
    if len(content) > 1024**2:
        raise ValueError("oversized status response")
    return json.loads(content) if content else None


def publish(source, state):
    if state not in {"pending", "success", "failure"}:
        raise ValueError("invalid test state")
    if source["mode"] != "merge_result":
        return
    if source["test_ids"] != list(BLUE_SMOKE_SUITE):
        return  # A single selected case must not publish a full-suite status.
    assert_current(source)
    run = api(f"/actions/runs/{source['run_id']}")
    if (
        run.get("run_attempt") != source["attempt"]
        or run.get("status") != "in_progress"
    ):
        raise ValueError(
            "superseded/cancelled invocation cannot publish current status"
        )
    post(
        "/statuses/" + source["head_sha"],
        {
            "state": state,
            "context": context(source["pr_number"]),
            "description": description(source["base_sha"], source["test_ids"], source["base_ref"]),
            "target_url": f"https://github.com/{REPO}/actions/runs/{source['run_id']}/attempts/{source['attempt']}",
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--state", choices=("pending", "success", "failure"), required=True)
    a = p.parse_args()
    source = json.loads(a.source.read_text())
    if (
        source.get("repository") != REPO
        or source.get("run_id") != int(os.environ["GITHUB_RUN_ID"])
        or source.get("attempt") != int(os.environ["GITHUB_RUN_ATTEMPT"])
        or source.get("control_sha") != os.environ["GITHUB_WORKFLOW_SHA"]
    ):
        raise ValueError("result source/run identity mismatch")
    publish(source, a.state)
    print(
        json.dumps(
            {
                "test_result": a.state,
                "merge_gate": "NOT_PROVIDED",
                "mode": "versioned_only",
                "H": source["head_sha"],
                "B": source["base_sha"],
                "M": source["source_sha"],
                "T": source["tree_sha"],
            }
        )
    )


if __name__ == "__main__":
    main()
