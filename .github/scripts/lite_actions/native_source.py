"""Read-only GitHub merge-source preparation; never imports PR Python."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

REPO = "cann-framework/vllm-ascend"
REPO_ID = 1438625957
SHA = re.compile(r"[0-9a-f]{40}\Z")


def api(path):
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("relative repository API path required")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        "https://api.github.com/repos/" + REPO + path,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=25) as response:
        data = response.read(2 * 1024**2 + 1)
    if len(data) > 2 * 1024**2:
        raise ValueError("source authority response exceeds budget")
    return json.loads(data)


def current_pr(number, get=api):
    if type(number) is not int or number < 1:
        raise ValueError("invalid PR number")
    pr = get(f"/pulls/{number}")
    head_repo = pr.get("head", {}).get("repo") or {}
    base_ref = pr.get("base", {}).get("ref")
    if (
        pr.get("number") != number
        or pr.get("state") != "open"
        or pr.get("merged")
        or pr.get("base", {}).get("repo", {}).get("id") != REPO_ID
        or pr["base"]["repo"].get("full_name") != REPO
        or not isinstance(base_ref, str)
        or not base_ref
        or len(base_ref) > 255
        or any(ord(c) < 32 for c in base_ref)
        or type(head_repo.get("id")) is not int
        or head_repo["id"] < 1
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", head_repo.get("full_name", ""))
    ):
        raise ValueError("unauthorized Blue PR source")
    head = pr["head"].get("sha")
    base = get("/branches/" + urllib.parse.quote(base_ref, safe="")).get("commit", {}).get("sha")
    if not isinstance(head, str) or not SHA.fullmatch(head) or not isinstance(base, str) or not SHA.fullmatch(base):
        raise ValueError("invalid Head/Base SHA")
    return pr, head, base


def assert_current(source, get=api):
    if source["mode"] == "merge_result":
        pr, head, base = current_pr(source["pr_number"], get)
        if (head, base, pr["head"]["repo"]["id"], pr["base"]["ref"]) != (
            source["head_sha"], source["base_sha"], source["head_repo_id"], source["base_ref"]
        ):
            raise ValueError("STALE: PR Head/Base advanced; rerun from the current PR")
def build_source(plan, event, get=api):
    control = os.environ["GITHUB_WORKFLOW_SHA"]
    if not SHA.fullmatch(control):
        raise ValueError("invalid immutable controller SHA")
    source = {
        "schema": 1,
        "provider": "github",
        "repository": REPO,
        "repository_id": REPO_ID,
        "run_id": plan["run_id"],
        "attempt": plan["attempt"],
        "plan_digest": plan["plan_digest"],
        "control_sha": control,
        "head_sha": plan["sha"],
        "test_ids": [c["test_id"] for c in plan["cases"]],
    }
    if "pull_request" in event:
        number = event["pull_request"]["number"]
        pr, head, base = current_pr(number, get)
        if head != plan["sha"]:
            raise ValueError("STALE: event Head is no longer current")
        merge = pr.get("merge_commit_sha")
        if pr.get("mergeable") is False:
            raise ValueError("MERGE_CONFLICT")
        if not isinstance(merge, str) or not SHA.fullmatch(merge):
            raise ValueError("MERGE_NOT_READY")
        commit = get("/git/commits/" + merge)
        if commit.get("sha") != merge or [p.get("sha") for p in commit.get("parents", [])] != [base, head]:
            raise ValueError("STALE_MERGE_PARENTS")
        source.update(
            mode="merge_result",
            pr_number=number,
            head_repo_id=pr["head"]["repo"]["id"],
            base_ref=pr["base"]["ref"],
            base_sha=base,
            source_sha=merge,
            tree_sha=commit["tree"]["sha"],
        )
    else:
        commit = get("/git/commits/" + plan["sha"])
        if commit.get("sha") != plan["sha"]:
            raise ValueError("manual source commit identity mismatch")
        source.update(
            mode="commit",
            pr_number=0,
            head_repo_id=REPO_ID,
            base_ref=os.environ["GITHUB_REF"].removeprefix("refs/heads/"),
            base_sha=plan["sha"],
            source_sha=plan["sha"],
            tree_sha=commit["tree"]["sha"],
        )
    if not SHA.fullmatch(source["tree_sha"]):
        raise ValueError("invalid tested tree")
    assert_current(source, get)
    return source


def prepare_source(plan, event):
    for attempt in range(5):
        try:
            return build_source(plan, event)
        except ValueError as exc:
            if str(exc) not in {"MERGE_NOT_READY", "STALE_MERGE_PARENTS"} or attempt == 4:
                raise
            time.sleep(2)
    raise ValueError("merge source unavailable")
