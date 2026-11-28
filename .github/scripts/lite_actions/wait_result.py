#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Consume only identity-matched results; never upload training logs."""

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = "cann-framework/vllm-ascend"
STATUS = {
    "PASS",
    "FAIL",
    "INFRA_ERROR",
    "RESOURCE_TIMEOUT",
    "TIMED_OUT",
    "CANCELLED",
    "NOT_RUN",
}


def same_json_value(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def find_result(
    comments,
    *,
    run_id,
    attempt,
    sha,
    digest,
    item,
    workflow,
    author,
    native_source=None,
):
    prefix = f"Lite-CI-V3 {run_id}:{attempt}:{item['case_id']}\n"
    accepted = None
    invalid = False
    for c in reversed(comments):
        if (
            not author
            or c.get("user", {}).get("login") != author
            or not c.get("body", "").startswith(prefix)
        ):
            continue
        try:
            r = json.loads(c["body"].split("\n", 1)[1])
        except ValueError:
            invalid = True
            continue
        if not isinstance(r, dict):
            invalid = True
            continue
        expected = dict(
            schema=3,
            repo=REPO,
            workflow=workflow,
            run_id=run_id,
            attempt=attempt,
            sha=sha,
            plan_digest=digest,
            case_id=item["case_id"],
            test_id=item["test_id"],
            worker=item["worker"],
            target_pool=item["target_pool"],
            docker=item["docker"],
            resources=item["resources"],
        )
        if any(k not in r or not same_json_value(r[k], v) for k, v in expected.items()):
            invalid = True
            continue
        if native_source is not None and not same_json_value(r.get("native_source"), native_source):
            invalid = True
            continue
        lines = r.get("last_20_lines")
        if (
            not isinstance(r.get("status"), str)
            or r.get("status") not in STATUS
            or not isinstance(lines, list)
            or len(lines) > 20
            or any(
                not isinstance(s, str) or len(s) > 180 or any(ord(c) < 32 for c in s)
                for s in lines
            )
        ):
            invalid = True
            continue
        if r["status"] == "PASS" and (
            type(r.get("exit_code")) is not int
            or r["exit_code"] != 0
            or r.get("cleanup_verified") is not True
        ):
            invalid = True
            continue
        if (
            len(c["body"].encode()) > 6100
            or len(json.dumps({"body": c["body"]}, separators=(",", ":")).encode())
            > 7000
        ):
            invalid = True
            continue
        if accepted is not None and not same_json_value(accepted, r):
            raise ValueError("conflicting case results")
        accepted = r
    if invalid and accepted is not None:
        raise ValueError("invalid case result alongside a matching result")
    return accepted


def find_progress(comments, run_id, attempt, item, digest, author):
    for c in reversed(comments):
        if c.get("user", {}).get("login") != author or not c.get("body", "").startswith(
            "Lite-CI-PROGRESS "
        ):
            continue
        try:
            r = json.loads(c["body"][17:])
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        if all(
            r.get(k) == v
            for k, v in dict(
                run_id=run_id,
                attempt=attempt,
                case_id=item["case_id"],
                worker=item["worker"],
                plan_digest=digest,
            ).items()
        ):
            return r.get("phase", "RECEIVED")
    return None


BEIJING = dt.timezone(dt.timedelta(hours=8))


def heartbeat_line(*, status, current_case):
    timestamp = dt.datetime.now(dt.UTC).astimezone(BEIJING)
    return f"{timestamp:%Y/%m/%d %H:%M:%S} {current_case} CI {status.lower()}."


HTTP_CACHE = {}


class RetryableAPIError(RuntimeError):
    pass


def get(path, token):
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
    }
    previous = HTTP_CACHE.get(path)
    if previous and previous[0]:
        headers["If-None-Match"] = previous[0]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}" + path, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response)
            HTTP_CACHE[path] = (response.headers.get("ETag"), data)
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and previous:
            return previous[1]
        if exc.code in (429, 500, 502, 503, 504) or (
            exc.code == 403
            and (
                exc.headers.get("X-RateLimit-Remaining") == "0"
                or exc.headers.get("Retry-After")
            )
        ):
            raise RetryableAPIError(
                f"GitHub temporarily unavailable: HTTP {exc.code}"
            ) from exc
        raise
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableAPIError("GitHub temporarily unreachable") from exc


def main():
    if os.environ["GITHUB_REPOSITORY"] != REPO:
        raise ValueError("untrusted repository")
    author = os.environ.get("CI_RESULT_AUTHOR")
    if not author:
        raise ValueError("CI_RESULT_AUTHOR must be configured")
    item = json.loads(os.environ["CI_CASE"])
    run_id = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    token = os.environ["GH_TOKEN"]
    sha = os.environ["CI_SOURCE_SHA"]
    digest = os.environ["CI_PLAN_DIGEST"]
    workflow = os.environ["CI_WORKFLOW"]
    native_source = None
    if os.environ.get("CI_NATIVE_SOURCE"):
        from pathlib import Path

        native_source = json.loads(Path(os.environ["CI_NATIVE_SOURCE"]).read_text())
        if (
            native_source.get("run_id") != run_id
            or native_source.get("attempt") != attempt
            or native_source.get("head_sha") != sha
            or native_source.get("plan_digest") != digest
            or native_source.get("control_sha") != os.environ["GITHUB_WORKFLOW_SHA"]
        ):
            raise ValueError("Waiter native source identity mismatch")
    start = time.monotonic()
    heartbeat = 0
    accepted = False
    while True:
        try:
            artifacts = get(f"/actions/runs/{run_id}/artifacts?per_page=100", token)
            break
        except RetryableAPIError:
            if time.monotonic() - start >= 300:
                raise TimeoutError("GitHub artifact API unavailable") from None
            time.sleep(60)
    if not any(
        a["name"] == f"lite-ci-request-{attempt}" and not a.get("expired")
        for a in artifacts.get("artifacts", [])
    ):
        raise ValueError("No plan for this attempt; re-run all jobs")
    while time.monotonic() - start < 16800:
        try:
            comments = []
            for page in range(1, 21):
                rows = get(f"/commits/{sha}/comments?per_page=100&page={page}", token)
                comments.extend(rows)
                if len(rows) < 100:
                    break
            else:
                raise ValueError("comment listing exceeds correlation budget")
        except RetryableAPIError:
            print(
                heartbeat_line(
                    status="WAITING_GITHUB_API", current_case=item["test_id"]
                ),
                flush=True,
            )
            time.sleep(60)
            continue
        phase = find_progress(comments, run_id, attempt, item, digest, author)
        accepted = accepted or phase is not None
        result = find_result(
            comments,
            run_id=run_id,
            attempt=attempt,
            sha=sha,
            digest=digest,
            item=item,
            workflow=workflow,
            author=author,
            native_source=native_source,
        )
        if result:
            if result["status"] == "PASS" and native_source is not None:
                from native_source import assert_current

                assert_current(native_source, lambda path: get(path, token))
            print(
                heartbeat_line(status=result["status"], current_case=item["test_id"]),
                flush=True,
            )
            print(
                f"CASE {item['test_id']}: {result['status']} rc={result['exit_code']}"
            )
            for line in result["last_20_lines"]:
                print(line)
            if os.environ.get("GITHUB_STEP_SUMMARY"):
                with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                    f.write("### " + item["test_id"] + "\n\n")
                    for k in ["worker", "target_pool", "docker", "resources"]:
                        f.write(f"- {k}: `{item[k]}`\n")
                    f.write(f"- SHA: `{sha}`\n- Status: **{result['status']}**\n")
                    if native_source is not None:
                        f.write(
                            "- Test source: `"
                            + json.dumps(native_source, sort_keys=True)
                            + "`\n"
                        )
                        f.write(
                            "- Merge guarantee: versioned_only; no atomic merge permission.\n"
                        )
            return 0 if result["status"] == "PASS" else 1
        if not accepted and time.monotonic() - start > 300:
            raise TimeoutError(
                "INFRA_ERROR: assigned worker did not acknowledge within 300 seconds"
            )
        if time.monotonic() >= heartbeat:
            print(
                heartbeat_line(
                    status=phase or "WAITING_WORKER", current_case=item["test_id"]
                ),
                flush=True,
            )
            heartbeat = time.monotonic() + 60
        time.sleep(60)
    raise TimeoutError("TIMED_OUT waiting for assigned worker")


if __name__ == "__main__":
    raise SystemExit(main())
