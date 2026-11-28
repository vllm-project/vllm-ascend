#!/usr/bin/env python3
#
# Copyright (c) 2026 vLLM Ascend contributors.
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

"""Check the PR diff and the authorization for its CI waiver."""

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

PAGE_SIZE = 100
FILE_LIST_LIMIT = 3000


def read_api(url: str, token: str):
    request = Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def skip_ci_label(pull: dict) -> str | None:
    return next((label["name"] for label in pull["labels"] if label["name"].casefold() == "skip_ci"), None)


def skip_ci_events(repo_url: str, token: str, pr_number: int) -> Iterator[dict]:
    page = 1
    while True:
        events = read_api(f"{repo_url}/issues/{pr_number}/events?per_page={PAGE_SIZE}&page={page}", token)
        for event in events:
            if event["event"] in ("labeled", "unlabeled") and event["label"]["name"].casefold() == "skip_ci":
                yield event
        if len(events) < PAGE_SIZE:
            return
        page += 1


def active_label_event(repo_url: str, token: str, pr_number: int) -> dict | None:
    latest = max(skip_ci_events(repo_url, token, pr_number), key=lambda e: (e["created_at"], e["id"]), default=None)
    return latest if latest and latest["event"] == "labeled" else None


def label_actor_has_write_access(repo_url: str, token: str, event: dict) -> bool:
    login = (event.get("actor") or {}).get("login")
    if not login:
        return False
    permission = read_api(f"{repo_url}/collaborators/{quote(login, safe='')}/permission", token)
    return permission["role_name"] in ("write", "maintain", "admin") or permission["permission"] in ("write", "admin")


def touches_github(path: str) -> bool:
    path = path.casefold()
    return path == ".github" or path.startswith(".github/")


def inspect_pr_paths(pr_url: str, token: str, pr: dict) -> tuple[dict, bool]:
    """Return the current PR and whether its verified diff touches .github."""
    changed_files = pr["changed_files"]
    if not 0 < changed_files < FILE_LIST_LIMIT:
        raise ValueError("The complete PR file list is unavailable.")

    seen = set()
    protected_paths = False
    for page in range(1, (changed_files - 1) // PAGE_SIZE + 2):
        for change in read_api(f"{pr_url}/files?per_page={PAGE_SIZE}&page={page}", token):
            filename = change["filename"]
            previous = change.get("previous_filename")
            if change["status"] == "renamed" and not previous:
                raise ValueError("A renamed file is missing its previous path.")
            if touches_github(filename) or (previous and touches_github(previous)):
                protected_paths = True
            seen.add(filename)

    current = read_api(pr_url, token)
    if (
        current["head"]["sha"] != pr["head"]["sha"]
        or current["base"]["sha"] != pr["base"]["sha"]
        or current["base"]["ref"] != pr["base"]["ref"]
        or current["changed_files"] != changed_files
        or len(seen) != changed_files
    ):
        raise ValueError("The PR changed or its complete file list is unavailable.")
    return current, protected_paths


def check_waiver(*, confirm: bool = False) -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    expected_head = event["pull_request"]["head"]["sha"]
    pr_number = event["number"]
    repo_url = f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}"
    pr_url = f"{repo_url}/pulls/{pr_number}"
    token = os.environ["GH_TOKEN"]
    try:
        pr = read_api(pr_url, token)
        if not (
            pr["state"] == "open"
            and pr["head"]["sha"] == expected_head
            and "[revert]" in pr["title"].casefold()
            and skip_ci_label(pr)
        ):
            raise ValueError("The PR no longer meets the waiver conditions.")

        label_event = active_label_event(repo_url, token, pr_number)
        if not label_event or not label_actor_has_write_access(repo_url, token, label_event):
            raise ValueError("The skip_ci label must be applied by a user with write access or higher.")
        if event["action"] == "labeled" and event["label"]["name"].casefold() == "skip_ci":
            if (label_event.get("actor") or {}).get("id") != event["sender"]["id"]:
                raise ValueError("The current label does not match the triggering skip_ci event.")

        if confirm:
            if (
                pr["base"]["sha"] != os.environ["REVERT_VERIFIED_BASE_SHA"]
                or str(label_event["id"]) != os.environ["REVERT_VERIFIED_LABEL_EVENT"]
            ):
                raise ValueError("The PR base or the verified skip_ci label has changed.")
            return 0

        current, protected_paths = inspect_pr_paths(pr_url, token, pr)
        if protected_paths:
            raise ValueError("PRs touching .github must run regular CI.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        if confirm:
            print(f"::error::Could not confirm the CI waiver ({error}).")
            return 1
        print(f"::notice::Could not waive CI ({error}). Running regular CI.")
    else:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            print(f"verified_base_sha={current['base']['sha']}", file=output)
            print(f"verified_label_event={label_event['id']}", file=output)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    raise SystemExit(check_waiver(confirm=parser.parse_args().confirm))
