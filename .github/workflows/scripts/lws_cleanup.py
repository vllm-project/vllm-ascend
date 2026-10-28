# SPDX-License-Identifier: Apache-2.0
"""Clean up the current LWS task or completed runs in one hardware cluster.

Requires Python and kubectl. Only independent cleanup reads GITHUB_TOKEN;
local cleanup does not call GitHub. Kubernetes credentials come from kubectl.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

LABEL_PREFIX = "ci.vllm.ai/"
MANAGED_BY = "lws-cleanup"
LWS_LABEL = "leaderworkerset.sigs.k8s.io/name"
REQUEST_TIMEOUT_SECONDS = 15
PROCESS_TIMEOUT_SECONDS = 20
POLL_SECONDS = 5
DELETE_WAIT_SECONDS = 300
# Delete controllers before their pods so missing pods are not recreated.
RESOURCE_PATHS = {
    "LeaderWorkerSet": ("/apis/leaderworkerset.x-k8s.io/v1", "leaderworkersets"),
    "StatefulSet": ("/apis/apps/v1", "statefulsets"),
    "Service": ("/api/v1", "services"),
    "Pod": ("/api/v1", "pods"),
}


class CleanupError(RuntimeError):
    pass


def labels(resource: dict) -> dict:
    if resource["kind"] == "StatefulSet":
        # LWS copies Pod labels into this template, not StatefulSet metadata.
        return resource["spec"]["template"].get("metadata", {}).get("labels", {})
    return resource["metadata"].get("labels", {})


def identity(resource: dict) -> tuple[str, str, str]:
    ownership = labels(resource)
    return (
        ownership[f"{LABEL_PREFIX}run-id"],
        ownership[f"{LABEL_PREFIX}run-attempt"],
        ownership[f"{LABEL_PREFIX}task-id"],
    )


class GitHub:
    def __init__(self, repository: str, token: str):
        self.repository = repository
        self.token = token

    def run_completed(self, run_id: str, attempt: str) -> bool:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repository}/actions/runs/{run_id}/attempts/{attempt}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            run = json.load(response)

        return run["status"] == "completed"


class Kubernetes:
    def __init__(self, namespace: str):
        self.namespace = namespace

    def command(self, *arguments: str, body: dict | None = None) -> str:
        result = subprocess.run(
            ["kubectl", f"--request-timeout={REQUEST_TIMEOUT_SECONDS}s", "-n", self.namespace, *arguments],
            input=json.dumps(body) if body is not None else None,
            text=True,
            capture_output=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode:
            raise CleanupError(f"kubectl failed: {result.stderr.strip()}")
        return result.stdout

    def resources(self, selector: str, *, include_children: bool = False) -> list[dict]:
        kinds = "leaderworkerset,service,pod" if include_children else "leaderworkerset,service"
        resources = json.loads(self.command("get", kinds, "-l", selector, "-o", "json"))["items"]
        if include_children:
            statefulsets = json.loads(self.command("get", "statefulsets", "-l", LWS_LABEL, "-o", "json"))["items"]
            expected = dict(part.split("=", 1) for part in selector.split(","))
            resources.extend(s for s in statefulsets if all(labels(s).get(k) == v for k, v in expected.items()))
        return resources

    def delete(self, resource: dict) -> None:
        metadata = resource["metadata"]
        api, plural = RESOURCE_PATHS[resource["kind"]]
        path = f"{api}/namespaces/{self.namespace}/{plural}/{metadata['name']}"
        options = {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": metadata["uid"]},
            # LWS v0.6.3 can recreate children during foreground deletion.
            "propagationPolicy": "Background",
        }

        try:
            self.command("delete", "--raw", path, "-f", "-", body=options)
        except CleanupError as error:
            # Another cleanup or Kubernetes GC may already have removed it.
            if "(NotFound)" not in str(error):
                raise

    def wait_deleted(self, selector: str, targets: set[tuple[str, str, str]], seconds: int) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = [r for r in self.resources(selector, include_children=True) if identity(r) in targets]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                names = ", ".join(f"{r['kind']}/{r['metadata']['name']}" for r in remaining)
                raise CleanupError(f"Resources still exist: {names}; rerun cleanup or wait for the next hardware sweep")
            time.sleep(POLL_SECONDS)


def managed_selector(repository_id: str) -> str:
    return f"{LABEL_PREFIX}managed-by={MANAGED_BY},{LABEL_PREFIX}repository-id={repository_id}"


def delete_resources(cluster: Kubernetes, resources: list[dict], *, dry_run: bool) -> bool:
    """Submit controller deletions first, continuing if an individual request fails."""
    succeeded = True
    for resource in sorted(resources, key=lambda r: list(RESOURCE_PATHS).index(r["kind"])):
        name = f"{resource['kind']}/{resource['metadata']['name']}"
        print(f"{'Would delete' if dry_run else 'Submitting deletion for'} {name}")
        if dry_run:
            continue

        try:
            cluster.delete(resource)
        except (CleanupError, KeyError, OSError, subprocess.TimeoutExpired) as error:
            print(f"::error::Could not delete {name}: {error}", file=sys.stderr)
            succeeded = False

    return succeeded


def cleanup_current_task(args, cluster: Kubernetes) -> int:
    selector = (
        f"{managed_selector(args.repository_id)}"
        f",{LABEL_PREFIX}run-id={args.run_id},{LABEL_PREFIX}run-attempt={args.attempt}"
        f",{LABEL_PREFIX}task-id={args.task_id}"
    )
    resources = cluster.resources(selector)
    if not delete_resources(cluster, resources, dry_run=args.dry_run):
        return 1

    if not args.dry_run:
        if args.wait_seconds:
            targets = {(args.run_id, args.attempt, args.task_id)}
            cluster.wait_deleted(selector, targets, args.wait_seconds)
            print("Verified absence of resources for the current task")
        else:
            print("Deletion requests submitted; Kubernetes continues resource termination in the background")
    return 0


def sweep_completed_runs(args, github: GitHub, cluster: Kubernetes) -> int:
    selector = managed_selector(args.repository_id)

    # Group the hardware's resources by run/attempt; query GitHub once per group.
    groups: dict[tuple[str, str], list[dict]] = {}
    failed = False
    for resource in cluster.resources(selector, include_children=True):
        try:
            run_id, attempt, _ = identity(resource)
            groups.setdefault((run_id, attempt), []).append(resource)
        except (KeyError, TypeError, ValueError) as error:
            print(f"::error::Invalid ownership for {resource['metadata']['name']}: {error}", file=sys.stderr)
            failed = True

    targets: set[tuple[str, str, str]] = set()
    for (run_id, attempt), resources in groups.items():
        try:
            if not github.run_completed(run_id, attempt):
                print(f"Keeping run {run_id}/{attempt}: not completed")
                continue
        except (CleanupError, KeyError, TypeError, ValueError, OSError) as error:
            print(f"::error::Cannot verify run {run_id}/{attempt}: {error}", file=sys.stderr)
            failed = True
            continue

        print(f"Run {run_id}/{attempt} has completed")
        if not delete_resources(cluster, resources, dry_run=args.dry_run):
            failed = True
        elif not args.dry_run:
            targets.update(identity(resource) for resource in resources)

    if not args.dry_run and not failed:
        if targets and args.wait_seconds:
            cluster.wait_deleted(selector, targets, args.wait_seconds)
            print("Verified absence of resources for the completed runs")
        elif not targets:
            print("No completed-run resources to reclaim")
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cleanup", "sweep"))
    parser.add_argument("--repository", help="GitHub owner/repository (required for sweep)")
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--namespace", default="vllm-project")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt")
    parser.add_argument("--task-id")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=DELETE_WAIT_SECONDS,
        help="Wait for resource deletion in either mode; 0 disables waiting (default: %(default)s seconds)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "sweep" and not args.repository:
        parser.error("sweep requires --repository")
    if args.command == "cleanup" and not (args.run_id and args.attempt and args.task_id):
        parser.error("cleanup requires run-id, attempt and task-id")
    if args.wait_seconds < 0:
        parser.error("wait-seconds must not be negative")

    try:
        cluster = Kubernetes(args.namespace)
        if args.command == "cleanup":
            return cleanup_current_task(args, cluster)

        github = GitHub(args.repository, os.environ.get("GITHUB_TOKEN", ""))
        return sweep_completed_runs(args, github, cluster)
    except (CleanupError, KeyError, TypeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"::error::LWS cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
