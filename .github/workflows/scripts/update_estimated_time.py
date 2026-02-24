#!/usr/bin/env python3
"""
Aggregate per-test timing data from recent merged PRs and update
estimated_time in config.yaml.

Usage:
    python3 update_estimated_time.py \\
        --repo owner/repo \\
        --num-prs 5 \\
        --config .github/workflows/scripts/config.yaml

Requires:
    - gh CLI authenticated with read access to Actions artifacts
    - pyyaml
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from statistics import median


def _gh(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["gh"] + args, capture_output=True, text=True, check=True, **kwargs)


def get_merged_pr_shas(repo: str, n: int, required_labels: list[str] | None = None) -> list[str]:
    """
    Return commit SHAs (merge commits) of the last n merged PRs that carry
    ALL labels in required_labels.  When required_labels is None or empty
    every merged PR qualifies.

    We fetch up to n * 10 PRs to ensure we can find n matches even if most
    PRs do not carry the required labels.
    """
    fetch_limit = n * 10 if required_labels else n
    result = _gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--limit",
            str(fetch_limit),
            "--json",
            "mergeCommit,labels,number,title",
        ]
    )
    prs = json.loads(result.stdout)

    label_set = set(required_labels) if required_labels else set()
    shas: list[str] = []
    for pr in prs:
        if not pr.get("mergeCommit"):
            continue
        if label_set:
            pr_labels = {lbl["name"] for lbl in pr.get("labels", [])}
            if not label_set.issubset(pr_labels):
                # Not all required labels present – skip
                continue
        shas.append(pr["mergeCommit"]["oid"])
        if len(shas) >= n:
            break

    return shas


def get_successful_run_ids_for_sha(repo: str, sha: str) -> list[int]:
    """Return IDs of successful workflow runs triggered by the given commit SHA."""
    result = _gh(
        [
            "api",
            f"repos/{repo}/actions/runs",
            "--field",
            f"head_sha={sha}",
            "--field",
            "status=completed",
            "--field",
            "conclusion=success",
            "--jq",
            "[.workflow_runs[].id]",
        ]
    )
    return json.loads(result.stdout)


def download_timing_artifacts(repo: str, run_id: int, dest_dir: Path) -> list[Path]:
    """
    Download all artifacts whose name starts with 'timing-data-' from a run.
    Returns paths to the downloaded JSON files.
    """
    try:
        result = _gh(
            [
                "api",
                f"repos/{repo}/actions/runs/{run_id}/artifacts",
                "--jq",
                '[.artifacts[] | select(.name | startswith("timing-data-")) | .name]',
            ]
        )
        artifact_names: list[str] = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Could not list artifacts for run {run_id}: {e.stderr.strip()}")
        return []

    downloaded: list[Path] = []
    for name in artifact_names:
        out_dir = dest_dir / str(run_id) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _gh(
                [
                    "run",
                    "download",
                    str(run_id),
                    "--repo",
                    repo,
                    "--name",
                    name,
                    "--dir",
                    str(out_dir),
                ]
            )
            downloaded.extend(out_dir.rglob("*.json"))
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Could not download artifact '{name}' from run {run_id}: {e.stderr.strip()}")

    return downloaded


def load_timings_from_files(json_files: list[Path]) -> dict[str, list[float]]:
    """
    Parse all JSON timing files and aggregate elapsed times per test name.
    Only includes tests that actually ran (elapsed > 0).
    """
    timings: dict[str, list[float]] = {}
    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
            for test in data.get("tests", []):
                name = test.get("name", "")
                elapsed = test.get("elapsed", 0)
                if name and elapsed > 0:
                    timings.setdefault(name, []).append(elapsed)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  Warning: Failed to parse {path}: {e}")
    return timings


def update_config_inplace(config_path: str, updates: dict[str, int]) -> int:
    """
    Update estimated_time values in config.yaml while preserving all formatting,
    comments, and ordering.  Uses a line-by-line state machine to avoid re-serializing
    the YAML (which would lose comments and style).

    Returns the number of entries updated.
    """
    with open(config_path) as f:
        lines = f.readlines()

    updated = 0
    current_name: str | None = None
    result: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Track the most recently seen test name
        if stripped.startswith("name:"):
            current_name = stripped.split("name:", 1)[1].strip()

        # Replace estimated_time for the current test entry if we have a new value
        elif stripped.startswith("estimated_time:") and current_name in updates:
            try:
                old_time = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                result.append(line)
                continue
            new_time = updates[current_name]
            if old_time != new_time:
                line = line.replace(f"estimated_time: {old_time}", f"estimated_time: {new_time}")
                print(f"  {current_name}: {old_time}s -> {new_time}s")
                updated += 1

        result.append(line)

    with open(config_path, "w") as f:
        f.writelines(result)

    return updated


def compute_updates(
    timings: dict[str, list[float]],
    min_samples: int,
    buffer_ratio: float,
) -> dict[str, int]:
    """
    For each test that has enough samples, compute the new estimated_time as:
        round(median(elapsed_times) * buffer_ratio / 10) * 10   (nearest 10 s)
    Returns {test_name: new_estimated_time}.
    """
    updates: dict[str, int] = {}
    for name, elapsed_list in timings.items():
        if len(elapsed_list) < min_samples:
            continue
        new_time = int(round(median(elapsed_list) * buffer_ratio / 10) * 10)
        new_time = max(new_time, 10)  # floor at 10 s
        updates[name] = new_time
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Update estimated_time in config.yaml from recent CI timing data")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in owner/name format (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--num-prs",
        type=int,
        default=5,
        help="Number of recent merged PRs to include (default: 5)",
    )
    parser.add_argument(
        "--config",
        default=".github/workflows/scripts/config.yaml",
        help="Path to config.yaml (default: .github/workflows/scripts/config.yaml)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum number of timing samples required before updating a value (default: 2)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=1.1,
        help="Multiplier applied to median to add a safety buffer (default: 1.1 = 10%%)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["ready", "ready-for-test"],
        metavar="LABEL",
        help=(
            "Only consider merged PRs that carry ALL of these labels simultaneously. "
            "Full CI runs only happen on PRs with both 'ready' and 'ready-for-test'. "
            "(default: ready ready-for-test)"
        ),
    )
    args = parser.parse_args()

    if not args.repo:
        parser.error("--repo is required (or set the GITHUB_REPOSITORY env variable)")

    print(f"Repository : {args.repo}")
    print(f"PRs to scan: {args.num_prs}")
    print(f"Labels     : {args.labels}")
    print(f"Config file: {args.config}")
    print()

    # ── 1. Find merged PR commit SHAs ───────────────────────────────────────
    print(f"Fetching last {args.num_prs} merged PRs with labels {args.labels}...")
    try:
        shas = get_merged_pr_shas(args.repo, args.num_prs, required_labels=args.labels)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching PRs: {e.stderr.strip()}")
        raise SystemExit(1)

    if not shas:
        print("No merged PRs found. Exiting without changes.")
        return

    print(f"Found {len(shas)} SHAs: {[s[:8] for s in shas]}\n")

    # ── 2. Download timing artifacts ────────────────────────────────────────
    all_json_files: list[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for sha in shas:
            print(f"SHA {sha[:8]}: looking for successful workflow runs...")
            try:
                run_ids = get_successful_run_ids_for_sha(args.repo, sha)
            except subprocess.CalledProcessError as e:
                print(f"  Warning: {e.stderr.strip()}")
                continue

            if not run_ids:
                print("  No successful runs found.")
                continue

            print(f"  Found {len(run_ids)} run(s): {run_ids}")
            for run_id in run_ids:
                files = download_timing_artifacts(args.repo, run_id, Path(tmpdir))
                if files:
                    print(f"  Run {run_id}: downloaded {len(files)} timing JSON file(s)")
                    all_json_files.extend(files)

        if not all_json_files:
            print(
                "\nNo timing-data artifacts found in the selected PRs. "
                "Make sure CI has run at least once with the timing artifact upload steps."
            )
            return

        # ── 3. Aggregate timings ─────────────────────────────────────────────
        print(f"\nAggregating {len(all_json_files)} timing file(s)...")
        timings = load_timings_from_files(all_json_files)
        print(f"Collected timing data for {len(timings)} unique test file(s).")

        # ── 4. Compute new estimated_time values ─────────────────────────────
        updates = compute_updates(timings, args.min_samples, args.buffer)
        if not updates:
            print(f"\nNo test has enough samples (min={args.min_samples}) to update. Run more CI jobs first.")
            return

        # ── 5. Update config.yaml ─────────────────────────────────────────────
        print(
            f"\nUpdating {args.config} ({len(updates)} candidate(s), buffer={args.buffer}x, rounded to nearest 10 s):"
        )
        n_updated = update_config_inplace(args.config, updates)
        print(f"\nDone. {n_updated} estimated_time value(s) changed.")


if __name__ == "__main__":
    main()
