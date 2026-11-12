"""Assemble published image metadata from one GitHub Actions run."""

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

COMMIT_HEX_LENGTH = 40
DIGEST_HEX_LENGTH = 64
LOWERCASE_HEX = frozenset("0123456789abcdef")
BUILD_TYPES = ("base", "nightly", "tag")
ACTIVE_DAYS = 7


def is_lowercase_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in LOWERCASE_HEX for char in value)


def expected_targets(raw_targets: list[dict]) -> set[tuple[str, str]]:
    """Normalize the two top-level workflow matrix shapes."""
    targets = set()
    for item in raw_targets:
        if "target" in item:
            hardware = item["target"]
            os_name = item["os"]
        else:
            suffix = item["suffix"]
            os_name = "openeuler" if "openeuler" in suffix else "ubuntu"
            hardware = suffix.replace("-openeuler", "") or "a2"
            if hardware == "openeuler":
                hardware = "a2"
        target = (hardware.lower(), os_name.lower())
        if target in targets:
            raise ValueError(f"Duplicate expected target: {target}")
        targets.add(target)
    if not targets:
        raise ValueError("The build matrix has no targets")
    return targets


def report_attempt(path: Path, artifact_dir: Path) -> int:
    """Read the attempt from the downloaded artifact's directory name."""
    artifact_name = path.relative_to(artifact_dir).parts[0]
    prefix = "build-info-image-"
    if not artifact_name.startswith(prefix):
        raise ValueError(f"Unexpected artifact directory: {artifact_name}")
    attempt_text = artifact_name.removeprefix(prefix).split("-", 1)[0]
    if not attempt_text.isdecimal() or int(attempt_text) < 1:
        raise ValueError(f"Invalid attempt in artifact name: {artifact_name}")
    return int(attempt_text)


def collect(
    artifact_dir: Path,
    targets: set[tuple[str, str]],
    expected_types: set[str],
    branch: str,
    run_id: int,
    run_attempt: int,
) -> dict:
    """Validate individual image reports and group them by build type."""
    if not expected_types or not expected_types.issubset(BUILD_TYPES):
        raise ValueError(f"Invalid expected build types: {expected_types}")
    if not branch:
        raise ValueError("Missing source branch or ref")

    reports = sorted(artifact_dir.rglob("*.json"))
    if not reports:
        raise ValueError("No published image reports were downloaded")

    latest_reports: dict[tuple[str, str, str], tuple[int, list[dict]]] = {}
    for path in reports:
        attempt = report_attempt(path, artifact_dir)
        if attempt > run_attempt:
            raise ValueError(f"Report from a future attempt: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        build_type = report["type"]
        hardware = report["hardware"]
        os_name = report["os"]
        commits = report["commits"]
        tag = report["tag"]
        digest = report["digest"]
        repositories = report["repositories"]
        target = (hardware, os_name)

        if build_type not in expected_types or target not in targets:
            raise ValueError(f"Unexpected image report in {path}: {build_type}, {target}")
        if not isinstance(commits, dict) or set(commits) != {"arm64", "amd64"}:
            raise ValueError(f"Invalid architecture commits in {path}")
        if any(
            not isinstance(commit, str) or not is_lowercase_hex(commit, COMMIT_HEX_LENGTH)
            for commit in commits.values()
        ):
            raise ValueError(f"Invalid commit in {path}")
        if not digest.startswith("sha256:") or not is_lowercase_hex(digest.removeprefix("sha256:"), DIGEST_HEX_LENGTH):
            raise ValueError(f"Invalid digest in {path}: {digest}")
        if not tag or not isinstance(repositories, list) or not repositories:
            raise ValueError(f"Missing tag or repositories in {path}")
        if any(not isinstance(repo, str) or not repo for repo in repositories):
            raise ValueError(f"Invalid repository in {path}")

        target_key = (build_type, hardware, os_name)
        previous = latest_reports.get(target_key)
        if previous is None or attempt > previous[0]:
            latest_reports[target_key] = (attempt, [report])
        elif attempt == previous[0]:
            previous[1].append(report)

    required = {(build_type, hardware, os_name) for build_type in expected_types for hardware, os_name in targets}
    missing = required - latest_reports.keys()
    if missing:
        raise ValueError(f"Missing published image reports: {sorted(missing)}")

    groups: dict[str, dict] = {}
    for target_key, (_, selected_reports) in latest_reports.items():
        build_type = target_key[0]
        observed_tags: set[str] = set()
        for report in selected_reports:
            tag = report["tag"]
            if tag in observed_tags:
                raise ValueError(f"Duplicate image report: {target_key}, {tag}")
            observed_tags.add(tag)
            group = groups.setdefault(build_type, {"images": []})
            group["images"].append(
                {
                    "hardware": report["hardware"],
                    "os": report["os"],
                    "commits": report["commits"],
                    "tag": tag,
                    "digest": report["digest"],
                    "repositories": report["repositories"],
                }
            )

    result = {
        "run_id": run_id,
        "build_type": [kind for kind in BUILD_TYPES if kind in groups],
        "branch": branch,
    }
    for kind in BUILD_TYPES:
        if kind in groups:
            groups[kind]["images"].sort(key=lambda image: (image["hardware"], image["os"], image["tag"]))
            result[kind] = groups[kind]
    return result


def assemble(args: argparse.Namespace) -> None:
    """Write one validated JSON file for the completed workflow run."""
    raw_targets = json.loads(args.targets_json)
    if isinstance(raw_targets, dict):
        raw_targets = raw_targets["include"]
    expected_types = set(json.loads(args.expected_types_json))
    result = collect(
        args.artifact_dir,
        expected_targets(raw_targets),
        expected_types,
        args.branch,
        args.run_id,
        args.run_attempt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(os.environ.get(name) for name in ("OBS_BUCKET", "OBS_ENDPOINT", "OBS_REGION")):
        publish_to_obs(args.output)


def read_index(client, bucket: str, key: str, default: dict | None) -> dict | None:
    """Read an OBS index, treating only a missing object as an empty index."""
    response = client.getObject(bucket, key, loadStreamInMemory=True)
    if response.status == 404 and response.errorCode == "NoSuchKey":
        return default
    if response.status >= 300:
        raise RuntimeError(
            f"Failed to read obs://{bucket}/{key}: HTTP {response.status}, "
            f"{response.errorCode}: {response.errorMessage}"
        )
    value = json.loads(response.body.buffer)
    if not isinstance(value, dict):
        raise ValueError(f"OBS object is not a JSON object: {key}")
    return value


def upload_json(client, bucket: str, key: str, value: dict) -> None:
    """Upload a prepared UTF-8 JSON object."""
    with tempfile.TemporaryDirectory() as directory:
        local_path = Path(directory) / "build-info.json"
        local_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        response = client.putFile(bucket, key, str(local_path))
    if response.status >= 300:
        raise RuntimeError(
            f"Failed to publish obs://{bucket}/{key}: HTTP {response.status}, "
            f"{response.errorCode}: {response.errorMessage}"
        )
    print(f"Published obs://{bucket}/{key}")


def prepare_root_index(client, bucket: str, prefix: str, current_name: str, current_index: dict, now: datetime) -> dict:
    """Prepare the root index without writing anything to OBS."""
    root_key = f"{prefix}/index.json"
    root = read_index(client, bucket, root_key, None)
    if root is None:
        root = {"index_urls": [], "active_urls": {}}
    index_urls = root["index_urls"]
    active_urls = root["active_urls"]
    if not isinstance(index_urls, list) or not isinstance(active_urls, dict):
        raise ValueError("Invalid branch indexes in index.json")
    if any(
        not isinstance(name, str) or not name.startswith("index_") or not name.endswith(".json") for name in index_urls
    ):
        raise ValueError("Invalid branch index URL in index.json")
    if len(set(index_urls)) != len(index_urls):
        raise ValueError("Duplicate branch index URL in index.json")
    if any(name not in index_urls or not isinstance(updated_at, str) for name, updated_at in active_urls.items()):
        raise ValueError("Invalid active branch in index.json")

    for index_name, updated_at in list(active_urls.items()):
        try:
            last_change = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid updated_at in {index_name}") from error
        if last_change.tzinfo is None:
            raise ValueError(f"updated_at must have a timezone in {index_name}")
        if now - last_change > timedelta(days=ACTIVE_DAYS):
            del active_urls[index_name]

    if current_name not in index_urls:
        index_urls.append(current_name)
    active_urls[current_name] = current_index["updated_at"]
    root["updated_at"] = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    return root


def publish(client, bucket: str, prefix: str, run_file: Path) -> None:
    """Prepare all JSON objects, then publish details before their indexes."""
    build = json.loads(run_file.read_text(encoding="utf-8"))
    run_id = build["run_id"]
    branch = build["branch"]
    if not isinstance(run_id, int) or run_id < 1:
        raise ValueError("Invalid run_id in build information")
    if not isinstance(branch, str) or not branch or branch.startswith("/") or ".." in branch:
        raise ValueError("Invalid branch in build information")
    branch_slug = branch.replace("/", "-")
    index_name = f"index_{branch_slug}.json"
    branch_key = f"{prefix}/{index_name}"
    branch_index = read_index(client, bucket, branch_key, {"updated_at": "", "latest_order": 0, "info_urls": {}})
    info_urls = branch_index["info_urls"]
    latest_order = branch_index["latest_order"]
    if not isinstance(info_urls, dict):
        raise ValueError("Invalid build information index")
    if not isinstance(latest_order, int) or latest_order < 0:
        raise ValueError("Invalid latest_order in branch index")
    if any(not order.isdecimal() or int(order) < 1 for order in info_urls):
        raise ValueError("Invalid order in branch index")
    if any(int(order) > latest_order for order in info_urls):
        raise ValueError("Branch index order exceeds latest_order")

    order = str(latest_order + 1)
    published_at = datetime.now(timezone.utc)
    build_types = build["build_type"]
    if not isinstance(build_types, list) or not build_types or any(kind not in BUILD_TYPES for kind in build_types):
        raise ValueError("Invalid build_type in build information")
    timestamp = published_at.strftime("%Y-%m-%d-%H-%M")
    filename = f"{timestamp}-{branch_slug}-{'-'.join(build_types)}.json"
    detail_url = f"builds/{published_at.year}/{branch_slug}/{filename}"
    branch_index["latest_order"] = int(order)
    info_urls[order] = detail_url

    now = datetime.now(timezone.utc)
    branch_index["updated_at"] = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    root = prepare_root_index(client, bucket, prefix, index_name, branch_index, now)

    # All three objects are validated and prepared before the first OBS write.
    uploads = [(f"{prefix}/{detail_url}", build), (branch_key, branch_index), (f"{prefix}/index.json", root)]
    for key, value in uploads:
        upload_json(client, bucket, key, value)


def obs_storage():
    """Create the OBS client only for operations that use remote storage."""
    bucket = os.environ.get("OBS_BUCKET", "")
    endpoint = os.environ.get("OBS_ENDPOINT", "")
    region = os.environ.get("OBS_REGION", "")
    prefix = os.environ.get("OBS_PREFIX", "ci/build-info").strip("/")
    if not all(
        (bucket, endpoint, region, prefix, os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY"))
    ):
        raise ValueError("Build information OBS configuration is incomplete")

    from obs import ObsClient

    client = ObsClient(
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        server=endpoint,
    )
    return client, bucket, prefix


def publish_to_obs(run_file: Path) -> None:
    """Publish the completed run when OBS storage has been configured."""
    client, bucket, prefix = obs_storage()
    publish(client, bucket, prefix, run_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--targets-json", required=True)
    parser.add_argument("--expected-types-json", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args)


if __name__ == "__main__":
    main()
