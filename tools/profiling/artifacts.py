"""Build small, self-contained profiling indexes for GitHub Artifacts."""

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path

from tools.profiling import safe_name
from tools.profiling.storage import DEFAULT_BUCKET, DEFAULT_ENDPOINT, DEFAULT_REGION, obs_http_url


def artifact_key(prefix: str) -> str:
    """Keep run, attempt, scene, branch, and job identity in artifact names."""
    return safe_name(prefix.strip("/").removeprefix("nightly-profiling/").replace("/", "-"))


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _rank_label(name: str) -> str:
    fields = re.findall(r"(?:^|_)(dp|pp|tp|dcp|ep|rank)(\d+)(?=_|$)", Path(name).name)
    return " / ".join(f"{key.upper()}={value}" for key, value in fields) or name


def _downloads(manifest: dict) -> list[dict]:
    """Extract confirmed objects once for the documents, log, and shell script."""
    downloads = []
    endpoint = manifest["storage"]["endpoint"]
    for benchmark in manifest["benchmarks"]:
        artifact = benchmark["benchmark_artifact_name"]
        for target in benchmark["targets"]:
            group = f"{artifact} / {target['name']}"
            for stage, records in (("raw", [target]), ("parsed", target.get("ranks", []))):
                for record in records:
                    if record.get("obs_url"):
                        downloads.append(
                            {
                                "stage": stage,
                                "group": group,
                                "label": target["name"] if stage == "raw" else _rank_label(record["name"]),
                                "path": f"{stage}/{record['archive']}",
                                "url": obs_http_url(record["obs_url"], endpoint),
                            }
                        )
    return sorted(downloads, key=lambda item: _natural_key(item["path"]))


def _failures(manifest: dict) -> list[str]:
    failures = []
    for benchmark in manifest["benchmarks"]:
        name = benchmark["benchmark_artifact_name"]
        if benchmark["status"] != "success" and benchmark.get("reason"):
            failures.append(f"{name}: {benchmark['reason']}")
        for target in benchmark["targets"]:
            for record in (target, *target.get("ranks", [])):
                label = f"{name}/{target['name']}"
                if record is not target:
                    label += f"/{record['name']}"
                failures.extend(
                    f"{label}: {key}: {value}" for key, value in record.items() if key.endswith("_error") and value
                )
    return failures


def _heading(title: str, separator: str = "=") -> str:
    line = separator * 60
    return f"{line}\n{title}\n{line}\n"


def _download_section(stage: str, downloads: list[dict], *, markdown: bool = False) -> str:
    selected = [item for item in downloads if item["stage"] == stage]
    title = f"{stage.upper()} — {len(selected)} files"
    lines = [f"# {title}" if markdown else _heading(title), ""]
    groups = dict.fromkeys(item["group"] for item in selected)
    for group in groups:
        items = [item for item in selected if item["group"] == group]
        title = f"{group} — {len(items)} files"
        lines.extend([f"## {title}" if markdown else _heading(title, "-"), ""])
        for index, item in enumerate(items, 1):
            if markdown:
                lines.extend([f"{index}. [{item['label']}]({item['url']})", ""])
            else:
                lines.extend([f"[{index}/{len(items)}] {item['label']}", item["url"], ""])
    if not selected:
        lines.extend(["No confirmed uploads available.", ""])
    return "\n".join(lines)


def _quick_download(manifest: dict, downloads: list[dict]) -> str:
    storage = manifest["storage"]
    source = f"obs://{storage['bucket']}/{storage['prefix'].strip('/')}"
    lines = [_heading("QUICK DOWNLOAD"), ""]
    if downloads:
        for stage in ("all", "raw", "parsed"):
            if stage != "all" and not any(item["stage"] == stage for item in downloads):
                continue
            suffix = "" if stage == "all" else stage + "/"
            lines.extend(
                [
                    f"Download {stage.upper()} (obsutil, up to 4 concurrent files):",
                    f"obsutil cp {shlex.quote(source + '/' + suffix)} ./profile-download/{suffix} -r -j=4",
                    "",
                ]
            )
        lines.extend(
            [
                "After extracting the index into ./profile_artifact (Bash + curl):",
                "bash ./profile_artifact/download.sh all",
                "Replace all with raw or parsed to download only that output.",
                "",
                "obsutil needs list/download access; HTTPS links use the object's existing read permissions.",
            ]
        )
    else:
        lines.append("No confirmed uploads available.")
    return "\n".join(lines) + "\n"


def _download_script(downloads: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'mode="${1:-all}"',
        'destination="${2:-./profile-download}"',
        'case "$mode" in',
        "  all|raw|parsed) ;;",
        '  *) echo "Usage: bash download.sh [all|raw|parsed] [destination]" >&2; exit 2 ;;',
        "esac",
        "",
        "download() {",
        '  local output="$destination/$1"',
        '  mkdir -p -- "$(dirname -- "$output")"',
        '  printf "Downloading %s\\n" "$1"',
        '  curl --fail --location --retry 3 --output "$output.part" "$2"',
        '  mv -- "$output.part" "$output"',
        "}",
        "",
    ]
    for stage in ("raw", "parsed"):
        lines.append(f'if [[ "$mode" == all || "$mode" == {stage} ]]; then')
        selected = [item for item in downloads if item["stage"] == stage]
        if selected:
            for item in selected:
                lines.append(f"  download {shlex.quote(item['path'])} {shlex.quote(item['url'])}")
        else:
            lines.append(f"  echo 'No confirmed {stage} uploads available.'")
        lines.extend(["fi", ""])
    return "\n".join(lines)


def _summary(manifest: dict, downloads: list[dict], *, fold_links: bool = False) -> str:
    overview = [
        _heading("PROFILING ARTIFACTS"),
        f"Status     : {manifest['status']}",
        f"Benchmarks : {len(manifest['benchmarks'])}",
        f"RAW        : {sum(item['stage'] == 'raw' for item in downloads)} files",
        f"PARSED     : {sum(item['stage'] == 'parsed' for item in downloads)} files",
    ]
    failures = _failures(manifest)
    if failures:
        overview.extend(["", _heading("FAILURES"), *failures])
    sections = ["\n".join(overview), _quick_download(manifest, downloads)]
    for stage in ("raw", "parsed"):
        if stage == "parsed" and manifest["requested"]["output"] == "raw":
            continue
        section = _download_section(stage, downloads)
        if fold_links:
            section = f"::group::{stage.upper()} download links\n{section}\n::endgroup::"
        sections.append(section)
    return "\n\n".join(sections) + "\n"


def write_artifact_bundle(manifest_path: Path, output: Path, prefix: str | None = None) -> None:
    """Publish only metadata and download instructions, never the raw trace tree."""
    manifest = json.loads(manifest_path.read_text())
    # A failed upload setup may not have recorded the destination yet.
    if "storage" not in manifest:
        manifest["storage"] = {
            "prefix": prefix,
            "bucket": DEFAULT_BUCKET,
            "endpoint": DEFAULT_ENDPOINT,
            "region": DEFAULT_REGION,
        }
    for benchmark in manifest["benchmarks"]:
        if not benchmark["targets"] or any(not target.get("obs_url") for target in benchmark["targets"]):
            benchmark["status"] = manifest["status"] = "partial"
        for target in benchmark["targets"]:
            if target.get("archive") and not target.get("obs_url"):
                target.setdefault("upload_error", "Raw archive was not uploaded")
    output.mkdir(parents=True, exist_ok=True)
    for benchmark in manifest["benchmarks"]:
        relative = benchmark["serve_manifest"]
        source, destination = manifest_path.parent / relative, output / relative
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    (output / "profile_manifest.json").write_text(json.dumps(manifest, indent=2))
    downloads = _downloads(manifest)
    for stage in ("raw", "parsed"):
        if stage == "parsed" and manifest["requested"]["output"] == "raw":
            continue
        directory = output / stage
        directory.mkdir(exist_ok=True)
        (directory / "downloads.md").write_text(_download_section(stage, downloads, markdown=True))
    (output / "summary.txt").write_text(_summary(manifest, downloads))
    (output / "download.sh").write_text(_download_script(downloads))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    identify = commands.add_parser("identify")
    identify.add_argument("--prefix", required=True)
    identify.add_argument("--github-output", type=Path, required=True)
    package = commands.add_parser("package")
    package.add_argument("--manifest", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--prefix")
    package.add_argument("--github-output", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summary":
        manifest = json.loads((args.root / "profile_manifest.json").read_text())
        print(_summary(manifest, _downloads(manifest), fold_links=True), flush=True)
        return
    if args.command == "package":
        write_artifact_bundle(args.manifest, args.output, args.prefix)
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(f"artifact_key={artifact_key(args.prefix)}\n")


if __name__ == "__main__":
    main()
