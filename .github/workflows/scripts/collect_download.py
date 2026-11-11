"""Validate and stage requested CI files or directories for GitHub artifacts."""

import argparse
import json
import os
import tarfile
from pathlib import Path

# Artifacts are visible to PR readers, so exclude system and credential roots.
PROTECTED_ROOTS = (
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/run"),
    Path("/etc"),
    Path("/root/.ssh"),
    Path("/root/.config"),
    Path("/root/.aws"),
    Path("/root/.kube"),
    Path("/tmp/kubeconfig"),
)
TEST_REPOSITORY = Path("/vllm-workspace/vllm-ascend")


def validate_path(raw_path: str, *, require_exists: bool = True) -> Path:
    if not raw_path or any(ord(char) < 32 for char in raw_path):
        raise ValueError("--download path contains a control character")
    path = Path(raw_path)
    if ".." in path.parts:
        raise ValueError("--download path must not contain '..'")
    if not path.is_absolute():
        path = TEST_REPOSITORY / path
    path = Path(os.path.normpath(path))
    if path == Path("/"):
        raise ValueError("--download cannot select the filesystem root")
    if any(part in {".git", ".ssh", ".aws", ".kube"} for part in path.parts):
        raise ValueError("--download cannot select a credential directory")
    if path.name == ".env" or path.name.endswith((".pem", ".key")):
        raise ValueError("--download cannot select a credential file")
    for protected in PROTECTED_ROOTS:
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f"--download cannot include protected path: {protected}")
    if require_exists and not (path.is_dir() or path.is_file()):
        raise ValueError(f"--download file or directory does not exist: {path}")
    if require_exists and (path.is_symlink() or any(parent.is_symlink() for parent in path.parents)):
        raise ValueError("--download path must not contain symlinks")
    return path


class _DownloadObject(list[tuple[str, object]]):
    """Keep every JSON object entry, including repeated node keys."""


def parse_request(raw: str) -> dict[str, list[str]]:
    """Parse paths into a node-to-paths mapping."""
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
    if raw.startswith(("{", "[", '"')):
        parsed = json.loads(raw, object_pairs_hook=_DownloadObject)
        if isinstance(parsed, str):
            paths = {"node0": [parsed]}
        elif isinstance(parsed, _DownloadObject):
            paths = {}
            for node, value in parsed:
                paths.setdefault(node, []).append(value)
        elif isinstance(parsed, list):
            paths = {}
            for item in parsed:
                if not isinstance(item, _DownloadObject) or len(item) != 1:
                    raise ValueError("each download list entry must map one node to one path")
                node, value = item[0]
                paths.setdefault(node, []).append(value)
        else:
            raise ValueError("--download must be a path or node mapping")
    else:
        paths = {"node0": [raw]}
    if not paths:
        raise ValueError("--download node mapping must not be empty")
    for node, node_paths in paths.items():
        if not isinstance(node, str) or not node.startswith("node") or not node[4:].isdigit():
            raise ValueError("--download node keys must be node0, node1, ...")
        if node != f"node{int(node[4:])}":
            raise ValueError("--download node numbers must not have leading zeros")
        if not node_paths:
            raise ValueError("--download node path lists must not be empty")
        for path in node_paths:
            if not isinstance(path, str):
                raise ValueError("--download node paths must be strings")
            validate_path(path, require_exists=False)
    return paths


def archive_path(sources: list[Path], destination: Path) -> None:
    """Compress files without following links or including common credentials."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for index, source in enumerate(sources, start=1):
            # Keep the existing layout for one path; separate multiple paths
            # so files from different directories cannot overwrite each other.
            prefix = Path(f"path-{index}") if len(sources) > 1 else Path()
            if source.is_file():
                archive.add(source, arcname=prefix / source.name, recursive=False)
                continue
            for root, dirs, files in os.walk(source, followlinks=False):
                root_path = Path(root)
                dirs[:] = [
                    name
                    for name in dirs
                    if name not in {".git", ".ssh", ".aws", ".kube"} and not (root_path / name).is_symlink()
                ]
                for name in files:
                    item = root_path / name
                    if (
                        item == destination
                        or name == ".env"
                        or name.endswith((".pem", ".key"))
                        or item.is_symlink()
                        or not item.is_file()
                    ):
                        continue
                    archive.add(item, arcname=prefix / item.relative_to(source), recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--node", default="node0")
    parser.add_argument("--node-count", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--path-for-node", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    paths = parse_request(args.request)
    if args.node_count is not None:
        if args.node_count < 1 or any(int(node[4:]) >= args.node_count for node in paths):
            parser.error("--download references a node outside this job")
    if args.list:
        for node, node_paths in sorted(paths.items(), key=lambda item: int(item[0][4:])):
            resolved = [str(validate_path(path, require_exists=False)) for path in node_paths]
            print(f"{node}\t{json.dumps(resolved, separators=(',', ':'))}")
        return
    if args.path_for_node:
        if args.node in paths:
            resolved = [str(validate_path(path)) for path in paths[args.node]]
            print(json.dumps(resolved, separators=(",", ":")))
        return
    if args.node not in paths:
        parser.error(f"--download does not specify {args.node}")
    sources = [validate_path(path, require_exists=not args.check_only) for path in paths[args.node]]
    if not args.check_only:
        if args.destination is None:
            parser.error("destination is required unless --check-only is used")
        archive_path(sources, Path(args.destination))


if __name__ == "__main__":
    main()
