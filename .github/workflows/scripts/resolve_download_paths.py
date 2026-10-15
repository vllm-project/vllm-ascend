"""Merge configured nightly log paths with a one-off --log-path request."""

import argparse
import json
from pathlib import Path

from collect_download import parse_request, validate_path


def resolve_paths(config_file: Path, size: int, request: str) -> str:
    if size < 1:
        raise ValueError("node count must be positive")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("log download defaults must be an object")
    configured = config.get(str(size), {})
    if not isinstance(configured, dict):
        raise ValueError(f"log download defaults for size {size} must be an object")

    paths: dict[str, list[str]] = {}
    for node, values in configured.items():
        if not isinstance(values, list) or not all(isinstance(path, str) for path in values):
            raise ValueError(f"configured paths for {node} must be a list of strings")
        paths[node] = values.copy()
    if request:
        for node, values in parse_request(request).items():
            paths.setdefault(node, []).extend(values)

    entries = []
    for node, values in paths.items():
        if not node.startswith("node") or not node[4:].isdigit() or node != f"node{int(node[4:])}":
            raise ValueError(f"invalid node name: {node}")
        if int(node[4:]) >= size:
            raise ValueError(f"{node} is outside a {size}-node job")
        for path in values:
            validate_path(path, require_exists=False)
            entries.append({node: path})
    return json.dumps(entries, separators=(",", ":")) if entries else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--request", default="")
    args = parser.parse_args()
    print(resolve_paths(args.config, args.size, args.request))


if __name__ == "__main__":
    main()
