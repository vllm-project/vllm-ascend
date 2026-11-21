# SPDX-License-Identifier: Apache-2.0
"""Isolate a vendor-built experimental MSA extension from the default package.

Run after installing the wheel built with TORCH_EXTENSION_VENDOR=a4c4.
Only Python dispatcher names and the plugin entry-point group are changed.
"""

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import regex as re

_PACKAGE = "cann_ops_transformer_a4c4"
_BASE_PACKAGE = "cann_ops_transformer"


def _rewrite(path: Path, updated: str) -> bool:
    original = path.read_text(encoding="utf-8")
    if original == updated:
        return False
    backup = path.with_name(path.name + ".before-a4c4-isolation")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    return True


def isolate_namespace() -> list[str]:
    spec = importlib.util.find_spec(_PACKAGE)
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the wheel built with TORCH_EXTENSION_VENDOR=a4c4 first")
    package_root = Path(spec.origin).parent
    distribution = importlib.metadata.distribution(_PACKAGE)
    metadata_files = [
        path for path in distribution.files or () if str(path).replace("\\", "/").endswith(".dist-info/METADATA")
    ]
    if len(metadata_files) != 1:
        raise RuntimeError("Expected one wheel METADATA file for the experimental package")
    metadata_root = Path(distribution.locate_file(metadata_files[0])).parent
    if metadata_root.parent.resolve() != package_root.parent.resolve():
        raise RuntimeError("Experimental package and wheel metadata are in different installations")
    changed = []
    for path in package_root.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        updated = re.sub(r"\btorch\.ops\.cann_ops_transformer\b", "torch.ops." + _PACKAGE, original)
        for quote in ('"', "'"):
            updated = updated.replace(
                "Library(" + quote + _BASE_PACKAGE + quote + ",", "Library(" + quote + _PACKAGE + quote + ","
            )
            updated = updated.replace(quote + _BASE_PACKAGE + ".ops" + quote, quote + _PACKAGE + ".ops" + quote)
        if _rewrite(path, updated):
            changed.append(str(path))
    entry_points = metadata_root / "entry_points.txt"
    original = entry_points.read_text(encoding="utf-8")
    updated = original.replace("[" + _BASE_PACKAGE + ".ops]", "[" + _PACKAGE + ".ops]")
    if _rewrite(entry_points, updated):
        changed.append(str(entry_points))
    baseline = importlib.metadata.entry_points(group=_BASE_PACKAGE + ".ops")
    if any(entry.value.startswith(_PACKAGE + ".") for entry in baseline):
        raise RuntimeError("Experimental entry points still override the baseline package")
    isolated = importlib.metadata.entry_points(group=_PACKAGE + ".ops")
    if not any(entry.value.startswith(_PACKAGE + ".") for entry in isolated):
        raise RuntimeError("Experimental entry-point group was not registered")
    return changed


if __name__ == "__main__":
    print(json.dumps({"package": _PACKAGE, "changed_files": isolate_namespace()}, indent=2))
