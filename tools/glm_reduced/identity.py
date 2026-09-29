# SPDX-License-Identifier: Apache-2.0
"""Stable checkpoint identity shared by the builder and the gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_identity(model: str | Path) -> str:
    """Identity survives an identical rebuild in a different directory/time."""
    manifest = json.loads((Path(model) / "reduction_manifest.json").read_text(encoding="utf-8"))
    return digest(
        {
            "reduction": manifest["reduction"],
            "files": {entry["name"]: entry["sha256"] for entry in manifest["output"]["files"]},
            "tensor_sha256": manifest["output"]["tensor_sha256"],
        }
    )
