# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Validate provisioned measured arrays against the reviewed manifest."""

import argparse
import json
from pathlib import Path

from .config import CASES
from .gate import check_capture, load_golden


def verify(root):
    count = 0
    for case in CASES:
        manifest, _ = load_golden(case, artifact_root=root)
        for entry in manifest["cases"][case]:
            result = check_capture(root / entry["file"], entry, manifest, root=root)
            if not result["passed"]:
                raise ValueError(f"Baseline self-check failed: {entry['file']}")
            count += 1
    return {"baseline_id": manifest["baseline_id"], "verified_artifacts": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact_root)))
