# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the public-parent Core tree used by the A5 optimization profile.

Create a new checkout; existing paths are never overwritten. No private merge
commit, workstation path, binary, or service credential is needed.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

MAIN = "43111fd1f0c3135ff671e25e2e85238aa05743d0"
PR = "5b8e00878206c228c6ec888fbd6494d6ffe07a03"
TREE = "314b378fa399edede28784fdf52d97964b9194a2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", default="https://github.com/vllm-project/vllm.git")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=output, text=True).strip()

    git("init")
    git("remote", "add", "origin", args.remote)
    git("fetch", "--no-tags", "origin", MAIN, PR)
    git("checkout", "--detach", MAIN)
    # The local merge's author/time do not affect its tree. Verify content
    # independently rather than relying on a private merge commit's SHA.
    git(
        "-c",
        "user.name=HiSparse local integration",
        "-c",
        "user.email=hisparse@localhost",
        "merge",
        "--no-ff",
        "-m",
        "merge(hisparse): reconstruct paired PR62592 source",
        PR,
    )
    actual = git("rev-parse", "HEAD^{tree}")
    if actual != TREE:
        raise RuntimeError(f"Paired Core tree differs: {actual}; expected {TREE}")
    source = (
        Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/hisparse/optimization_profile.py"
    )
    spec = importlib.util.spec_from_file_location("hisparse_source_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_core_sources(output / "vllm")
    print(f"Verified paired Core tree {actual} at {output}")


if __name__ == "__main__":
    main()
