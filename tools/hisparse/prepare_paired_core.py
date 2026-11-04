# SPDX-License-Identifier: Apache-2.0
"""Check out the merged-main Core tree used by the A5 optimization profile.

Create a new checkout; existing paths are never overwritten. No private merge
commit, workstation path, binary, or service credential is needed.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

MAIN = "d3fb7166e0ac9db82158129face12ae28209757a"
TREE = "d7eb5822d87cc449f1377cf59bf499116832bb9f"


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
    git("fetch", "--depth=1", "--no-tags", "origin", MAIN)
    git("checkout", "--detach", MAIN)
    actual = git("rev-parse", "HEAD^{tree}")
    if actual != TREE:
        raise RuntimeError(f"Paired Core tree differs: {actual}; expected {TREE}")
    source = (
        Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/hisparse/optimization_profile.py"
    )
    spec = importlib.util.spec_from_file_location("hisparse_source_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_core_sources(output / "vllm", "a5_c8_main")
    print(f"Verified paired Core tree {actual} at {output}")


if __name__ == "__main__":
    main()
