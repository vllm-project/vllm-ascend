# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a portable GLM Triton comparison from two pinned Git revisions."""

import argparse
import ast
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
BASELINE = "ffc1c7a82bfef1d9b434da1206e08589387fdb28"
CANDIDATE = "faaa46690d751b22ce5ddd8673ca56a5fabcf776"
OP_DIR = "vllm_ascend/ops/triton/"
TEST_DIR = "tests/e2e/nightly/single_node/ops/singlecard_ops/triton/"
NORM_NAMES = frozenset(
    (
        "layer_norm_gated_fwd_kernel",
        "layer_norm_gated_fwd_kernel1",
        "layer_norm_gated_fwd",
        "rms_norm_gated",
    )
)


def read_source(repo, revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=repo).decode("utf-8")


def extract_norm(content):
    """Keep the four norm functions verbatim, including their decorators."""
    lines = content.splitlines(keepends=True)
    nodes = [node for node in ast.parse(content).body if isinstance(node, ast.FunctionDef) and node.name in NORM_NAMES]
    if {node.name for node in nodes} != NORM_NAMES:
        raise ValueError("Expected gated-norm functions are missing from the selected revision")
    result = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "import torch\nfrom vllm.triton_utils import tl, triton\n"
        "from vllm.utils.math_utils import cdiv, next_power_of_2\n\n"
    )
    for node in nodes:
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        result += "".join(lines[start - 1 : node.end_lineno]) + "\n\n"
    return result


def build(repo, output, revisions):
    output.mkdir(parents=True, exist_ok=False)
    origins = {}
    for role, revision in revisions.items():
        (output / role).mkdir()
        (output / role / "__init__.py").write_text("", encoding="utf-8")
        for filename in ("glm5_next_lightning_indexer.py", "glm5_next_kpool_tail_compress.py", "kda/kda.py"):
            content = read_source(repo, revision, OP_DIR + filename)
            target = output / role / Path(filename).name
            target.write_text(content, encoding="utf-8", newline="\n")
            origins[target.relative_to(output).as_posix()] = {"revision": revision, "path": OP_DIR + filename}
            if filename == "kda/kda.py":
                (output / role / "gated_norm.py").write_text(extract_norm(content), encoding="utf-8", newline="\n")
    for filename in ("test_glm5next_pool_key_indexer_triton.py", "test_glm5next_kpool_tail_triton.py"):
        content = read_source(repo, revisions["candidate"], TEST_DIR + filename)
        content = content.replace("from vllm_ascend.ops.triton import", "from candidate import")
        content = content.replace(
            "from vllm_ascend.ops.triton.glm5_next_kpool_tail_compress import",
            "from candidate.glm5_next_kpool_tail_compress import",
        )
        (output / filename).write_text(content, encoding="utf-8", newline="\n")
        origins[filename] = {
            "revision": revisions["candidate"],
            "path": TEST_DIR + filename,
            "adaptation": "import path only",
        }
    for name in ("run_a5.py", "run.sh", "README.md", "norm_checks.py", "check_score_regression.py"):
        (output / name).write_text((SOURCE_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "harness_revision": "r2-valid-score-rows",
        "revisions": revisions,
        "origins": origins,
        "sha256": {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = output.parent / (output.name + ".tar.gz")
    with tarfile.open(archive, "x:gz") as bundle:
        for path in files + [manifest_path]:
            bundle.add(path, arcname=(Path(output.name) / path.relative_to(output)).as_posix())
    print(f"Bundle: {output}\nArchive: {archive}\nSHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=SOURCE_ROOT.parents[2])
    parser.add_argument("--output", type=Path, required=True, help="New directory for generated sources and scripts")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=CANDIDATE)
    options = parser.parse_args()
    # Resolve immutable commits before creating any output. Shallow clones may need a fetch.
    revisions = {
        role: subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=options.repo, text=True
        ).strip()
        for role, revision in (("baseline", options.baseline), ("candidate", options.candidate))
    }
    build(options.repo.resolve(), options.output.resolve(), revisions)


if __name__ == "__main__":
    main()
