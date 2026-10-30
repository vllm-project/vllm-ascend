# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression checks for Triton module relocation."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TRITON_PREFIX = "vllm_ascend.ops.triton"


def test_triton_import_targets_exist():
    missing = []
    for source_root in ("vllm_ascend", "tests"):
        for source in (_ROOT / source_root).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.level == 0 and node.module else []
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module != _TRITON_PREFIX and not module.startswith(_TRITON_PREFIX + "."):
                        continue
                    target = _ROOT.joinpath(*module.split("."))
                    if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                        missing.append(f"{source.relative_to(_ROOT)}:{node.lineno}: {module}")
    assert not missing, "Missing relocated Triton imports:\n" + "\n".join(missing)
