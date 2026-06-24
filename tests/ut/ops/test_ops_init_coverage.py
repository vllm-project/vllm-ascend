import ast
import os
from pathlib import Path


def _collect_ops_py_files() -> set[str]:
    """Collect all .py files under vllm_ascend/ops/ (excluding __init__.py files)."""
    ops_dir = Path(__file__).parent.parent.parent.parent / "vllm_ascend" / "ops"
    files: set[str] = set()
    for root, _dirs, filenames in os.walk(ops_dir):
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                rel = os.path.relpath(os.path.join(root, f), ops_dir)
                mod = "vllm_ascend.ops." + rel.replace(os.sep, ".").removesuffix(".py")
                files.add(mod)
    return files


def _parse_imported_modules() -> set[str]:
    """Parse __init__.py AST to extract all imported vllm_ascend.ops.* modules."""
    init_path = Path(__file__).parent.parent.parent.parent / "vllm_ascend" / "ops" / "__init__.py"
    tree = ast.parse(init_path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vllm_ascend.ops."):
                    imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("vllm_ascend.ops."):
                imported.add(node.module)
    return imported


def test_ops_init_covers_all_modules():
    """Ensure every .py file under vllm_ascend/ops/ is imported in __init__.py."""
    all_modules = _collect_ops_py_files()
    imported = _parse_imported_modules()

    missing = all_modules - imported
    extra = imported - all_modules

    assert not missing, "Missing imports in vllm_ascend/ops/__init__.py:\n" + "\n".join(
        f"  import {m}  # noqa" for m in sorted(missing)
    )
    assert not extra, "Stale imports in vllm_ascend/ops/__init__.py (no such file):\n" + "\n".join(sorted(extra))
