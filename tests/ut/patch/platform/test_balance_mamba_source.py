# SPDX-License-Identifier: Apache-2.0
"""Source-level guards for Mamba handling in ``BalanceScheduler``."""

import ast
import unittest
from pathlib import Path


def _schedule_tree() -> ast.FunctionDef:
    repo_root = Path(__file__).resolve().parents[4]
    source = (
        repo_root / "vllm_ascend" / "patch" / "platform" / "patch_balance_schedule.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    scheduler = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "BalanceScheduler"
    )
    return next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name == "schedule"
    )


class TestBalanceMambaSource(unittest.TestCase):
    def test_mamba_block_aligned_split_uses_upstream_interface(self):
        """Guard the waiting-path call against upstream signature drift."""
        branches = [
            node
            for node in ast.walk(_schedule_tree())
            if isinstance(node, ast.If)
            and "need_mamba_block_aligned_split" in ast.unparse(node.test)
            and "load_kv_async" in ast.unparse(node.test)
        ]

        self.assertEqual(len(branches), 1)
        calls = [
            node
            for node in ast.walk(branches[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_mamba_block_aligned_split"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in calls[0].args],
            [
                "request",
                "num_new_tokens",
                "num_new_local_computed_tokens",
                "num_external_computed_tokens",
            ],
        )

    def test_hybrid_connector_clears_shared_prefix_boundary(self):
        """The per-group lookup cannot provide a shared-prefix junction."""
        branches = [
            node
            for node in ast.walk(_schedule_tree())
            if isinstance(node, ast.If)
            and "has_mamba_layers" in ast.unparse(node.test)
            and "HybridKVCacheCoordinator" in ast.unparse(node.test)
        ]

        self.assertEqual(len(branches), 1)
        resets = [
            node
            for node in ast.walk(branches[0])
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "request.shared_prefix_boundary"
            and isinstance(node.value, ast.Constant)
            and node.value.value == 0
        ]
        self.assertEqual(len(resets), 1)


if __name__ == "__main__":
    unittest.main()
