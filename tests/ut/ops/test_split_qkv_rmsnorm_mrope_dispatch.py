"""Host-only checks for the Split-QKV RMSNorm MRoPE adaptive dispatch.

The policy is loaded without importing torch, Triton, or touching an NPU.
Device numerics and performance require separate on-device validation.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import ClassVar, cast
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
OP_DIR = ROOT / "vllm_ascend/ops/triton/linearnorm"
OP_PATH = OP_DIR / "split_qkv_rmsnorm_mrope.py"
ENV_PATH = ROOT / "vllm_ascend/envs.py"
ENV_NAME = "VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"
FINGERPRINT = "torch==2.10.0+cpu;triton==3.5.0;torch-npu==2.10.0.post2"


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capacity_from_helper(helper):
    """Exercise the real observation function without importing an NPU runtime."""
    tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_ac_observe_capacity"
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.startswith("_UB_BYTES_") for target in node.targets)
        )
    ]
    namespace = {"get_ub_size_bytes": helper}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OP_PATH), "exec"), namespace)
    return namespace["_ac_observe_capacity"](FINGERPRINT)


def _decision(policy, *, q_heads: int, kv_heads: int, tokens: int, capacity=None, layout_valid=True):
    if capacity is None:
        capacity = {
            "c1": {"value": 1, "state": "invalid"},
            "c2": {"value": 196608, "state": "valid", "target": FINGERPRINT},
            "toolchain_fingerprint": FINGERPRINT,
        }
    shape = policy.ShapeKey(
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=256,
        has_gate=True,
        rope_dim=64,
        has_bias=False,
        is_interleaved=True,
        mrope_section=(11, 11, 10),
        eps=1e-6,
        dtype="bfloat16",
    )
    partition = policy.make_partition(tokens, 40)
    return policy.select_dispatch(
        shape=shape,
        num_tokens=tokens,
        vector_core_count=40,
        soc="unknown",  # Informational, not a hardware allowlist.
        resource_profile="unknown-vc40",
        current_source_sha256="a" * 64,
        current_compiler_config=policy.CompilerConfig(
            pair_capable=partition.max_active_tokens >= 2,
            multibuffer=False,
            num_stages=1,
            num_warps=32,
        ),
        current_layout_contract="valid-host-only-layout",
        current_layout_valid=layout_valid,
        current_capacity=capacity,
    )


class SplitQKVMRoPEDispatchTests(unittest.TestCase):
    policy: ClassVar[ModuleType]
    envs: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls):
        cls.policy = _load_source_module("test_split_qkv_mrope_policy", OP_DIR / "ac_dispatch_policy.py")
        cls.envs = _load_source_module("test_split_qkv_mrope_envs", ENV_PATH)

    def test_public_op_and_single_launch_contract(self):
        tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        wrapper = functions["triton_split_qkv_rmsnorm_mrope"]
        fake = functions["triton_split_qkv_rmsnorm_mrope_fake"]
        self.assertEqual(len(wrapper.args.args), 14)
        self.assertEqual(
            [arg.arg for arg in wrapper.args.args],
            [arg.arg for arg in fake.args.args],
        )
        launches = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "launch_kernel"
        ]
        self.assertEqual(len(launches), 1)
        pair_arg = launches[0].args[-1]
        self.assertIsInstance(pair_arg, ast.Name)
        self.assertEqual(cast(ast.Name, pair_arg).id, "pair_capable")
        self.assertEqual(
            OP_PATH.read_text(encoding="utf-8").count('op_name="triton_split_qkv_rmsnorm_mrope"'),
            1,
        )

    def test_centralized_block_m_env_is_lazy_and_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(getattr(self.envs, ENV_NAME), 2)
        for raw, expected in (("1", 1), ("2", 2)):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}):
                self.assertEqual(getattr(self.envs, ENV_NAME), expected)
        for raw in ("0", "3", "4", "bogus"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}), self.assertRaises(ValueError):
                getattr(self.envs, ENV_NAME)
        wrapper_source = OP_PATH.read_text(encoding="utf-8")
        self.assertIn("block_m = envs.VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M", wrapper_source)
        self.assertNotIn('os.environ.get("SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"', wrapper_source)

    def test_capacity_uses_public_byte_helper_without_backend_probe(self):
        helper = mock.Mock(return_value=196608)
        observed = _capacity_from_helper(helper)
        helper.assert_called_once_with()
        self.assertEqual(observed["c1"], {})
        self.assertEqual(observed["c2"]["value"], 196608)
        self.assertEqual(observed["c2"]["state"], "valid")
        self.assertIn("shared default/override", observed["c2"]["reason"])
        source = OP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("triton.backends.ascend.runtime", source)
        self.assertNotIn("ub_size_in_kbytes", source)
        self.assertNotIn("get_device_properties", source)
        self.assertIn("get_ub_size_bytes", source)

    def test_helper_capacity_changes_resource_choice_not_workload_gate(self):
        for capacity_bytes, gate, block_m in (
            (64 * 1024, "g2_resource", 1),
            (192 * 1024, "shape_resource", 2),
            (256 * 1024, "shape_resource", 2),
        ):
            with self.subTest(capacity_bytes=capacity_bytes):
                observed = _capacity_from_helper(mock.Mock(return_value=capacity_bytes))
                result = _decision(self.policy, q_heads=6, kv_heads=1, tokens=1024, capacity=observed)
                self.assertEqual((result.gate, result.block_m), (gate, block_m))
                self.assertEqual(result.envelope_bytes, capacity_bytes)
        short = _decision(
            self.policy,
            q_heads=6,
            kv_heads=1,
            tokens=192,
            capacity=_capacity_from_helper(mock.Mock(return_value=256 * 1024)),
        )
        self.assertEqual((short.gate, short.block_m), ("g3_workload", 1))

    def test_invalid_or_unavailable_helper_capacity_falls_back(self):
        helpers = [mock.Mock(return_value=value) for value in (1, 0, -1, True, 196608.0, None)]
        helpers.append(mock.Mock(side_effect=AssertionError("device properties not initialized")))
        for helper in helpers:
            with self.subTest(helper=helper):
                observed = _capacity_from_helper(helper)
                self.assertNotEqual(observed["c2"]["state"], "valid")
                result = _decision(self.policy, q_heads=6, kv_heads=1, tokens=1024, capacity=observed)
                self.assertEqual((result.gate, result.block_m), ("g2_resource", 1))
                self.assertIsNone(result.envelope_bytes)

    def test_helper_reuses_budget_without_relaxing_boundary_guard(self):
        observed = _capacity_from_helper(mock.Mock(return_value=256 * 1024))
        result = _decision(self.policy, q_heads=16, kv_heads=4, tokens=1024, capacity=observed)
        self.assertEqual((result.gate, result.block_m), ("g2_resource", 1))
        self.assertEqual(result.feasible_variants, ())

    def test_kernel_only_contains_m1_and_m2_paths(self):
        tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "split_qkv_rmsnorm_mrope_kernel"
        )
        branches = [
            node for node in kernel.body if isinstance(node, ast.If) and ast.unparse(node.test) == "BLOCK_M == 1"
        ]
        self.assertEqual(len(branches), 1)
        m2 = branches[0].orelse
        self.assertEqual(len(m2), 1)
        self.assertIsInstance(m2[0], ast.If)
        self.assertEqual(ast.unparse(cast(ast.If, m2[0]).test), "BLOCK_M == 2")
        self.assertEqual(cast(ast.If, m2[0]).orelse, [])
        assertions = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "static_assert"
        ]
        self.assertEqual(len(assertions), 1)
        self.assertEqual(ast.unparse(assertions[0].args[0]), "BLOCK_M == 1 or BLOCK_M == 2")

    def test_seven_b3_routes_without_case_allowlist(self):
        points = (
            ("E2C", 16, 4, 192, "g2_resource", 1, False),
            ("P3", 6, 1, 1, "g3_workload", 1, False),
            ("P1", 6, 1, 192, "g3_workload", 1, False),
            ("P2", 24, 4, 192, "g3_workload", 1, False),
            ("T1024", 6, 1, 1024, "shape_resource", 2, True),
            ("P4", 6, 1, 4096, "shape_resource", 2, True),
            ("P5", 4, 1, 8192, "shape_resource", 2, True),
        )
        for point, q_heads, kv_heads, tokens, gate, block_m, pair in points:
            with self.subTest(point=point):
                result = _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, tokens=tokens)
                self.assertEqual((result.gate, result.block_m, result.pair_capable), (gate, block_m, pair))

    def test_unknown_capacity_and_bad_layout_fall_back(self):
        unknown_capacity = {
            "c1": {"value": 1, "state": "invalid"},
            "c2": {"value": None, "state": "unknown", "target": FINGERPRINT},
            "toolchain_fingerprint": FINGERPRINT,
        }
        for kwargs, gate in (
            ({"capacity": unknown_capacity}, "g2_resource"),
            ({"layout_valid": False}, "g1_semantic"),
        ):
            result = _decision(self.policy, q_heads=6, kv_heads=1, tokens=1024, **kwargs)
            self.assertEqual((result.gate, result.block_m), (gate, 1))

    def test_g3_trial_threshold(self):
        below = _decision(self.policy, q_heads=6, kv_heads=1, tokens=959)
        at = _decision(self.policy, q_heads=6, kv_heads=1, tokens=960)
        self.assertEqual((below.gate, below.block_m), ("g3_workload", 1))
        self.assertEqual((at.gate, at.block_m), ("shape_resource", 2))

    def test_hq24_long_partitions_select_pair_instance(self):
        for tokens in (960, 961, 1024, 4096):
            with self.subTest(tokens=tokens):
                result = _decision(self.policy, q_heads=24, kv_heads=4, tokens=tokens)
                self.assertEqual((result.gate, result.block_m, result.pair_capable), ("shape_resource", 2, True))
        below = _decision(self.policy, q_heads=24, kv_heads=4, tokens=959)
        self.assertEqual((below.gate, below.block_m, below.pair_capable), ("g3_workload", 1, False))

    def test_singleton_partitions_select_m1(self):
        for q_heads, kv_heads in ((6, 1), (24, 4)):
            for tokens in (1, 39, 40):
                with self.subTest(q_heads=q_heads, tokens=tokens):
                    result = _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, tokens=tokens)
                    self.assertEqual((result.gate, result.block_m, result.pair_capable), ("g3_workload", 1, False))


if __name__ == "__main__":
    unittest.main()
