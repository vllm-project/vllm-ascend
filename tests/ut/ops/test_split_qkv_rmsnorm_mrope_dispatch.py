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
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
OP_DIR = ROOT / "vllm_ascend/ops/triton/linearnorm"
OP_PATH = OP_DIR / "split_qkv_rmsnorm_mrope.py"
ENV_PATH = ROOT / "vllm_ascend/envs.py"
ENV_NAME = "VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"
FINGERPRINT = "torch==2.10.0+cpu;triton==3.5.0;torch-npu==2.10.0.post2"


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
        self.assertIsInstance(launches[0].args[-1], ast.Name)
        self.assertEqual(launches[0].args[-1].id, "pair_capable")
        self.assertEqual(
            OP_PATH.read_text(encoding="utf-8").count('op_name="triton_split_qkv_rmsnorm_mrope"'),
            1,
        )

    def test_centralized_block_m_env_is_lazy_and_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(getattr(self.envs, ENV_NAME), 2)
        for raw, expected in (("1", 1), ("2", 2), ("4", 4)):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}):
                self.assertEqual(getattr(self.envs, ENV_NAME), expected)
        for raw in ("0", "3", "bogus"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}), self.assertRaises(ValueError):
                getattr(self.envs, ENV_NAME)
        wrapper_source = OP_PATH.read_text(encoding="utf-8")
        self.assertIn("block_m = envs.VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M", wrapper_source)
        self.assertNotIn('os.environ.get("SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"', wrapper_source)

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
