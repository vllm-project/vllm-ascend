"""Host-only tests for the lightweight adaptive Split-QKV selector."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar, cast
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
OP_DIR = ROOT / "vllm_ascend/ops/triton/linearnorm"
OP_PATH = OP_DIR / "split_qkv_rmsnorm_mrope.py"
ENV_PATH = ROOT / "vllm_ascend/envs.py"
ENV_NAME = "VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _wrapper_namespace(policy, helper, requested=2):
    """Run the real Python wrapper with tensor/launch stubs, never a compiler."""
    tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_get_ub_capacity", "triton_split_qkv_rmsnorm_mrope")
    )

    class Launch:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def record(*args):
                self.calls.append((grid, args))

            return record

    m1, m2 = Launch(), Launch()
    namespace = {
        "torch": SimpleNamespace(empty=lambda *shape, **kwargs: _tensor(shape, **kwargs)),
        "get_ub_size_bytes": helper,
        "get_vectorcore_num": lambda: 40,
        "ShapeKey": policy.ShapeKey,
        "layout_valid": policy.layout_valid,
        "select_dispatch": policy.select_dispatch,
        "envs": SimpleNamespace(**{ENV_NAME: requested}),
        "split_qkv_rmsnorm_mrope_kernel": m1,
        "_M2_KERNEL": m2,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(OP_PATH), "exec"), namespace)
    return namespace, m1, m2


def _tensor(shape, dtype="bfloat16", device="npu:0", contiguous=True):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype, device=device, is_contiguous=lambda: contiguous)


def _decision(
    policy,
    *,
    q_heads=6,
    kv_heads=1,
    tokens=1024,
    cores=40,
    capacity=196608,
    layout_valid=True,
    head_size=256,
    has_gate=True,
    rope_dim=64,
    has_bias=False,
    dtype="bfloat16",
):
    return policy.select_dispatch(
        shape=policy.ShapeKey(
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_size=head_size,
            has_gate=has_gate,
            rope_dim=rope_dim,
            has_bias=has_bias,
            is_interleaved=True,
            mrope_section=(11, 11, 10),
            eps=1e-6,
            dtype=dtype,
        ),
        # Independent partition formula: min/max over active cores only.
        min_active_tokens=(tokens // cores if tokens >= cores else int(tokens > 0)),
        max_active_tokens=(tokens + cores - 1) // cores,
        current_layout_valid=layout_valid,
        capacity_bytes=capacity,
    )


class SplitQKVMRoPEDispatchTests(unittest.TestCase):
    policy: ClassVar[ModuleType]
    envs: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls):
        package = ModuleType("test_split_qkv_mrope")
        package.__path__ = [str(OP_DIR)]
        sys.modules[package.__name__] = package
        cls.policy = _load_source_module(package.__name__ + ".ac_dispatch_policy", OP_DIR / "ac_dispatch_policy.py")
        cls.envs = _load_source_module("test_split_qkv_mrope_envs", ENV_PATH)

    def test_public_op_and_single_launch_contract(self):
        tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        wrapper = functions["triton_split_qkv_rmsnorm_mrope"]
        fake = functions["triton_split_qkv_rmsnorm_mrope_fake"]
        self.assertEqual(len(wrapper.args.args), 14)
        self.assertEqual([arg.arg for arg in wrapper.args.args], [arg.arg for arg in fake.args.args])
        launches = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "launch_kernel"
        ]
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(launches[0].args), 30)
        self.assertEqual(cast(ast.Name, launches[0].args[-1]).id, "block_m")
        source = OP_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count('op_name="triton_split_qkv_rmsnorm_mrope"'), 1)
        self.assertNotIn("PAIR_CAPABLE", source)
        self.assertNotIn("_ac_last_decision", source)
        self.assertNotIn("importlib", source)
        self.assertNotIn("hashlib", source)
        self.assertNotIn("get_soc_version", source)
        self.assertNotIn("toolchain", source)
        self.assertNotIn("make_partition", source)

    def test_centralized_block_m_env_is_lazy_and_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(getattr(self.envs, ENV_NAME), 2)
        for raw, expected in (("1", 1), ("2", 2)):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}):
                self.assertEqual(getattr(self.envs, ENV_NAME), expected)
        for raw in ("0", "3", "4", "bogus"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {ENV_NAME: raw}), self.assertRaises(ValueError):
                getattr(self.envs, ENV_NAME)
        self.assertIn("block_m = envs." + ENV_NAME, OP_PATH.read_text(encoding="utf-8"))

    def test_capacity_uses_public_byte_helper_without_backend_probe(self):
        helper = mock.Mock(return_value=196608)
        namespace, _, _ = _wrapper_namespace(self.policy, helper)
        self.assertEqual(namespace["_get_ub_capacity"](), 196608)
        helper.assert_called_once_with()
        source = OP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("triton.backends.ascend.runtime", source)
        self.assertNotIn("ub_size_in_kbytes", source)
        self.assertNotIn("get_device_properties", source)

    def test_helper_capacity_changes_resource_choice_not_workload_gate(self):
        for capacity_bytes, gate, block_m in (
            (64 * 1024, "g2_resource", 1),
            (192 * 1024, "shape_resource", 2),
            (256 * 1024, "shape_resource", 2),
        ):
            with self.subTest(capacity_bytes=capacity_bytes):
                result = _decision(self.policy, capacity=capacity_bytes)
                self.assertEqual((result.gate, result.block_m), (gate, block_m))
        short = _decision(self.policy, tokens=192, capacity=256 * 1024)
        self.assertEqual((short.gate, short.block_m), ("g3_workload", 1))

    def test_invalid_or_unavailable_helper_capacity_falls_back(self):
        helpers = [mock.Mock(return_value=value) for value in (1, 0, -1, True, 196608.0, None, 1024 * 1024 + 1)]
        helpers.append(mock.Mock(side_effect=AssertionError("properties uninitialized")))
        for helper in helpers:
            with self.subTest(helper=helper):
                namespace, _, _ = _wrapper_namespace(self.policy, helper)
                result = _decision(self.policy, capacity=namespace["_get_ub_capacity"]())
                self.assertEqual((result.gate, result.block_m), ("g2_resource", 1))
                self.assertIsNone(result.demand_estimate_bytes)

    def test_capacity_does_not_relax_boundary_guard(self):
        for q_heads in (13, 16, 28):
            for head_size in (128, 256):
                result = _decision(self.policy, q_heads=q_heads, kv_heads=4, head_size=head_size, capacity=256 * 1024)
                self.assertEqual((result.gate, result.block_m), ("g2_resource", 1))

    def test_kernel_only_contains_m1_and_m2_with_odd_tail(self):
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
        m2 = cast(ast.If, branches[0].orelse[0])
        self.assertEqual(ast.unparse(m2.test), "BLOCK_M == 2")
        self.assertEqual(m2.orelse, [])
        self.assertTrue(any(isinstance(node, ast.For) and ast.unparse(node.target) == "index" for node in m2.body))
        self.assertTrue(any(isinstance(node, ast.For) and ast.unparse(node.target) == "tail_index" for node in m2.body))
        assertions = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "static_assert"
        ]
        self.assertEqual(len(assertions), 1)
        self.assertEqual(ast.unparse(assertions[0].args[0]), "BLOCK_M == 1 or BLOCK_M == 2")

    def test_seven_b3_routes_without_case_allowlist(self):
        for point, q_heads, kv_heads, tokens, gate, block_m in (
            ("E2C", 16, 4, 192, "g2_resource", 1),
            ("P3", 6, 1, 1, "g3_workload", 1),
            ("P1", 6, 1, 192, "g3_workload", 1),
            ("P2", 24, 4, 192, "g3_workload", 1),
            ("T1024", 6, 1, 1024, "shape_resource", 2),
            ("P4", 6, 1, 4096, "shape_resource", 2),
            ("P5", 4, 1, 8192, "shape_resource", 2),
        ):
            with self.subTest(point=point):
                result = _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, tokens=tokens)
                self.assertEqual((result.gate, result.block_m), (gate, block_m))

    def test_unknown_capacity_bad_layout_and_empty_input_fall_back(self):
        for kwargs, gate in (
            ({"capacity": None}, "g2_resource"),
            ({"layout_valid": False}, "g1_semantic"),
            ({"layout_valid": 1}, "g1_semantic"),
            ({"tokens": 0}, "g1_semantic"),
        ):
            result = _decision(self.policy, **kwargs)
            self.assertEqual((result.gate, result.block_m), (gate, 1))

    def test_g3_threshold_tracks_active_core_partition(self):
        for cores in (1, 8, 40, 64):
            for tokens, block_m in ((24 * cores - 1, 1), (24 * cores, 2), (24 * cores + 1, 2)):
                with self.subTest(cores=cores, tokens=tokens):
                    result = _decision(self.policy, tokens=tokens, cores=cores)
                    self.assertEqual(result.block_m, block_m)

    def test_hq24_long_partitions_select_m2(self):
        for tokens in (960, 961, 1024, 4096):
            result = _decision(self.policy, q_heads=24, kv_heads=4, tokens=tokens)
            self.assertEqual((result.gate, result.block_m), ("shape_resource", 2))
            self.assertEqual(result.demand_estimate_bytes, 166119)
        self.assertEqual(_decision(self.policy, q_heads=24, kv_heads=4, tokens=959).block_m, 1)

    def test_singleton_partitions_select_m1(self):
        for q_heads, kv_heads in ((6, 1), (24, 4), (16, 4)):
            for tokens in (1, 39, 40):
                result = _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, tokens=tokens)
                self.assertEqual((result.gate, result.block_m), ("g3_workload", 1))

    def test_exact_fit_and_frozen_demand_values(self):
        for q_heads, kv_heads, has_bias, demand in ((6, 1, False, 77348), (4, 1, False, 54701), (24, 4, False, 166119)):
            result = _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, has_bias=has_bias)
            self.assertEqual(result.demand_estimate_bytes, demand)
            self.assertEqual(_decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, capacity=demand).block_m, 1)
            # Range validation stays separate; the smallest model point is below 64 KiB.
            if demand >= 64 * 1024:
                self.assertEqual(
                    _decision(self.policy, q_heads=q_heads, kv_heads=kv_heads, capacity=demand + 1).block_m, 2
                )

    def test_semantic_checks_preserved(self):
        for kwargs in (
            {"num_q_heads": 0},
            {"num_q_heads": True},
            {"has_gate": 1},
            {"eps": float("nan")},
            {"mrope_section": (11, 11, 9)},
            {"rope_dim": 63},
            {"dtype": "float32"},
        ):
            values = dict(
                num_q_heads=6,
                num_kv_heads=1,
                head_size=256,
                has_gate=True,
                rope_dim=64,
                has_bias=False,
                is_interleaved=True,
                mrope_section=(11, 11, 10),
                eps=1e-6,
                dtype="bfloat16",
            )
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.policy.ShapeKey(**values)

    def test_layout_checks_without_identity_strings(self):
        tensors = [_tensor((1024, 2048)), _tensor((256,)), _tensor((256,)), _tensor((3, 1024, 64)), None, None]

        def valid():
            return self.policy.layout_valid(*tensors, 1024, 1536, 0, 256, 256, 64)

        self.assertTrue(valid())
        for slot, bad in (
            (0, _tensor((1024, 2048), contiguous=False)),
            (1, _tensor((128,))),
            (2, _tensor((256,), dtype="float16")),
            (3, _tensor((3, 1024, 64), device="npu:1")),
            (4, _tensor((256,))),
            (0, object()),
        ):
            original = tensors[slot]
            tensors[slot] = bad
            self.assertFalse(valid())
            tensors[slot] = original
        tensors[4:] = [_tensor((256,)), _tensor((256,))]
        self.assertTrue(valid())

    def test_real_wrapper_reuses_partition_and_emits_selected_instance(self):
        for tokens, q_heads, kv_heads, expected in (
            (1, 6, 1, 1),
            (41, 6, 1, 1),
            (959, 6, 1, 1),
            (960, 6, 1, 2),
            (961, 24, 4, 2),
            (1024, 16, 4, 1),
        ):
            helper = mock.Mock(return_value=196608)
            namespace, m1, m2 = _wrapper_namespace(self.policy, helper)
            outputs = namespace["triton_split_qkv_rmsnorm_mrope"](
                _tensor((tokens, (2 * q_heads + 2 * kv_heads) * 256)),
                _tensor((256,)),
                _tensor((256,)),
                _tensor((3, tokens, 64)),
                q_heads,
                kv_heads,
                256,
                1e-6,
                [11, 11, 10],
                True,
                rope_dim=64,
                has_gate=True,
            )
            helper.assert_called_once_with()
            selected = m2 if expected == 2 else m1
            other = m1 if expected == 2 else m2
            self.assertEqual(len(selected.calls), 1)
            self.assertEqual(other.calls, [])
            grid, args = selected.calls[0]
            self.assertEqual(args[-1], expected)
            self.assertEqual(grid, (min(tokens, 40),))
            self.assertEqual(
                [output.shape for output in outputs],
                [(tokens, q_heads * 256), (tokens, kv_heads * 256), (tokens, kv_heads * 256), (tokens, q_heads * 256)],
            )

    def test_explicit_m1_bypasses_capacity_and_dispatch(self):
        helper = mock.Mock(side_effect=AssertionError("must not probe"))
        namespace, m1, m2 = _wrapper_namespace(self.policy, helper, requested=1)
        namespace["select_dispatch"] = mock.Mock(side_effect=AssertionError("must not select"))
        namespace["triton_split_qkv_rmsnorm_mrope"](
            _tensor((1024, 3072)),
            _tensor((256,)),
            _tensor((256,)),
            _tensor((3, 1024, 64)),
            6,
            1,
            256,
            1e-6,
            [11, 11, 10],
            True,
            rope_dim=64,
            has_gate=True,
        )
        helper.assert_not_called()
        self.assertEqual(len(m1.calls), 1)
        self.assertEqual(m2.calls, [])


if __name__ == "__main__":
    unittest.main()
