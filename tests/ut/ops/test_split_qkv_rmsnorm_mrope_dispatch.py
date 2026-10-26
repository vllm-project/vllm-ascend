"""Host-only tests for the lightweight adaptive Split-QKV selector."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar, cast
from unittest import mock

import numpy as np

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


def _wrapper_namespace(policy, helper, requested=2, cores=40):
    """Run the real Python wrapper with tensor/launch stubs, never a compiler."""
    tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_get_ub_capacity",
            "_t1_grouped_eligible",
            "triton_split_qkv_rmsnorm_mrope",
            "triton_split_qkv_rmsnorm_mrope_fake",
        )
    )

    class Launch:
        def __init__(self):
            self.calls = []
            self.options = []

        def __getitem__(self, grid):
            def record(*args, **kwargs):
                self.calls.append((grid, args))
                self.options.append(kwargs)

            return record

    m1, m2 = Launch(), Launch()
    namespace = {
        "torch": SimpleNamespace(
            Tensor=_Tensor, bfloat16="bfloat16", empty=lambda *shape, **kwargs: _tensor(shape, **kwargs)
        ),
        "get_ub_size_bytes": helper,
        "get_vectorcore_num": lambda: cores,
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


class _Device:
    def __init__(self, name):
        self.name = name
        self.type = name.split(":")[0]

    def __eq__(self, other):
        return isinstance(other, _Device) and self.name == other.name


class _Tensor:
    def __init__(self, shape, dtype="bfloat16", device="npu:0", contiguous=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = _Device(device) if isinstance(device, str) else device
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


def _tensor(shape, **kwargs):
    return _Tensor(shape, **kwargs)


def _t1_args(hq, hkv):
    """Keep the packed layout valid while varying the head geometry."""
    return (
        _tensor((1, 2 * (hq + hkv) * 256)),
        _tensor((256,)),
        _tensor((256,)),
        _tensor((3, 1, 64)),
        hq,
        hkv,
        256,
        1e-6,
        [11, 11, 10],
        True,
    )


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


class _Array(np.ndarray):
    """Only the tensor operations needed to execute the actual kernel body."""

    def to(self, dtype):
        return self.astype(dtype)


def _array(value):
    return np.asarray(value).view(_Array)


class _Pointer:
    def __init__(self, values, counts, offsets=0):
        self.values, self.counts, self.offsets = values.reshape(-1), counts, offsets

    def __add__(self, offsets):
        return _Pointer(self.values, self.counts, self.offsets + offsets)


class _Language:
    """CPU FP32 semantics, not a Triton compiler or BF16 accuracy emulator."""

    float32 = np.float32

    def __init__(self):
        self.core = 0

    def program_id(self, axis):
        assert axis == 0
        return self.core

    @staticmethod
    def static_assert(condition, message):
        assert condition, message

    @staticmethod
    def arange(start, end):
        return _array(np.arange(start, end))

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        result = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        result[mask] = pointer.values[offsets[mask]]
        np.add.at(pointer.counts, offsets[mask], 1)
        return _array(result)

    @staticmethod
    def store(pointer, values, mask=True):
        offsets, values, mask = np.broadcast_arrays(pointer.offsets, values, mask)
        pointer.values[offsets[mask]] = values[mask]
        np.add.at(pointer.counts, offsets[mask], 1)

    @staticmethod
    def sum(values, axis):
        return _array(np.sum(values, axis=axis))

    @staticmethod
    def sqrt(values):
        return _array(np.sqrt(values))

    @staticmethod
    def where(condition, x, y):
        return _array(np.where(condition, x, y))

    @staticmethod
    def broadcast_to(values, shape):
        return _array(np.broadcast_to(values, shape))


def _slice(values, offsets, sizes, strides):
    return _array(values[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides))])


def _insert(values, source, offsets, sizes, strides):
    result = values.copy()
    result[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides))] = source
    return result


class SplitQKVMRoPEBodyTests(unittest.TestCase):
    def _check_body(self, hq, hkv, *, tokens=1, cores=None, block_m=1, grouped=False, launch=None):
        tree = ast.parse(OP_PATH.read_text())
        body = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "split_qkv_rmsnorm_mrope_kernel"
        )
        body.decorator_list = []
        tl = _Language()
        namespace = {"tl": tl, "extract_slice": _slice, "insert_slice": _insert}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), body],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(OP_PATH), "exec"), namespace)
        rng = np.random.default_rng(20260812)
        qg = rng.normal(size=(tokens, hq, 2, 256)).astype(np.float32)
        k = rng.normal(size=(tokens, hkv, 256)).astype(np.float32)
        v = rng.normal(size=(tokens, hkv, 256)).astype(np.float32)
        qg[0, 0, 1, 0] = v[0, 0, 0] = -0.0
        qkv = np.concatenate((qg.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)), axis=1)
        qw, kw = [rng.normal(size=256).astype(np.float32) for _ in range(2)]
        coeff = rng.normal(size=(3, tokens, 64)).astype(np.float32)
        inputs = [qkv, qw, kw, coeff]
        outputs = [np.full(tokens * n * 256, np.nan, dtype=np.float32) for n in (hq, hkv, hkv, hq)]
        pointers = [_Pointer(x, np.zeros(x.size, dtype=np.int32)) for x in inputs + outputs]
        jobs = tokens * max(hq, hkv) if grouped else tokens
        cores = jobs if cores is None else cores
        base, extra = divmod(jobs, cores)
        scalar_args = (
            tokens,
            extra,
            base + 1,
            base,
            hq,
            hkv,
            256,
            hq * 256,
            hkv * 256,
            1e-6,
            11,
            11,
            10,
            False,
            True,
            64,
            32,
            True,
            hq * 256,
            block_m,
        )
        options = {"HEAD_GROUPED": grouped}
        if launch is not None:
            grid, args, options = launch
            cores = grid[0]
            self.assertEqual(grid, (jobs,))
            self.assertEqual(args[10], tokens)
            self.assertEqual(args[14:16], (hq, hkv))
            self.assertEqual(options, {"HEAD_GROUPED": grouped})
            scalar_args = args[10:]
        for core in range(cores):
            tl.core = core
            cast(Callable[..., None], namespace["split_qkv_rmsnorm_mrope_kernel"])(
                pointers[0],
                pointers[1],
                None,
                pointers[2],
                None,
                pointers[3],
                *pointers[4:],
                *scalar_args,
                **options,
            )
        np.testing.assert_array_equal(pointers[0].counts, 1)
        for pointer in pointers[4:]:
            np.testing.assert_array_equal(pointer.counts, 1)
        # Independent mathematical reference: lane i chooses T/H/W by i % 3.
        lane = np.arange(32)
        plane = np.where((lane % 3 == 1) & (lane < 33), 1, np.where((lane % 3 == 2) & (lane < 30), 2, 0))
        cos = coeff[plane, :, lane].T[:, None, :]
        sin = coeff[plane, :, lane + 32].T[:, None, :]
        for actual, values, weight in ((outputs[0], qg[:, :, 0], qw), (outputs[1], k, kw)):
            normalized = values / np.sqrt(np.sum(values * values, axis=2, keepdims=True) / 256 + 1e-6) * weight
            expected = normalized.copy()
            expected[:, :, :32] = normalized[:, :, :32] * cos - normalized[:, :, 32:64] * sin
            expected[:, :, 32:64] = normalized[:, :, 32:64] * cos + normalized[:, :, :32] * sin
            np.testing.assert_allclose(actual.reshape(tokens, -1, 256), expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(outputs[2].tobytes(), v.tobytes())
        self.assertEqual(outputs[3].tobytes(), qg[:, :, 1].copy().tobytes())

    def test_actual_body_math_and_asymmetric_head_ownership(self):
        for hq, hkv in ((4, 1), (6, 1), (24, 4), (1, 3)):
            with self.subTest(heads=(hq, hkv)):
                self._check_body(hq, hkv, grouped=True)

    def test_t1_head_boundaries_real_wrapper_to_body(self):
        package = ModuleType("test_split_qkv_head_boundary")
        package.__path__ = [str(OP_DIR)]
        sys.modules[package.__name__] = package
        policy = _load_source_module(package.__name__ + ".ac_dispatch_policy", OP_DIR / "ac_dispatch_policy.py")
        for hq, hkv in ((2, 1), (1, 2), (40, 1), (1, 40)):
            with self.subTest(heads=(hq, hkv)):
                helper = mock.Mock(return_value=196608)
                wrapper, m1, m2 = _wrapper_namespace(policy, helper)
                wrapper["triton_split_qkv_rmsnorm_mrope"](*_t1_args(hq, hkv), rope_dim=64, has_gate=True)
                self.assertEqual(len(m1.calls), 1)
                self.assertEqual(m2.calls, [])
                helper.assert_called_once_with()
                self._check_body(hq, hkv, grouped=True, launch=(*m1.calls[0], m1.options[0]))

    def test_actual_m2_full_tiles_and_odd_token_tail(self):
        for hq, hkv, tokens, cores in ((6, 1, 4, 1), (24, 4, 5, 2)):
            with self.subTest(heads=(hq, hkv), tokens=tokens, cores=cores):
                self._check_body(hq, hkv, tokens=tokens, cores=cores, block_m=2)

    def test_forced_m2_rejects_head_remainder_even_with_only_token_tail(self):
        for hq in (16, 32):
            for tokens in (1, 3):
                with self.subTest(hq=hq, tokens=tokens):
                    with self.assertRaisesRegex(AssertionError, "M2 requires complete Q-head tiles"):
                        self._check_body(hq, 4, tokens=tokens, cores=1, block_m=2)
                    self._check_body(hq, 4, tokens=tokens, cores=1)
            self._check_body(hq, 4, grouped=True)


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

    def _shared_ub_helper(self, properties=None):
        """Load the actual shared helper; replace only its external runtime APIs."""
        torch_stub = ModuleType("torch")
        torch_stub.__dict__["npu"] = SimpleNamespace(current_device=lambda: 0)
        triton_stub = ModuleType("vllm.triton_utils")
        triton_stub.__dict__.update(
            HAS_TRITON=False,  # No extension/compiler import on the host.
            tl=SimpleNamespace(),
            triton=SimpleNamespace(
                runtime=SimpleNamespace(
                    driver=SimpleNamespace(
                        active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda _: properties))
                    )
                )
            ),
        )
        with mock.patch.dict(sys.modules, {"torch": torch_stub, "vllm.triton_utils": triton_stub}):
            helper = _load_source_module("test_split_qkv_shared_ub", OP_DIR.parent / "triton_utils.py")
        if properties is not None:
            helper.__dict__["HAS_TRITON"] = True
            helper.init_device_properties_triton()
        return helper

    def test_shared_ub_contract_detects_or_defaults_without_env_override(self):
        removed = "VLLM_ASCEND_ROPE_UB_SIZE_KB"
        self.assertNotIn(removed, self.envs.env_variables)
        for raw in ("1", "1024", "invalid"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {removed: raw}):
                with self.assertRaises(AttributeError):
                    getattr(self.envs, removed)
                for properties, expected in (
                    ({"ub_size": 256 * 1024}, 256 * 1024),
                    ({"ub_size": 1, "max_ub_size": 128 * 1024}, 128 * 1024),
                    ({"ub_size": 1}, 192 * 1024),
                    ({}, 192 * 1024),
                ):
                    with self.subTest(properties=properties):
                        helper = self._shared_ub_helper({"num_aicore": 20, "num_vectorcore": 40, **properties})
                        self.assertEqual(helper.get_ub_size_bytes(), expected)

    def test_real_shared_helper_wires_default_and_detected_capacity_to_wrapper(self):
        for properties, expected_pair_m in (({}, 2), ({"ub_size": 64 * 1024}, 1)):
            helper = self._shared_ub_helper({"num_aicore": 20, "num_vectorcore": 40, **properties})
            for tokens in (1, 4, 1024):
                with self.subTest(properties=properties, tokens=tokens):
                    namespace, m1, m2 = _wrapper_namespace(self.policy, helper.get_ub_size_bytes)
                    namespace["triton_split_qkv_rmsnorm_mrope"](
                        _tensor((tokens, 3584)),
                        _tensor((256,)),
                        _tensor((256,)),
                        _tensor((3, tokens, 64)),
                        6,
                        1,
                        256,
                        1e-6,
                        [11, 11, 10],
                        True,
                        rope_dim=64,
                        has_gate=True,
                    )
                    expected = expected_pair_m if tokens == 1024 else 1
                    selected, other = (m2, m1) if expected == 2 else (m1, m2)
                    self.assertEqual(len(selected.calls), 1)
                    self.assertEqual(other.calls, [])
                    self.assertEqual(selected.calls[0][1][-1], expected)
                    self.assertEqual(selected.options, [{"HEAD_GROUPED": tokens == 1}])

    def test_uninitialized_shared_helper_falls_back_and_explicit_m1_bypasses_it(self):
        helper = self._shared_ub_helper()
        with self.assertRaisesRegex(AssertionError, "Device properties not initialized"):
            helper.get_ub_size_bytes()
        for tokens in (1, 1024):
            for requested in (1, 2):
                with self.subTest(tokens=tokens, requested=requested):
                    capacity = mock.Mock(wraps=helper.get_ub_size_bytes)
                    namespace, m1, m2 = _wrapper_namespace(self.policy, capacity, requested=requested)
                    namespace["triton_split_qkv_rmsnorm_mrope"](
                        _tensor((tokens, 3584)),
                        _tensor((256,)),
                        _tensor((256,)),
                        _tensor((3, tokens, 64)),
                        6,
                        1,
                        256,
                        1e-6,
                        [11, 11, 10],
                        True,
                        rope_dim=64,
                        has_gate=True,
                    )
                    self.assertEqual(capacity.call_count, int(requested == 2))
                    self.assertEqual(len(m1.calls), 1)
                    self.assertEqual(m1.options, [{"HEAD_GROUPED": False}])
                    self.assertEqual(m2.calls, [])

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

    def _check_boundary_guard(self, policy):
        # Isolate the structural gate: valid layout/semantics, 25 rows per
        # active core (12 pairs + odd tail), and capacity above every demand.
        for hq in range(1, 41):
            q_size, kv_size = hq * 256, 256
            self.assertTrue(
                policy.layout_valid(
                    _tensor((1000, 2 * q_size + 2 * kv_size)),
                    _tensor((256,)),
                    _tensor((256,)),
                    _tensor((3, 1000, 64)),
                    None,
                    None,
                    1000,
                    q_size,
                    q_size,
                    kv_size,
                    256,
                    64,
                )
            )
            for capacity in (768 * 1024, 1024 * 1024):
                result = _decision(policy, q_heads=hq, tokens=1000, capacity=capacity)
                self.assertLess(result.demand_estimate_bytes, capacity)
                supported = hq % min(hq, 12) == 0
                self.assertEqual(policy.pair_boundary_screen_supported(hq), supported)
                self.assertEqual(
                    (result.block_m, result.gate), (2, "shape_resource") if supported else (1, "g2_resource")
                )

    def test_structural_gate_matches_whole_q_head_tiles(self):
        self._check_boundary_guard(self.policy)

    def test_deleted_structural_gate_mutation_is_detected(self):
        tree = ast.parse((OP_DIR / "ac_dispatch_policy.py").read_text())
        selector = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_dispatch"
        )
        guard = next(
            node
            for node in selector.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "has_pair_work and (not pair_boundary_screen_supported(shape.num_q_heads))"
        )
        selector.body.remove(guard)
        namespace = dict(vars(self.policy))
        exec(
            compile(ast.Module(body=[selector], type_ignores=[]), "<policy-without-structure-gate>", "exec"), namespace
        )
        mutant = SimpleNamespace(**namespace)
        self.assertEqual(_decision(mutant, q_heads=13, tokens=1000, capacity=1024 * 1024).block_m, 2)
        with self.assertRaises(AssertionError):
            self._check_boundary_guard(mutant)

    @staticmethod
    def _kernel_ast():
        tree = ast.parse(OP_PATH.read_text(encoding="utf-8"))
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "split_qkv_rmsnorm_mrope_kernel"
        )

    def _check_kernel_contract(self, kernel):
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
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "tl.static_assert"
        ]
        for condition, scope in (
            ("BLOCK_M == 1 or BLOCK_M == 2", kernel.body),
            ("not HEAD_GROUPED or BLOCK_M == 1", kernel.body),
            ("num_q_heads % BLOCK_H_Q == 0", m2.body),
        ):
            matches = [node for node in assertions if ast.unparse(cast(ast.Call, node.value).args[0]) == condition]
            self.assertEqual(len(matches), 1, condition)
            self.assertIn(matches[0], scope, condition)
        head_selection = next(
            node for node in m2.body if isinstance(node, ast.If) and ast.unparse(node.test) == "num_q_heads < 12"
        )
        for definition in (head_selection.body[0], head_selection.orelse[0]):
            self.assertIsInstance(definition, ast.AnnAssign)
            definition = cast(ast.AnnAssign, definition)
            self.assertEqual(ast.unparse(definition.target), "BLOCK_H_Q")
            self.assertEqual(ast.unparse(definition.annotation), "tl.constexpr")
        boundary_assertion = next(
            node
            for node in assertions
            if ast.unparse(cast(ast.Call, node.value).args[0]) == "num_q_heads % BLOCK_H_Q == 0"
        )
        self.assertEqual(m2.body.index(boundary_assertion), m2.body.index(head_selection) + 1)
        self.assertNotIn("N_BOUNDARY_HEADS", ast.unparse(kernel))

    def test_kernel_only_contains_m1_and_m2_with_odd_tail(self):
        self._check_kernel_contract(self._kernel_ast())

    def test_misplaced_m2_assertion_mutation_is_detected(self):
        kernel = self._kernel_ast()
        m1 = next(node for node in kernel.body if isinstance(node, ast.If) and ast.unparse(node.test) == "BLOCK_M == 1")
        m2 = cast(ast.If, m1.orelse[0])
        assertion = next(
            node
            for node in m2.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "tl.static_assert"
        )
        m2.body.remove(assertion)
        kernel.body.insert(0, assertion)
        with self.assertRaises(AssertionError):
            self._check_kernel_contract(kernel)

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
            self.assertEqual(grid, (q_heads if tokens == 1 else min(tokens, 40),))
            self.assertEqual(selected.options[0], {"HEAD_GROUPED": tokens == 1})
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

    def test_t1_grouped_real_wrapper_and_explicit_opt_out(self):
        for hq, hkv in ((4, 1), (6, 1), (24, 4), (1, 3), (40, 1)):
            for requested in (1, 2):
                with self.subTest(heads=(hq, hkv), requested=requested):
                    helper = mock.Mock(return_value=196608)
                    namespace, m1, m2 = _wrapper_namespace(self.policy, helper, requested)
                    outputs = namespace["triton_split_qkv_rmsnorm_mrope"](
                        _tensor((1, (2 * hq + 2 * hkv) * 256)),
                        _tensor((256,)),
                        _tensor((256,)),
                        _tensor((3, 1, 64)),
                        hq,
                        hkv,
                        256,
                        1e-6,
                        [11, 11, 10],
                        True,
                        rope_dim=64,
                        has_gate=True,
                    )
                    grouped = requested == 2
                    self.assertEqual(m1.calls[0][0], (max(hq, hkv) if grouped else 1,))
                    self.assertEqual(m1.options, [{"HEAD_GROUPED": grouped}])
                    self.assertEqual(m1.calls[0][1][-1], 1)
                    self.assertEqual(len(m1.calls), 1)
                    self.assertEqual(m2.calls, [])
                    self.assertEqual(helper.call_count, int(grouped))
                    self.assertEqual(
                        [x.shape for x in outputs], [(1, hq * 256), (1, hkv * 256), (1, hkv * 256), (1, hq * 256)]
                    )

    def test_grouped_eligibility_rejects_unqualified_inputs(self):
        namespace, _, _ = _wrapper_namespace(self.policy, mock.Mock(return_value=196608))
        defaults = [
            _tensor((1, 3584)),
            _tensor((256,)),
            _tensor((256,)),
            _tensor((3, 1, 64)),
            6,
            1,
            256,
            1e-6,
            [11, 11, 10],
            True,
            64,
            None,
            None,
            True,
            40,
        ]
        eligible = namespace["_t1_grouped_eligible"]
        self.assertTrue(eligible(*defaults))
        mutations = [
            (14, 48),
            (14, None),
            (14, True),
            (4, True),
            (5, 0),
            (6, 128),
            (7, 1e-5),
            (7, float("nan")),
            (8, [10, 11, 11]),
            (8, [True, 11, 20]),
            (9, False),
            (10, 256),
            (11, _tensor((256,))),
            (12, _tensor((256,))),
            (13, False),
            (0, object()),
            (0, _tensor((1, 3584), dtype="float16")),
            (0, _tensor((1, 3584), device="cpu")),
            (0, _tensor((1, 3584), contiguous=False)),
            (1, _tensor((255,))),
            (2, _tensor((256,), device="npu:1")),
            (3, _tensor((3, 1, 63))),
        ]
        for index, value in mutations:
            with self.subTest(index=index, value=value):
                args = defaults.copy()
                args[index] = value
                self.assertFalse(eligible(*args))

    def test_grouped_head_boundaries_have_valid_layout(self):
        for hq, hkv, grouped in (
            (2, 1, True),
            (1, 2, True),
            (40, 1, True),
            (1, 40, True),
            (1, 1, False),
            (41, 41, False),
        ):
            with self.subTest(heads=(hq, hkv)):
                helper = mock.Mock(return_value=196608)
                namespace, m1, m2 = _wrapper_namespace(self.policy, helper)
                args = _t1_args(hq, hkv)
                self.assertTrue(
                    self.policy.layout_valid(*args[:4], None, None, 1, hq * 256, hq * 256, hkv * 256, 256, 64)
                )
                layout = mock.Mock(wraps=self.policy.layout_valid)
                namespace["layout_valid"] = layout
                self.assertIs(namespace["_t1_grouped_eligible"](*args, 64, None, None, True, 40), grouped)
                if grouped:
                    layout.assert_called_once()
                else:
                    # Valid layout must not hide removal of a head eligibility guard.
                    layout.assert_not_called()
                outputs = namespace["triton_split_qkv_rmsnorm_mrope"](*args, rope_dim=64, has_gate=True)
                self.assertEqual(len(m1.calls), 1)
                self.assertEqual(m2.calls, [])
                self.assertEqual(m1.calls[0][0], (max(hq, hkv) if grouped else 1,))
                self.assertEqual(
                    m1.calls[0][1][10:14], (1, max(hq, hkv) if grouped else 1, 1, max(hq, hkv) // 40 if grouped else 0)
                )
                self.assertEqual(m1.options, [{"HEAD_GROUPED": grouped}])
                helper.assert_called_once_with()
                self.assertEqual(
                    [output.shape for output in outputs],
                    [(1, hq * 256), (1, hkv * 256), (1, hkv * 256), (1, hq * 256)],
                )

    def test_t1_unknown_capacity_and_p48_keep_legacy_launch(self):
        for capacity, cores in ((None, 40), (True, 40), (65535, 40), (1048577, 40), (196608, 48)):
            with self.subTest(capacity=capacity, cores=cores):
                helper = mock.Mock(return_value=capacity)
                namespace, m1, m2 = _wrapper_namespace(self.policy, helper, cores=cores)
                namespace["triton_split_qkv_rmsnorm_mrope"](
                    _tensor((1, 3584)),
                    _tensor((256,)),
                    _tensor((256,)),
                    _tensor((3, 1, 64)),
                    6,
                    1,
                    256,
                    1e-6,
                    [11, 11, 10],
                    True,
                    rope_dim=64,
                    has_gate=True,
                )
                self.assertEqual(m1.calls[0][0], (1,))
                self.assertEqual(m1.options, [{"HEAD_GROUPED": False}])
                self.assertEqual(m2.calls, [])
                helper.assert_called_once_with()

    def test_fake_abi_never_queries_environment_or_launches(self):
        namespace, m1, m2 = _wrapper_namespace(self.policy, mock.Mock(side_effect=AssertionError("must not probe")))
        namespace["get_vectorcore_num"] = mock.Mock(side_effect=AssertionError("must not query cores"))
        outputs = namespace["triton_split_qkv_rmsnorm_mrope_fake"](
            _tensor((1, 2048)),
            _tensor((256,)),
            _tensor((256,)),
            _tensor((3, 1, 64)),
            6,
            1,
            256,
            1e-6,
            [11, 11, 10],
            True,
            rope_dim=64,
        )
        self.assertEqual([x.shape for x in outputs], [(1, 1536), (1, 256), (1, 256), (1, 0)])
        self.assertEqual(m1.calls + m2.calls, [])


if __name__ == "__main__":
    unittest.main()
