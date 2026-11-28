"""Host-only tests for the lightweight adaptive Split-QKV selector."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast
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


def _wrapper_namespace(contracts, helper, requested=2, cores=40, *, source_text=None):
    """Run the real Python wrapper with tensor/launch stubs, never a compiler."""
    tree = ast.parse(source_text if source_text is not None else OP_PATH.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_validate_scalars",
            "_layout_valid",
            "_get_ub_capacity",
            "_t1_grouped_eligible",
            "_select_tiling",
            "triton_split_qkv_rmsnorm_mrope",
            "triton_split_qkv_rmsnorm_mrope_fake",
        )
    )

    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            in (
                "_MIN_PAIR_ITERATIONS",
                "_UB_BYTES_MIN",
                "_UB_BYTES_MAX",
                "_M4_MAX_KV_HEADS",
                "_M4_SCREEN_CAP_BYTES",
                "_M2_PERF_MAX_HEAD_SIZE",
            )
            for target in node.targets
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
        "math": math,
        "working_set": contracts.working_set,
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


def _load_contracts(package_name):
    resource = _load_source_module(package_name + ".ac_working_set", OP_DIR / "ac_working_set.py")
    return SimpleNamespace(working_set=resource.working_set)


def _decision(
    contracts,
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
    namespace, _, _ = _wrapper_namespace(contracts, lambda: capacity, cores=cores)
    bias = _tensor((head_size,), dtype=dtype) if has_bias else None
    sections = (11, 11, 10) if rope_dim == 64 else (rope_dim // 2, 0, 0)
    return namespace["_select_tiling"](
        _tensor(
            (tokens, (q_heads * (2 if has_gate else 1) + 2 * kv_heads) * head_size),
            dtype=dtype,
            contiguous=layout_valid is True,
        ),
        _tensor((head_size,), dtype=dtype),
        _tensor((head_size,), dtype=dtype),
        _tensor((3, tokens, rope_dim), dtype=dtype),
        q_heads,
        kv_heads,
        head_size,
        1e-6,
        sections,
        True,
        rope_dim,
        bias,
        bias,
        has_gate,
        cores,
        2,
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

    def load(self, pointer, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
        result = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        result[mask] = pointer.values[offsets[mask]]
        np.add.at(pointer.counts, offsets[mask], 1)
        return _array(result)

    def store(self, pointer, values, mask=True):
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
    def _check_body(self, hq, hkv, *, tokens=1, cores=None, block_m=1, grouped=False, launch=None, source_text=None):
        tree = ast.parse(OP_PATH.read_text() if source_text is None else source_text)
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_process_single_row", "split_qkv_rmsnorm_mrope_kernel")
        ]
        for function in functions:
            function.decorator_list = []
        tl = _Language()
        namespace = {"tl": tl, "extract_slice": _slice, "insert_slice": _insert}
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *functions,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(OP_PATH), "exec"), namespace)
        rng = np.random.default_rng(20260812)
        qg = np.asarray(rng.normal(size=(tokens, hq, 2, 256)), dtype=np.float32)
        k = np.asarray(rng.normal(size=(tokens, hkv, 256)), dtype=np.float32)
        v = np.asarray(rng.normal(size=(tokens, hkv, 256)), dtype=np.float32)
        qg[0, 0, 1, 0] = v[0, 0, 0] = -0.0
        qkv = np.concatenate((qg.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)), axis=1)
        qw, kw = [np.asarray(rng.normal(size=256), dtype=np.float32) for _ in range(2)]
        coeff = np.asarray(rng.normal(size=(3, tokens, 64)), dtype=np.float32)
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
            self.assertEqual(grid, (min(jobs, cores),))
            cores = grid[0]
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
        return tuple(outputs), tuple(pointer.counts for pointer in pointers)

    def test_actual_body_math_and_asymmetric_head_ownership(self):
        for hq, hkv in ((4, 1), (6, 1), (24, 4), (1, 3)):
            with self.subTest(heads=(hq, hkv)):
                self._check_body(hq, hkv, grouped=True)

    def test_t1_head_boundaries_real_wrapper_to_body(self):
        package = ModuleType("test_split_qkv_head_boundary")
        package.__path__ = [str(OP_DIR)]
        sys.modules[package.__name__] = package
        contracts = _load_contracts(package.__name__)
        for hq, hkv in ((2, 1), (1, 2), (40, 1), (1, 40)):
            with self.subTest(heads=(hq, hkv)):
                helper = mock.Mock(return_value=196608)
                wrapper, m1, m2 = _wrapper_namespace(contracts, helper)
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
    contracts: ClassVar[SimpleNamespace]
    envs: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls):
        package = ModuleType("test_split_qkv_mrope")
        package.__path__ = [str(OP_DIR)]
        sys.modules[package.__name__] = package
        cls.contracts = _load_contracts(package.__name__)
        cls.envs = _load_source_module("test_split_qkv_mrope_envs", ENV_PATH)

    def _run(
        self,
        *,
        tokens=4096,
        q=8,
        k=2,
        D=256,
        R=64,
        gate=True,
        bias=False,
        dtype="bfloat16",
        eps=1e-6,
        sections=None,
        interleaved=True,
        cores=40,
        capacity=196608,
        requested=2,
        contiguous=True,
        resource=None,
    ):
        capacity_query = mock.Mock(return_value=capacity)
        namespace, m1, m2 = _wrapper_namespace(self.contracts, capacity_query, requested=requested, cores=cores)
        if resource is not None:
            namespace["working_set"] = resource
        qb = _tensor((D,), dtype=dtype) if bias else None
        output = namespace["triton_split_qkv_rmsnorm_mrope"](
            _tensor((tokens, (q * (1 + int(gate)) + 2 * k) * D), dtype=dtype, contiguous=contiguous),
            _tensor((D,), dtype=dtype),
            _tensor((D,), dtype=dtype),
            _tensor((3, tokens, R), dtype=dtype),
            q,
            k,
            D,
            eps,
            ([11, 11, 10] if R == 64 else [R // 2, 0, 0]) if sections is None else sections,
            interleaved,
            rope_dim=R,
            q_bias=qb,
            k_bias=qb,
            has_gate=gate,
        )
        selected = m2 if m2.calls else m1
        return selected.calls[0][1][-1], namespace, m1, m2, capacity_query, output

    def test_routes_keep_measured_anchors_and_supported_families(self):
        cases: tuple[tuple[dict[str, Any], int], ...] = (
            ({}, 4),
            ({"k": 8}, 2),
            ({"q": 12, "k": 3}, 2),
            ({"q": 24, "k": 4, "tokens": 1024}, 2),
            ({"q": 6, "k": 1}, 4),
            ({"q": 4, "k": 1, "tokens": 8192}, 4),
            ({"gate": False, "D": 128, "R": 128, "sections": [24, 20, 20]}, 2),
            ({"dtype": "float16"}, 2),
            ({"bias": True}, 2),
            ({"dtype": "float16", "bias": True}, 2),
            ({"D": 128}, 2),
            ({"R": 128}, 2),
            ({"interleaved": False}, 2),
            ({"sections": [0, 16, 16]}, 2),
            ({"eps": 2e-6}, 2),
            ({"cores": 48}, 2),
            ({"q": 48, "capacity": 1048576}, 2),
            ({"k": 41, "capacity": 1048576}, 2),
            ({"q": 2, "k": 1, "D": 3, "R": 2, "gate": False}, 2),
            ({"q": 2, "k": 1, "D": 129, "R": 66, "gate": False}, 2),
        )
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                route, _, m1, m2, query, output = self._run(**kwargs)
                self.assertEqual(route, expected)
                self.assertEqual(m1.calls, [])
                self.assertEqual(len(m2.calls), 1)
                self.assertEqual(len(m2.calls[0][1]) + len(m2.options[0]), 31)
                self.assertEqual(len(output), 4)
                query.assert_called_once_with()

    def test_m2_performance_domain_and_accepted_n3_migration(self):
        # This is an empirical policy: N3 gives up measured M2 speed for ordinary M1.
        # Resource feasibility and the BASE comparison are separate qualifications.
        for D, expected in ((255, 2), (256, 2), (257, 1), (512, 1)):
            with self.subTest(D=D):
                calls: list[int] = []

                def resource(M, *args, calls=calls):
                    calls.append(M)
                    return self.contracts.working_set(M, *args)

                route, _, m1, m2, query, output = self._run(
                    q=2,
                    k=1,
                    D=D,
                    R=192,
                    gate=False,
                    sections=[32, 32, 32],
                    resource=resource,
                )
                self.assertEqual(route, expected)
                self.assertEqual(calls, [2] if expected == 2 else [])
                self.assertEqual(len(m1.calls), int(expected == 1))
                self.assertEqual(len(m2.calls), int(expected == 2))
                self.assertEqual(len(output), 4)
                query.assert_called_once_with()
        self.assertEqual(self._run(q=9, k=1)[0], 4)
        # D: restored FP16/paired-bias/non-interleaved M2 has device numerics evidence.
        self.assertEqual(self._run(dtype="float16", bias=True, R=192, interleaved=False, sections=[32, 32, 32])[0], 2)

    def test_m4_independent_then_m2_and_kv_guard(self):
        calls = []

        def m4_only(M, *args):
            calls.append(M)
            self.assertEqual(M, 4)
            return self.contracts.working_set(M, *args)

        self.assertEqual(self._run(resource=m4_only)[0], 4)
        self.assertEqual(calls, [4])
        for kwargs in (
            {"k": 8},
            {"q": 12, "k": 3},
            {"dtype": "float16"},
            {"bias": True},
            {"cores": 48},
            {"eps": 2e-6},
            {"sections": [0, 16, 16]},
            {"interleaved": False},
        ):
            calls = []

            def m2_only(M, *args, calls=calls):
                calls.append(M)
                self.assertEqual(M, 2)
                return self.contracts.working_set(M, *args)

            self.assertEqual(self._run(resource=m2_only, **kwargs)[0], 2)
            self.assertEqual(calls, [2])
        calls = []

        def exceeds(M, *_args):
            calls.append(M)
            return 2000000

        self.assertEqual(self._run(resource=exceeds)[0], 1)
        self.assertEqual(calls, [4, 2])

    def test_resource_exact_fit_source_cap_and_invalid_budgets(self):
        for cap, expected in ((181556, 2), (181557, 4)):
            self.assertEqual(self._run(capacity=cap)[0], expected)
        for cap, expected in ((67424, 1), (67425, 2)):
            self.assertEqual(self._run(gate=False, D=128, R=128, capacity=cap)[0], expected)
        self.assertEqual(self._run(capacity=1048576, resource=lambda M, *args: 196608 if M == 4 else 92956)[0], 2)
        for bad in (None, 1.0, True, 0, -1, RuntimeError("resource defect")):
            capacity = mock.Mock(return_value=196608)
            namespace, m1, m2 = _wrapper_namespace(self.contracts, capacity)

            def broken(*_args, bad=bad):
                if isinstance(bad, Exception):
                    raise bad
                return bad

            namespace["working_set"] = broken
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                namespace["triton_split_qkv_rmsnorm_mrope"](
                    _tensor((4096, 5120)),
                    _tensor((256,)),
                    _tensor((256,)),
                    _tensor((3, 4096, 64)),
                    8,
                    2,
                    256,
                    1e-6,
                    [11, 11, 10],
                    True,
                    rope_dim=64,
                    has_gate=True,
                )
            self.assertEqual(m1.calls + m2.calls, [])

    def test_scalar_contract_and_early_return_boundaries(self):
        kwargs: dict[str, Any]
        for kwargs in (
            {"tokens": 0},
            {"tokens": 959},
            {"q": 16},
            {"dtype": "float32"},
            {"capacity": None},
            {"capacity": 65535},
            {"capacity": 1048577},
            {"contiguous": False},
        ):
            self.assertEqual(self._run(**kwargs)[0], 1)
        for kwargs in (
            {"cores": 40.0},
            {"cores": True},
            {"cores": 0},
            {"tokens": -1},
            {"sections": [11, 11, 10, 0]},
            {"sections": [True, 11, 20]},
            {"eps": 1},
            {"eps": float("inf")},
            {"R": 63},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self._run(**kwargs)

    def test_live_scalar_validation_patch_target(self):
        for eps in (1e-6, float("inf")):
            with self.subTest(eps=eps):
                capacity = mock.Mock(return_value=196608)
                namespace, _, _ = _wrapper_namespace(self.contracts, capacity)
                self.assertIs(namespace["_validate_scalars"].__globals__["math"], math)
                validator = mock.Mock(wraps=namespace["_validate_scalars"])
                namespace["_validate_scalars"] = validator
                args = (
                    _tensor((1024, 3584)),
                    _tensor((256,)),
                    _tensor((256,)),
                    _tensor((3, 1024, 64)),
                    6,
                    1,
                    256,
                    eps,
                    [11, 11, 10],
                    True,
                )
                if math.isfinite(eps):
                    namespace["triton_split_qkv_rmsnorm_mrope"](*args, rope_dim=64, has_gate=True)
                    capacity.assert_called_once_with()
                else:
                    with self.assertRaisesRegex(ValueError, "^eps must be a finite positive float$"):
                        namespace["triton_split_qkv_rmsnorm_mrope"](*args, rope_dim=64, has_gate=True)
                    capacity.assert_not_called()
                validator.assert_called_once()
                self.assertEqual(len(validator.call_args.kwargs), 8)

    def test_production_single_selection_and_real_import_wiring(self):
        tree = ast.parse(OP_PATH.read_text())
        imports = {
            alias.name: node.module for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        self.assertNotIn("ac_input_validation", OP_PATH.read_text())
        self.assertFalse((OP_DIR / "ac_input_validation.py").exists())
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"_validate_scalars", "_layout_valid"} <= functions)
        self.assertEqual(imports["working_set"], "ac_working_set")
        self.assertFalse((OP_DIR / "ac_ub_demand.py").exists())
        source = OP_PATH.read_text()
        for removed in ("ShapeKey", "DispatchDecision", "select_dispatch", "ac_dispatch_policy"):
            self.assertNotIn(removed, source)
        config = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_M2_KERNEL" for target in node.targets)
        )
        expected = ast.parse(
            '_M2_KERNEL = triton.autotune(configs=[triton.Config({"multibuffer": False}, '
            "num_stages=1, num_warps=32)], key=[])(split_qkv_rmsnorm_mrope_kernel)"
        ).body[0]
        self.assertEqual(ast.dump(config), ast.dump(expected))
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "triton_split_qkv_rmsnorm_mrope"
        )
        self.assertEqual(
            sum(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_select_tiling"
                for node in ast.walk(wrapper)
            ),
            1,
        )
        self.assertEqual(
            sum(
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "num_tasks" for t in node.targets)
                for node in ast.walk(wrapper)
            ),
            1,
        )

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
        self.assertIn("requested_m = envs." + ENV_NAME, OP_PATH.read_text(encoding="utf-8"))

    def test_actual_env_getter_default_and_explicit_two_both_run_auto(self):
        for raw in (None, "2"):
            env = {} if raw is None else {ENV_NAME: raw}
            with self.subTest(raw=raw), mock.patch.dict(os.environ, env, clear=True):
                requested = getattr(self.envs, ENV_NAME)
                namespace, m1, m2 = _wrapper_namespace(self.contracts, lambda: 196608, requested=requested)
                namespace["triton_split_qkv_rmsnorm_mrope"](
                    _tensor((960, 2560)),
                    _tensor((256,)),
                    _tensor((256,)),
                    _tensor((3, 960, 64)),
                    4,
                    1,
                    256,
                    1e-6,
                    [11, 11, 10],
                    True,
                    rope_dim=64,
                    has_gate=True,
                )
                self.assertEqual(requested, 2)
                self.assertEqual(m1.calls, [])
                self.assertEqual(len(m2.calls), 1)
                self.assertEqual(m2.calls[0][1][-1], 4)

    def test_capacity_uses_public_byte_helper_without_backend_probe(self):
        helper = mock.Mock(return_value=196608)
        namespace, _, _ = _wrapper_namespace(self.contracts, helper)
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
        for properties, expected_pair_m in (({}, 4), ({"ub_size": 64 * 1024}, 1)):
            helper = self._shared_ub_helper({"num_aicore": 20, "num_vectorcore": 40, **properties})
            for tokens in (1, 4, 1024):
                with self.subTest(properties=properties, tokens=tokens):
                    namespace, m1, m2 = _wrapper_namespace(self.contracts, helper.get_ub_size_bytes)
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
                    selected, other = (m2, m1) if expected in (2, 4) else (m1, m2)
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
                    namespace, m1, m2 = _wrapper_namespace(self.contracts, capacity, requested=requested)
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
        self.assertEqual(ast.unparse(m2.test), "BLOCK_M == 2 or BLOCK_M == 4")
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
        allowed_m = next(
            cast(ast.Call, node.value).args[0]
            for node in assertions
            if isinstance(node, ast.Expr)
            and ast.unparse(cast(ast.Call, node.value).args[0]) == "(BLOCK_M == 1 or BLOCK_M == 2) or BLOCK_M == 4"
        )
        self.assertTrue(all(len(node.values) == 2 for node in ast.walk(allowed_m) if isinstance(node, ast.BoolOp)))
        for condition, scope in (
            ("(BLOCK_M == 1 or BLOCK_M == 2) or BLOCK_M == 4", kernel.body),
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

    def test_kernel_contains_m1_and_bounded_experimental_m2_m4_with_row_tail(self):
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

    def test_layout_checks_without_identity_strings(self):
        namespace, _, _ = _wrapper_namespace(self.contracts, lambda: 196608)
        tensors = [_tensor((1024, 2048)), _tensor((256,)), _tensor((256,)), _tensor((3, 1024, 64)), None, None]

        def valid():
            return namespace["_layout_valid"](*tensors, 1024, 1536, 0, 256, 256, 64)

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
            (960, 6, 1, 4),
            (961, 24, 4, 2),
            (1024, 16, 4, 1),
        ):
            helper = mock.Mock(return_value=196608)
            namespace, m1, m2 = _wrapper_namespace(self.contracts, helper)
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
            selected = m1 if expected == 1 else m2
            other = m2 if expected == 1 else m1
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
        namespace, m1, m2 = _wrapper_namespace(self.contracts, helper, requested=1)
        namespace["_validate_scalars"] = mock.Mock(side_effect=AssertionError("must not validate"))
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
                    namespace, m1, m2 = _wrapper_namespace(self.contracts, helper, requested)
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
        namespace, _, _ = _wrapper_namespace(self.contracts, mock.Mock(return_value=196608))
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
                namespace, m1, m2 = _wrapper_namespace(self.contracts, helper)
                args = _t1_args(hq, hkv)
                self.assertTrue(
                    namespace["_layout_valid"](*args[:4], None, None, 1, hq * 256, hq * 256, hkv * 256, 256, 64)
                )
                layout = mock.Mock(wraps=namespace["_layout_valid"])
                namespace["_layout_valid"] = layout
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
                namespace, m1, m2 = _wrapper_namespace(self.contracts, helper, cores=cores)
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
        namespace, m1, m2 = _wrapper_namespace(self.contracts, mock.Mock(side_effect=AssertionError("must not probe")))
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
