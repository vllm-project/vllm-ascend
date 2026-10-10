"""Source-level checks for the PR1 wrapper boundary."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LAYER_NORM = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated.py"
DISPATCH = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated_dispatch.py"


class RouteSourceTests(unittest.TestCase):
    def test_sources_parse_and_exclude_wide_path_symbols(self):
        layer_source = LAYER_NORM.read_text()
        dispatch_source = DISPATCH.read_text()
        ast.parse(layer_source)
        ast.parse(dispatch_source)
        self.assertNotIn("c2_", layer_source.lower())
        self.assertNotIn("c2_", dispatch_source.lower())
        self.assertNotIn("compile_target", layer_source.lower())
        self.assertNotIn("try_get_compile_target", layer_source)

    def test_device_name_and_dtype_allowlists_are_absent(self):
        source = LAYER_NORM.read_text()
        self.assertNotIn("_PR1_QUALIFIED_DEVICE_NAMES", source)
        self.assertNotIn("_is_pr1_dtype", source)
        self.assertNotIn("try_get_vectorcore_num", source)
        self.assertIn("get_vectorcore_num", source)
        self.assertIn("get_ub_size_bytes", source)
        self.assertNotIn("qualified=qualified", source)


class _FakeTensor:
    _ELEMENT_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4}

    def __init__(self, shape, dtype, device_type="npu", device_index=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device_type, index=device_index)

    def stride(self, dim):
        return 1 if dim == -1 else self.shape[-1]

    def element_size(self):
        return self._ELEMENT_SIZE[self.dtype]


class _FakeKernel:
    def __init__(self, name, launches):
        self.name = name
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((self.name, grid, args, kwargs))

        return launch


def _load_layernorm_with_fakes():
    """Load the public wrapper with only stdlib fake torch/Triton modules."""
    launches: list[tuple[Any, ...]] = []
    state: dict[str, Any] = {
        "device_name": "UnknownAscendModel",
        "vector_cores": 40,
        "name_calls": 0,
        "getter_calls": 0,
        "ub_size": 196608,
        "ub_getter_calls": 0,
    }

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    class _Npu:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(index):
            state["name_calls"] += 1
            return state["device_name"]

    fake_torch.npu = _Npu()
    fake_torch.empty_like = lambda x: _FakeTensor(x.shape, x.dtype, x.device.type, x.device.index)
    fake_torch.empty = lambda shape, dtype, device: _FakeTensor(shape, dtype, device.type, device.index)

    fake_vllm: Any = types.ModuleType("vllm")
    fake_vllm_triton: Any = types.ModuleType("vllm.triton_utils")
    # Python 3.12 evaluates the kernel's tl.constexpr annotations at import time.
    fake_vllm_triton.tl = types.SimpleNamespace(float32="float32", constexpr=object)

    class _Triton:
        @staticmethod
        def heuristics(_config):
            return lambda function: function

        @staticmethod
        def jit(**_kwargs):
            return lambda function: _FakeKernel(function.__name__, launches)

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

        @staticmethod
        def next_power_of_2(value):
            return 1 << (value - 1).bit_length()

    fake_vllm_triton.triton = _Triton
    fake_vllm.triton_utils = fake_vllm_triton

    fake_ascend: Any = types.ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_ops: Any = types.ModuleType("vllm_ascend.ops")
    fake_ops.__path__ = []
    fake_triton_pkg: Any = types.ModuleType("vllm_ascend.ops.triton")
    fake_triton_pkg.__path__ = []
    fake_utils: Any = types.ModuleType("vllm_ascend.ops.triton.triton_utils")

    def get_vectorcore_num():
        state["getter_calls"] += 1
        assert state["vector_cores"] is not None, "Device properties not initialized."
        return state["vector_cores"]

    fake_utils.get_vectorcore_num = get_vectorcore_num

    def get_ub_size_bytes():
        state["ub_getter_calls"] += 1
        return state["ub_size"]

    fake_utils.get_ub_size_bytes = get_ub_size_bytes
    fake_triton_pkg.triton_utils = fake_utils
    fake_ops.triton = fake_triton_pkg
    fake_ascend.ops = fake_ops

    dispatch_name = "vllm_ascend.ops.triton.layernorm_gated_dispatch"
    dispatch_spec = importlib.util.spec_from_file_location(dispatch_name, DISPATCH)
    assert dispatch_spec is not None and dispatch_spec.loader is not None
    dispatch = importlib.util.module_from_spec(dispatch_spec)
    layer_name = "vllm_ascend.ops.triton.layernorm_gated"
    layer_spec = importlib.util.spec_from_file_location(layer_name, LAYER_NORM)
    assert layer_spec is not None and layer_spec.loader is not None
    layer = importlib.util.module_from_spec(layer_spec)
    saved = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "vllm",
            "vllm.triton_utils",
            "vllm_ascend",
            "vllm_ascend.ops",
            "vllm_ascend.ops.triton",
            "vllm_ascend.ops.triton.triton_utils",
            dispatch_name,
            layer_name,
        )
    }
    sys.modules.update(
        {
            "torch": fake_torch,
            "vllm": fake_vllm,
            "vllm.triton_utils": fake_vllm_triton,
            "vllm_ascend": fake_ascend,
            "vllm_ascend.ops": fake_ops,
            "vllm_ascend.ops.triton": fake_triton_pkg,
            "vllm_ascend.ops.triton.triton_utils": fake_utils,
            dispatch_name: dispatch,
            layer_name: layer,
        }
    )
    try:
        dispatch_spec.loader.exec_module(dispatch)
        layer_spec.loader.exec_module(layer)
        return layer, state, launches, saved
    except Exception:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        raise


def _unload_layernorm_fakes(saved):
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


class WrapperRouteTests(unittest.TestCase):
    def test_grouped_baseline_launch_needs_no_initialized_resource_getters(self):
        layer, state, launches, saved = _load_layernorm_with_fakes()
        state["vector_cores"] = None
        state["ub_size"] = None
        try:
            for rows, groups in ((65, 2), (65536, 4)):
                for width in (63, 128, 192, 512, 513):
                    for rms, gate_before in ((False, False), (True, True)):
                        with self.subTest(rows=rows, groups=groups, width=width, rms=rms):
                            columns = width * groups
                            x = _FakeTensor((rows, columns), "bfloat16")
                            weight = _FakeTensor((columns,), "bfloat16")
                            bias = None if rms else _FakeTensor((columns,), "bfloat16")
                            z = _FakeTensor((rows, columns), "bfloat16")
                            out = _FakeTensor((rows, columns), "bfloat16")
                            result = layer.layer_norm_fwd_npu(
                                x,
                                weight,
                                bias,
                                1e-6,
                                z=z,
                                out=out,
                                group_size=width,
                                norm_before_gate=not gate_before,
                                is_rms_norm=rms,
                            )
                            name, grid, args, kwargs = launches[-1]
                            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
                            self.assertEqual(grid, ((rows + 63) // 64, groups))
                            self.assertEqual(len(args), 13)
                            self.assertEqual(args[7:13], (columns, columns, columns, rows, width, 1e-6))
                            self.assertEqual(
                                (kwargs["BLOCK_M"], kwargs["BLOCK_N"]), (64, 1 << (width - 1).bit_length())
                            )
                            self.assertEqual(kwargs["NORM_BEFORE_GATE"], not gate_before)
                            self.assertEqual(kwargs["IS_RMS_NORM"], rms)
                            self.assertIs(result[0], out)
                            self.assertIs(args[0], x)
                            self.assertIs(args[1], out)
                            self.assertIs(args[2], weight)
                            self.assertIs(args[3], bias)
                            self.assertIs(args[4], z)
                            self.assertIs(args[5], result[1])
                            self.assertIs(args[6], result[2])
                            if rms:
                                self.assertIsNone(result[1])
                            else:
                                self.assertEqual(result[1].shape, (groups * rows,))
                            self.assertEqual(result[2].shape, (groups * rows,))
            self.assertEqual(state["getter_calls"], 0)
            self.assertEqual(state["ub_getter_calls"], 0)
        finally:
            _unload_layernorm_fakes(saved)

    def test_public_wrapper_launch_contract_and_fallbacks(self):
        layer, state, launches, saved = _load_layernorm_with_fakes()
        try:

            def call(
                rows,
                columns=128,
                dtype="bfloat16",
                group_size=None,
                out=None,
                bias=True,
                z=False,
                norm_before_gate=True,
                is_rms_norm=False,
                device_type="npu",
            ):
                x = _FakeTensor((rows, columns), dtype, device_type)
                weight = _FakeTensor((columns,), dtype, device_type)
                bias = _FakeTensor((columns,), dtype, device_type) if bias else None
                z = _FakeTensor((rows, columns), dtype, device_type) if z else None
                return layer.layer_norm_fwd_npu(
                    x,
                    weight,
                    bias,
                    1e-5,
                    z=z,
                    out=out,
                    group_size=group_size,
                    norm_before_gate=norm_before_gate,
                    is_rms_norm=is_rms_norm,
                )

            caller_out = _FakeTensor((288, 128), "bfloat16")
            returned = call(288, out=caller_out)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
            self.assertEqual(grid, (18, 1))
            self.assertEqual(len(args), 13)
            self.assertEqual((kwargs["BLOCK_M"], kwargs["BLOCK_N"]), (16, 128))
            self.assertIs(returned[0], caller_out)
            self.assertIs(returned[0], args[1])
            self.assertEqual(returned[1].shape, (288,))
            self.assertEqual(returned[2].shape, (288,))
            self.assertIs(returned[1], args[5])
            self.assertIs(returned[2], args[6])

            call(288)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
            self.assertEqual(grid, (18, 1))
            self.assertEqual(len(args), 13)
            self.assertEqual((kwargs["BLOCK_M"], kwargs["BLOCK_N"]), (16, 128))
            self.assertEqual(kwargs["NORM_BEFORE_GATE"], True)
            self.assertEqual(kwargs["IS_RMS_NORM"], False)

            rms_result = call(
                288,
                bias=False,
                z=True,
                norm_before_gate=False,
                is_rms_norm=True,
            )
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
            self.assertIsNone(args[3])
            self.assertIsNotNone(args[4])
            self.assertIsNone(rms_result[1])
            self.assertEqual(rms_result[2].shape, (288,))
            self.assertIs(rms_result[2], args[6])
            self.assertEqual(kwargs["NORM_BEFORE_GATE"], False)
            self.assertEqual(kwargs["IS_RMS_NORM"], True)

            call(639)
            self.assertEqual(launches[-1][0], "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(launches[-1][1], (20,))
            self.assertEqual(launches[-1][3]["BLOCK_M"], 32)

            call(289)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(grid, (10,))
            self.assertEqual(len(args), 14)
            self.assertEqual(args[13], 10)
            self.assertEqual(kwargs["BLOCK_M"], 32)

            call(640)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(grid, (20,))
            self.assertEqual(len(args), 14)
            self.assertEqual(args[13], 20)
            self.assertEqual(kwargs["BLOCK_M"], 32)

            call(20449)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(grid, (40,))
            self.assertEqual(len(args), 14)
            self.assertEqual(args[13], 640)
            self.assertEqual(kwargs["BLOCK_M"], 32)

            call(2048, columns=256, group_size=128)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
            self.assertEqual(grid, (32, 2))
            self.assertEqual(kwargs["BLOCK_M"], 64)

            call(287, bias=False, z=True)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
            self.assertEqual(grid, (18, 1))
            self.assertEqual(kwargs["BLOCK_M"], 16)
            self.assertIsNone(args[3])
            self.assertIsNotNone(args[4])
            self.assertEqual(args[5].shape, (287,))
            self.assertEqual(args[6].shape, (287,))

            # NPU dtype and product name do not gate the experimental route.
            before_name_calls = state["name_calls"]
            before_getter_calls = state["getter_calls"]
            call(20449, dtype="float32")
            self.assertEqual(launches[-1][0], "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(launches[-1][3]["BLOCK_M"], 32)
            self.assertEqual(state["name_calls"], before_name_calls)
            self.assertEqual(state["getter_calls"], before_getter_calls + 1)

            state["device_name"] = "AnotherUnknownAscendModel"
            call(20449)
            self.assertEqual(launches[-1][0], "_layer_norm_fwd_persistent_hoist_kernel_npu")
            self.assertEqual(state["name_calls"], before_name_calls)

            call(64, columns=256)
            self.assertEqual(launches[-1][3]["BLOCK_M"], 16)
            self.assertEqual(launches[-1][3]["BLOCK_N"], 256)
            self.assertEqual(state["ub_getter_calls"], 1)

            for group_size, block_n in (
                (129, 256),
                (192, 256),
                (256, 256),
                (257, 512),
                (384, 512),
                (512, 512),
            ):
                call(65, columns=group_size)
                name, grid, args, kwargs = launches[-1]
                self.assertEqual(name, "_layer_norm_fwd_1pass_kernel_npu")
                self.assertEqual((grid, args[10], args[11]), ((5, 1), 65, group_size))
                self.assertEqual(kwargs["BLOCK_M"], 16)
                self.assertEqual(kwargs["BLOCK_N"], block_n)

            # Routing uses per-group width, not the total tensor width.
            before_ub_calls = state["ub_getter_calls"]
            call(65, columns=384, group_size=128)
            self.assertEqual(launches[-1][1], (2, 3))
            self.assertEqual(launches[-1][2][11], 128)
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            self.assertEqual(launches[-1][3]["BLOCK_N"], 128)
            self.assertEqual(state["ub_getter_calls"], before_ub_calls)

            call(65, columns=384, group_size=192)
            self.assertEqual(launches[-1][1], (2, 2))
            self.assertEqual(launches[-1][2][11], 192)
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            self.assertEqual(launches[-1][3]["BLOCK_N"], 256)
            self.assertEqual(state["ub_getter_calls"], before_ub_calls)

            for ub_size in (196607, None):
                state["ub_size"] = ub_size
                call(65, columns=192)
                self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
                self.assertEqual(launches[-1][3]["BLOCK_N"], 256)

            state["ub_size"] = 196608
            before_ub_calls = state["ub_getter_calls"]
            call(65, columns=513)
            self.assertEqual(launches[-1][1], (2, 1))
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            self.assertEqual(launches[-1][3]["BLOCK_N"], 1024)
            self.assertEqual(state["ub_getter_calls"], before_ub_calls)

            state["vector_cores"] = None
            with self.assertRaisesRegex(AssertionError, "Device properties not initialized"):
                call(65, columns=256)
            self.assertEqual(state["ub_getter_calls"], before_ub_calls)
            state["vector_cores"] = 40

            before_ub_calls = state["ub_getter_calls"]
            call(65, columns=256, device_type="cpu")
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            self.assertEqual(launches[-1][3]["BLOCK_N"], 256)
            self.assertEqual(state["ub_getter_calls"], before_ub_calls)

            state["vector_cores"] = None
            with self.assertRaisesRegex(AssertionError, "Device properties not initialized"):
                call(288)

            for device_type in ("cpu", "cuda"):
                before_getter_calls = state["getter_calls"]
                state["vector_cores"] = 40
                call(289, device_type=device_type)
                self.assertEqual(launches[-1][0], "_layer_norm_fwd_1pass_kernel_npu")
                self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
                self.assertEqual(state["getter_calls"], before_getter_calls)
                self.assertEqual(state["name_calls"], 0)
        finally:
            _unload_layernorm_fakes(saved)


if __name__ == "__main__":
    unittest.main()
