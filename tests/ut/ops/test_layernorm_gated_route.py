"""Source-level checks for the narrow PR1 wrapper boundary."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


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

    def test_exact_device_targets_and_dtype_gate_are_present(self):
        source = LAYER_NORM.read_text()
        self.assertIn('"Ascend910B3"', source)
        self.assertIn('"Ascend910_9382"', source)
        self.assertIn("torch.float16", source)
        self.assertIn("torch.bfloat16", source)
        self.assertIn("try_get_vectorcore_num", source)
        self.assertIn("qualified=qualified", source)


class _FakeTensor:
    _ELEMENT_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4}

    def __init__(self, shape, dtype, device_index=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(index=device_index)

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
    launches = []
    state = {"device_name": "Ascend910B3", "vector_cores": 40, "name_calls": 0}

    fake_torch = types.ModuleType("torch")
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
    fake_torch.empty_like = lambda x: _FakeTensor(x.shape, x.dtype, x.device.index)
    fake_torch.empty = lambda shape, dtype, device: _FakeTensor(shape, dtype, device.index)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm_triton = types.ModuleType("vllm.triton_utils")
    fake_vllm_triton.tl = types.SimpleNamespace(float32="float32")

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

    fake_ascend = types.ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_ops = types.ModuleType("vllm_ascend.ops")
    fake_ops.__path__ = []
    fake_triton_pkg = types.ModuleType("vllm_ascend.ops.triton")
    fake_triton_pkg.__path__ = []
    fake_utils = types.ModuleType("vllm_ascend.ops.triton.triton_utils")
    fake_utils.try_get_vectorcore_num = lambda: state["vector_cores"]
    fake_triton_pkg.triton_utils = fake_utils
    fake_ops.triton = fake_triton_pkg
    fake_ascend.ops = fake_ops

    dispatch_name = "vllm_ascend.ops.triton.layernorm_gated_dispatch"
    dispatch_spec = importlib.util.spec_from_file_location(dispatch_name, DISPATCH)
    dispatch = importlib.util.module_from_spec(dispatch_spec)
    layer_name = "vllm_ascend.ops.triton.layernorm_gated"
    layer_spec = importlib.util.spec_from_file_location(layer_name, LAYER_NORM)
    layer = importlib.util.module_from_spec(layer_spec)
    saved = {name: sys.modules.get(name) for name in (
        "torch", "vllm", "vllm.triton_utils", "vllm_ascend", "vllm_ascend.ops",
        "vllm_ascend.ops.triton", "vllm_ascend.ops.triton.triton_utils",
        dispatch_name, layer_name,
    )}
    sys.modules.update({
        "torch": fake_torch,
        "vllm": fake_vllm,
        "vllm.triton_utils": fake_vllm_triton,
        "vllm_ascend": fake_ascend,
        "vllm_ascend.ops": fake_ops,
        "vllm_ascend.ops.triton": fake_triton_pkg,
        "vllm_ascend.ops.triton.triton_utils": fake_utils,
        dispatch_name: dispatch,
        layer_name: layer,
    })
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
    def test_public_wrapper_launch_contract_and_fallbacks(self):
        layer, state, launches, saved = _load_layernorm_with_fakes()
        try:
            def call(rows, columns=128, dtype="bfloat16", group_size=None, out=None):
                x = _FakeTensor((rows, columns), dtype)
                weight = _FakeTensor((columns,), dtype)
                bias = _FakeTensor((columns,), dtype)
                return layer.layer_norm_fwd_npu(
                    x, weight, bias, 1e-5, out=out, group_size=group_size
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

            call(289)
            name, grid, args, kwargs = launches[-1]
            self.assertEqual(name, "_layer_norm_fwd_persistent_kernel_npu")
            self.assertEqual(grid, (10,))
            self.assertEqual(len(args), 15)
            self.assertEqual((args[13], args[14]), (10, 1))
            self.assertEqual((kwargs["BLOCK_M"], kwargs["BLOCK_N"]), (32, 128))

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
            self.assertEqual(grid, (64, 2))
            self.assertEqual(kwargs["BLOCK_M"], 32)

            # FP32, wide-N, and an unknown device all retain BASE64.  The
            # unknown-name probe is cache-cleared to model a distinct device.
            before_name_calls = state["name_calls"]
            call(20449, dtype="float32")
            self.assertEqual(launches[-1][0], "_layer_norm_fwd_1pass_kernel_npu")
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            self.assertEqual(state["name_calls"], before_name_calls)
            call(64, columns=256)
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
            layer._is_pr1_device_name_qualified.cache_clear()
            state["device_name"] = "Ascend910_9362"
            call(20449)
            self.assertEqual(launches[-1][3]["BLOCK_M"], 64)
        finally:
            _unload_layernorm_fakes(saved)


if __name__ == "__main__":
    unittest.main()
