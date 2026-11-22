# SPDX-License-Identifier: Apache-2.0
"""Regression tests for preserving ``scale_alg`` in MXFP8 norm-quant fusion."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from tests.ut.base import TestBase
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg

_CANN_DTYPE_E4M3FN = 24
_CANN_DTYPE_E5M2 = 23
_CANN_DTYPE_FP4_E2M1FN_X2 = 296

_DTYPE_TO_CANN_CODE = {
    torch.float8_e4m3fn: _CANN_DTYPE_E4M3FN,
    torch.float8_e5m2: _CANN_DTYPE_E5M2,
}
_CANN_CODE_TO_DTYPE = {
    _CANN_DTYPE_E4M3FN: torch.float8_e4m3fn,
    _CANN_DTYPE_E5M2: torch.float8_e5m2,
    _CANN_DTYPE_FP4_E2M1FN_X2: torch.uint8,
}

_TEST_LIB = torch.library.Library("norm_mx_fusion_test", "DEF")
_TEST_LIB.define("npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor, Tensor)")
_TEST_LIB.define("npu_rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor)")
_TEST_LIB.define(
    "npu_dynamic_mx_quant("
    "Tensor input, *, int axis=-1, str round_mode='rint', "
    "int dst_type=296, int block_size=32, "
    "int? scale_alg=0, float dst_type_max=0.0, float max_low_bound=0.0"
    ") -> (Tensor, Tensor)"
)
_TEST_LIB.define(
    "npu_add_rms_norm_dynamic_mx_quant("
    "Tensor x1, Tensor x2, Tensor gamma, *, Tensor? beta=None, "
    "float epsilon=1e-6, int scale_alg=0, str round_mode='rint', "
    "int dst_type=296"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)
_TEST_LIB.define(
    "npu_rms_norm_dynamic_mx_quant("
    "Tensor x, Tensor gamma, *, Tensor? beta=None, "
    "float epsilon=1e-6, int scale_alg=0, str round_mode='rint', "
    "int dst_type=296"
    ") -> (Tensor, Tensor, Tensor)"
)

_TEST_CPU_LIB = torch.library.Library("norm_mx_fusion_test", "IMPL", "CPU")
_TEST_META_LIB = torch.library.Library("norm_mx_fusion_test", "IMPL", "Meta")


def _add_rms_norm_impl(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
    residual = x1 + x2
    rstd = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
    return residual * gamma, rstd, residual


def _rms_norm_impl(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
    rstd = torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)
    return x * gamma, rstd


def _dynamic_mx_quant_impl(
    x: torch.Tensor,
    *,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = _CANN_DTYPE_FP4_E2M1FN_X2,
    block_size: int = 32,
    scale_alg: int | None = 0,
    dst_type_max: float = 0.0,
    max_low_bound: float = 0.0,
):
    eff_alg = 0 if scale_alg is None else int(scale_alg)
    torch_dtype = _CANN_CODE_TO_DTYPE.get(dst_type, torch.uint8)
    q = x.to(torch_dtype)
    scale = torch.full((x.shape[0], x.shape[1] // 64, 2), 127 + eff_alg, dtype=torch.uint8, device=x.device)
    return q, scale


def _add_rms_norm_dynamic_mx_quant_impl(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = _CANN_DTYPE_FP4_E2M1FN_X2,
):
    y, rstd, residual = _add_rms_norm_impl(x1, x2, gamma, epsilon)
    if beta is not None:
        y = y + beta
    q, scale = _dynamic_mx_quant_impl(y, dst_type=dst_type, scale_alg=scale_alg, round_mode=round_mode)
    return q, residual, scale, rstd


def _rms_norm_dynamic_mx_quant_impl(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    beta: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = _CANN_DTYPE_FP4_E2M1FN_X2,
):
    y, rstd = _rms_norm_impl(x, gamma, epsilon)
    if beta is not None:
        y = y + beta
    q, scale = _dynamic_mx_quant_impl(y, dst_type=dst_type, scale_alg=scale_alg, round_mode=round_mode)
    return q, scale, rstd


for _lib in (_TEST_CPU_LIB, _TEST_META_LIB):
    _lib.impl("npu_add_rms_norm", _add_rms_norm_impl)
    _lib.impl("npu_rms_norm", _rms_norm_impl)
    _lib.impl("npu_dynamic_mx_quant", _dynamic_mx_quant_impl)
    _lib.impl("npu_add_rms_norm_dynamic_mx_quant", _add_rms_norm_dynamic_mx_quant_impl)
    _lib.impl("npu_rms_norm_dynamic_mx_quant", _rms_norm_dynamic_mx_quant_impl)


class _CannDstTypeOpAdapter:
    """Adapt ``torch.dtype`` ``dst_type`` kwargs to CANN integer codes like ``torch_npu``."""

    def __init__(self, op_packet):
        self._op = op_packet

    def __call__(self, *args, **kwargs):
        if "dst_type" in kwargs and isinstance(kwargs["dst_type"], torch.dtype):
            kwargs = {**kwargs, "dst_type": _DTYPE_TO_CANN_CODE[kwargs["dst_type"]]}
        return self._op(*args, **kwargs)

    def __eq__(self, other):
        return other is self or other == self._op

    def __hash__(self):
        return hash(self._op)


class _TestNpuNamespace:
    npu_add_rms_norm = torch.ops.norm_mx_fusion_test.npu_add_rms_norm
    npu_rms_norm = torch.ops.norm_mx_fusion_test.npu_rms_norm
    npu_dynamic_mx_quant = _CannDstTypeOpAdapter(torch.ops.norm_mx_fusion_test.npu_dynamic_mx_quant)
    npu_add_rms_norm_dynamic_mx_quant = _CannDstTypeOpAdapter(
        torch.ops.norm_mx_fusion_test.npu_add_rms_norm_dynamic_mx_quant
    )
    npu_rms_norm_dynamic_mx_quant = _CannDstTypeOpAdapter(torch.ops.norm_mx_fusion_test.npu_rms_norm_dynamic_mx_quant)


def _normalize_fx(gm: fx.GraphModule) -> fx.GraphModule:
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _trace_add_rms_norm_mx_quant(
    *,
    eps: float = 1e-6,
    dst_type: torch.dtype | int | None = torch.float8_e4m3fn,
    quant_kwargs: dict | None = None,
) -> fx.GraphModule:
    extra = dict(quant_kwargs or {})
    if dst_type is not None:
        extra["dst_type"] = dst_type

    def fn(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor):
        out = torch.ops.npu.npu_add_rms_norm(x, r, w, eps)
        q = torch.ops.npu.npu_dynamic_mx_quant(out[0], **extra)
        return q[0], q[1], out[2]

    with FakeTensorMode():
        inputs = (
            torch.empty(2, 64, dtype=torch.bfloat16),
            torch.empty(2, 64, dtype=torch.bfloat16),
            torch.empty(64, dtype=torch.bfloat16),
        )
        gm = make_fx(fn)(*inputs)
    return _normalize_fx(gm)


def _trace_rms_norm_mx_quant(
    *,
    eps: float = 1e-6,
    dst_type: torch.dtype | int | None = torch.float8_e4m3fn,
    quant_kwargs: dict | None = None,
) -> fx.GraphModule:
    extra = dict(quant_kwargs or {})
    if dst_type is not None:
        extra["dst_type"] = dst_type

    def fn(x: torch.Tensor, w: torch.Tensor):
        out = torch.ops.npu.npu_rms_norm(x, w, eps)
        q = torch.ops.npu.npu_dynamic_mx_quant(out[0], **extra)
        return q[0], q[1]

    with FakeTensorMode():
        inputs = (
            torch.empty(2, 64, dtype=torch.bfloat16),
            torch.empty(64, dtype=torch.bfloat16),
        )
        gm = make_fx(fn)(*inputs)
    return _normalize_fx(gm)


def _call_nodes(gm: fx.GraphModule, op_packet) -> list[fx.Node]:
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and getattr(node.target, "overloadpacket", node.target) == op_packet
    ]


class TestMXFP8NormQuantFusionScaleAlg(TestBase):
    _MANAGED_MODULES = (
        "npugraph_ex",
        "vllm_ascend.compilation.passes.base_pattern",
        "vllm_ascend.compilation.passes.norm_quant_fusion_pass",
    )

    @staticmethod
    def _has_module(name: str) -> bool:
        if name in sys.modules:
            return sys.modules[name] is not None
        return importlib.util.find_spec(name) is not None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        missing = object()
        saved_modules = {name: sys.modules.get(name, missing) for name in cls._MANAGED_MODULES}
        passes_pkg = importlib.import_module("vllm_ascend.compilation.passes")
        saved_pkg_attrs = {
            attr: getattr(passes_pkg, attr)
            for attr in ("base_pattern", "norm_quant_fusion_pass")
            if hasattr(passes_pkg, attr)
        }

        def _restore_modules():
            for name, prev in saved_modules.items():
                if prev is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
            for attr in ("base_pattern", "norm_quant_fusion_pass"):
                if attr in saved_pkg_attrs:
                    setattr(passes_pkg, attr, saved_pkg_attrs[attr])
                elif hasattr(passes_pkg, attr):
                    delattr(passes_pkg, attr)

        cls.addClassCleanup(_restore_modules)
        if not cls._has_module("npugraph_ex") and not cls._has_module("torchair"):
            sys.modules["npugraph_ex"] = MagicMock()
        cls.base_pattern = importlib.import_module("vllm_ascend.compilation.passes.base_pattern")
        cls.norm_fusion = importlib.import_module("vllm_ascend.compilation.passes.norm_quant_fusion_pass")

    def setUp(self):
        super().setUp()
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            quant_config=None,
            device_config=SimpleNamespace(device="cpu"),
            compilation_config=SimpleNamespace(
                splitting_ops=[],
                use_inductor_graph_partition=False,
                pass_config=SimpleNamespace(),
            ),
        )

    def _register_mx_patterns(
        self,
        matcher: pm.PatternMatcherPass,
        *,
        scale_algs: tuple[int, ...] = (DynamicMXQuantScaleAlg.MAX_EXPONENT, DynamicMXQuantScaleAlg.CEIL),
        eps: float = 1e-6,
    ):
        for scale_alg in scale_algs:
            add_pat = self.norm_fusion.AddRMSNormDynamicMXQuantPattern(self.vllm_config, eps=eps, scale_alg=scale_alg)
            rms_pat = self.norm_fusion.RMSNormDynamicMXQuantPattern(self.vllm_config, eps=eps, scale_alg=scale_alg)
            with (
                patch.object(
                    add_pat,
                    "get_inputs",
                    return_value=[
                        torch.randn(2, 64, dtype=torch.bfloat16),
                        torch.randn(2, 64, dtype=torch.bfloat16),
                        torch.randn(64, dtype=torch.bfloat16),
                    ],
                ),
                patch.object(
                    rms_pat,
                    "get_inputs",
                    return_value=[
                        torch.randn(2, 64, dtype=torch.bfloat16),
                        torch.randn(64, dtype=torch.bfloat16),
                    ],
                ),
            ):
                add_pat.register(matcher)
                rms_pat.register(matcher)

    def _patched_ops(self):
        return patch.object(torch.ops, "npu", _TestNpuNamespace)

    def test_add_rms_norm_mx_quant_preserves_scale_alg_zero_and_one(self):
        matcher = pm.PatternMatcherPass(pass_name="test_add_mx")
        with (
            self._patched_ops(),
            patch.object(self.base_pattern, "_registered_patterns", set()),
            patch.object(self.base_pattern.nge, "register_replacement"),
        ):
            self._register_mx_patterns(matcher)

            for label, dst_type, quant_kwargs, expected_alg in (
                ("omitted_torch_dtype", torch.float8_e4m3fn, {}, 0),
                ("omitted_int_24", _CANN_DTYPE_E4M3FN, {}, 0),
                ("explicit_0", torch.float8_e4m3fn, {"scale_alg": 0}, 0),
                ("explicit_none", torch.float8_e4m3fn, {"scale_alg": None}, 0),
                ("explicit_ceil_1_torch_dtype", torch.float8_e4m3fn, {"scale_alg": 1}, 1),
                ("explicit_ceil_1_int_24", _CANN_DTYPE_E4M3FN, {"scale_alg": 1}, 1),
            ):
                with self.subTest(case=label):
                    gm = _trace_add_rms_norm_mx_quant(dst_type=dst_type, quant_kwargs=quant_kwargs)
                    pre_quant_nodes = _call_nodes(gm, torch.ops.npu.npu_dynamic_mx_quant)
                    self.assertEqual(len(pre_quant_nodes), 1)
                    self.assertEqual(pre_quant_nodes[0].kwargs.get("dst_type"), _CANN_DTYPE_E4M3FN)

                    x = torch.randn(2, 64, dtype=torch.bfloat16)
                    r = torch.randn(2, 64, dtype=torch.bfloat16)
                    w = torch.randn(64, dtype=torch.bfloat16)
                    before_q, before_s, before_res = gm(x, r, w)

                    self.assertEqual(matcher.apply(gm.graph), 1)
                    gm.recompile()

                    fused_nodes = _call_nodes(gm, torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant)
                    self.assertEqual(len(fused_nodes), 1)
                    self.assertEqual(fused_nodes[0].kwargs.get("dst_type"), _CANN_DTYPE_E4M3FN)
                    self.assertEqual(fused_nodes[0].kwargs.get("scale_alg", 0), expected_alg)

                    after_q, after_s, after_res = gm(x, r, w)
                    self.assertTrue(torch.equal(after_q, before_q))
                    self.assertTrue(torch.equal(after_s, before_s))
                    self.assertTrue(torch.equal(after_res, before_res))

    def test_rms_norm_mx_quant_preserves_scale_alg_zero_and_one(self):
        matcher = pm.PatternMatcherPass(pass_name="test_rms_mx")
        with (
            self._patched_ops(),
            patch.object(self.base_pattern, "_registered_patterns", set()),
            patch.object(self.base_pattern.nge, "register_replacement"),
        ):
            self._register_mx_patterns(matcher)

            for label, dst_type, quant_kwargs, expected_alg in (
                ("omitted_torch_dtype", torch.float8_e4m3fn, {}, 0),
                ("omitted_int_24", _CANN_DTYPE_E4M3FN, {}, 0),
                ("explicit_0", torch.float8_e4m3fn, {"scale_alg": 0}, 0),
                ("explicit_none", torch.float8_e4m3fn, {"scale_alg": None}, 0),
                ("explicit_ceil_1_torch_dtype", torch.float8_e4m3fn, {"scale_alg": 1}, 1),
                ("explicit_ceil_1_int_24", _CANN_DTYPE_E4M3FN, {"scale_alg": 1}, 1),
            ):
                with self.subTest(case=label):
                    gm = _trace_rms_norm_mx_quant(dst_type=dst_type, quant_kwargs=quant_kwargs)
                    pre_quant_nodes = _call_nodes(gm, torch.ops.npu.npu_dynamic_mx_quant)
                    self.assertEqual(len(pre_quant_nodes), 1)
                    self.assertEqual(pre_quant_nodes[0].kwargs.get("dst_type"), _CANN_DTYPE_E4M3FN)

                    x = torch.randn(2, 64, dtype=torch.bfloat16)
                    w = torch.randn(64, dtype=torch.bfloat16)
                    before_q, before_s = gm(x, w)

                    self.assertEqual(matcher.apply(gm.graph), 1)
                    gm.recompile()

                    fused_nodes = _call_nodes(gm, torch.ops.npu.npu_rms_norm_dynamic_mx_quant)
                    self.assertEqual(len(fused_nodes), 1)
                    self.assertEqual(fused_nodes[0].kwargs.get("dst_type"), _CANN_DTYPE_E4M3FN)
                    self.assertEqual(fused_nodes[0].kwargs.get("scale_alg", 0), expected_alg)

                    after_q, after_s = gm(x, w)
                    self.assertTrue(torch.equal(after_q, before_q))
                    self.assertTrue(torch.equal(after_s, before_s))

    def test_non_default_or_unsupported_mx_quant_kwargs_are_not_fused(self):
        matcher = pm.PatternMatcherPass(pass_name="test_mx_reject")
        with (
            self._patched_ops(),
            patch.object(self.base_pattern, "_registered_patterns", set()),
            patch.object(self.base_pattern.nge, "register_replacement"),
        ):
            self._register_mx_patterns(matcher)

            for label, dst_type, quant_kwargs in (
                ("unsupported_scale_alg_2", torch.float8_e4m3fn, {"scale_alg": 2}),
                ("non_default_round_mode", torch.float8_e4m3fn, {"scale_alg": 1, "round_mode": "floor"}),
                ("non_default_axis", torch.float8_e4m3fn, {"axis": 0}),
                ("non_default_block_size", torch.float8_e4m3fn, {"block_size": 64}),
                ("non_default_dst_type_max", torch.float8_e4m3fn, {"dst_type_max": 448.0}),
                ("non_default_max_low_bound", torch.float8_e4m3fn, {"max_low_bound": 1.0}),
                ("omitted_dst_type_fp4_default", None, {}),
                ("explicit_fp4_296", _CANN_DTYPE_FP4_E2M1FN_X2, {}),
                ("non_e4m3fn_e5m2_torch_dtype", torch.float8_e5m2, {"scale_alg": 1}),
                ("non_e4m3fn_e5m2_int_23", _CANN_DTYPE_E5M2, {"scale_alg": 1}),
            ):
                with self.subTest(norm="add_rms_norm", case=label):
                    gm = _trace_add_rms_norm_mx_quant(dst_type=dst_type, quant_kwargs=quant_kwargs)
                    self.assertEqual(matcher.apply(gm.graph), 0)
                    self.assertEqual(len(_call_nodes(gm, torch.ops.npu.npu_dynamic_mx_quant)), 1)

                with self.subTest(norm="rms_norm", case=label):
                    gm = _trace_rms_norm_mx_quant(dst_type=dst_type, quant_kwargs=quant_kwargs)
                    self.assertEqual(matcher.apply(gm.graph), 0)
                    self.assertEqual(len(_call_nodes(gm, torch.ops.npu.npu_dynamic_mx_quant)), 1)

    def test_without_scale_alg_one_capability_ceil_stays_unfused(self):
        matcher = pm.PatternMatcherPass(pass_name="test_mx_no_alg1")
        with (
            self._patched_ops(),
            patch.object(self.base_pattern, "_registered_patterns", set()),
            patch.object(self.base_pattern.nge, "register_replacement"),
        ):
            self._register_mx_patterns(matcher, scale_algs=(DynamicMXQuantScaleAlg.MAX_EXPONENT,))

            gm_alg0 = _trace_add_rms_norm_mx_quant()
            self.assertEqual(matcher.apply(gm_alg0.graph), 1)

            gm_alg1 = _trace_add_rms_norm_mx_quant(quant_kwargs={"scale_alg": 1})
            self.assertEqual(matcher.apply(gm_alg1.graph), 0)
            self.assertEqual(len(_call_nodes(gm_alg1, torch.ops.npu.npu_dynamic_mx_quant)), 1)

    def test_nge_registration_preserves_scale_alg_and_stream_scope_check(self):
        backend_pass = pm.PatternMatcherPass(pass_name="test_nge_mx")

        def register_backend(**kwargs):
            pm.register_replacement(**kwargs, trace_fn=pm.fwd_only, pass_dicts=backend_pass)

        with (
            self._patched_ops(),
            patch.object(self.base_pattern, "_registered_patterns", set()),
            patch.object(self.base_pattern.nge, "register_replacement", side_effect=register_backend),
        ):
            self._register_mx_patterns(pm.PatternMatcherPass(pass_name="inductor_mx"))

            gm_same_stream = _trace_add_rms_norm_mx_quant(quant_kwargs={"scale_alg": 1})
            self.assertEqual(backend_pass.apply(gm_same_stream.graph), 1)
            fused_nodes = _call_nodes(gm_same_stream, torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant)
            self.assertEqual(len(fused_nodes), 1)
            self.assertEqual(fused_nodes[0].kwargs.get("scale_alg", 0), 1)

            # Cross-stream nodes must still be rejected by extra_stream_scope_check.
            gm_cross_stream = _trace_add_rms_norm_mx_quant(quant_kwargs={"scale_alg": 1})
            quant_node = _call_nodes(gm_cross_stream, torch.ops.npu.npu_dynamic_mx_quant)[0]
            quant_node.meta["stream_label"] = "aux_stream"
            self.assertEqual(backend_pass.apply(gm_cross_stream.graph), 0)

            # Unsupported scale_alg must be rejected in the NGE registry too.
            gm_alg2 = _trace_add_rms_norm_mx_quant(quant_kwargs={"scale_alg": 2})
            self.assertEqual(backend_pass.apply(gm_alg2.graph), 0)

    def test_pass_init_registers_both_scale_algs_on_a5_only_when_supported(self):
        recorded: list[tuple[str, float, int]] = []

        def record_register(pattern_self, _pm_pass):
            if hasattr(pattern_self, "scale_alg"):
                recorded.append((type(pattern_self).__name__, pattern_self.eps, pattern_self.scale_alg))

        with (
            patch.object(self.norm_fusion, "enable_custom_op", return_value=False),
            patch.object(self.norm_fusion.BasePattern, "register", record_register),
            patch.object(
                self.norm_fusion,
                "get_current_hardware_profile",
                return_value=get_hardware_profile(AscendDeviceType.A5),
            ),
        ):
            self.norm_fusion.AddRMSNormQuantFusionPass(self.vllm_config)

        self.assertEqual(
            set(recorded),
            {
                ("AddRMSNormDynamicMXQuantPattern", 1e-5, 0),
                ("AddRMSNormDynamicMXQuantPattern", 1e-5, 1),
                ("AddRMSNormDynamicMXQuantPattern", 1e-6, 0),
                ("AddRMSNormDynamicMXQuantPattern", 1e-6, 1),
                ("RMSNormDynamicMXQuantPattern", 1e-5, 0),
                ("RMSNormDynamicMXQuantPattern", 1e-5, 1),
                ("RMSNormDynamicMXQuantPattern", 1e-6, 0),
                ("RMSNormDynamicMXQuantPattern", 1e-6, 1),
            },
        )

        # Profile with DYNAMIC_MX_QUANT_FUSION only (without SCALE_ALG_ONE) registers only scale_alg=0.
        recorded.clear()
        fusion_only_profile = SimpleNamespace(
            supports=lambda cap: cap == HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
        )
        with (
            patch.object(self.norm_fusion, "enable_custom_op", return_value=False),
            patch.object(self.norm_fusion.BasePattern, "register", record_register),
            patch.object(self.norm_fusion, "get_current_hardware_profile", return_value=fusion_only_profile),
        ):
            self.norm_fusion.AddRMSNormQuantFusionPass(self.vllm_config)

        self.assertEqual(
            set(recorded),
            {
                ("AddRMSNormDynamicMXQuantPattern", 1e-5, 0),
                ("AddRMSNormDynamicMXQuantPattern", 1e-6, 0),
                ("RMSNormDynamicMXQuantPattern", 1e-5, 0),
                ("RMSNormDynamicMXQuantPattern", 1e-6, 0),
            },
        )
