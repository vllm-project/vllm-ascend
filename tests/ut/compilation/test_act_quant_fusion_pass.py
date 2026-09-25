"""Stage5 F4 UT: the SwiGlu + static-quant fusion pattern, CPU fx level.

Mirrors the test_norm_quant_fusion_w2_patterns.py idiom: fake-tensor traced
synthetic FX graphs, pure-FX matching, no NPU execution.

Covers:
* direct chain ``npu_swiglu(x) -> torch.ops.vllm.quantize(...)`` fuses to
  ``npu_swiglu_quant`` (with the fp32 casts on smooth_scales/offsets);
* the shape guard fails closed: input last dim > 8192 does NOT fuse
  (the aclnnSwiGluQuantV2 hard limit — the Qwen3-8B TP1/TP2 case);
* odd input last dim does NOT fuse;
* an unmatched quant chain (no swiglu anchor) is left untouched.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx.experimental.proxy_tensor as proxy_tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.ut.base import TestBase
from vllm_ascend.utils import enable_custom_op

try:
    _CUSTOM_OPS_READY = enable_custom_op()
except Exception:
    _CUSTOM_OPS_READY = False

pytestmark = [
    __import__("pytest").mark.skipif(
        not _CUSTOM_OPS_READY,
        reason="_C_ascend custom ops unavailable: vllm.quantize impl needs "
        "the enable_custom_op() registration",
    ),
]

_FUSED_OP = torch.ops.npu.npu_swiglu_quant.default
_SWIGLU_OP = torch.ops.npu.npu_swiglu.default
_DTYPE = torch.bfloat16


def _register(pm_pass):
    from vllm_ascend.compilation.passes.act_quant_fusion_pass import SwiGluQuantPattern

    pattern = SwiGluQuantPattern(
        SimpleNamespace(model_config=SimpleNamespace(dtype=_DTYPE))
    )
    # meta example inputs: registration-time trace is metadata-only
    pattern.get_inputs = lambda: [
        torch.randn(2, 16, device="meta", dtype=_DTYPE),
        torch.ones(8, device="meta", dtype=_DTYPE),
        torch.ones(8, device="meta", dtype=_DTYPE),
        torch.zeros(8, device="meta", dtype=_DTYPE),
    ]
    pattern.register(pm_pass)


def _build_graph(fn, twoI):
    mode = FakeTensorMode()
    with mode:
        args = (
            torch.empty(2, twoI, dtype=_DTYPE),
            torch.empty(twoI // 2, dtype=_DTYPE),
            torch.empty(twoI // 2, dtype=_DTYPE),
            torch.empty(twoI // 2, dtype=_DTYPE),
        )
        gm = proxy_tensor.make_fx(fn, tracing_mode="fake")(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _chain(twoI=None):
    def fn(x, scale, recip, offset):
        act = _SWIGLU_OP(x)
        return torch.ops.vllm.quantize(act, scale, recip, offset)

    return fn


class TestSwiGluQuantPattern(TestBase):
    def test_direct_chain_fuses(self):
        pm_pass = pm.PatternMatcherPass(pass_name="actq_direct")
        _register(pm_pass)
        gm = _build_graph(_chain(), twoI=16)
        count = pm_pass.apply(gm)
        self.assertEqual(count, 1)
        fused = [n for n in gm.graph.nodes if n.target is _FUSED_OP]
        self.assertEqual(len(fused), 1)
        # the fused call drops the (zero) scale output: single getitem only
        getitems = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target.__name__ == "getitem"
        ]
        self.assertEqual(len(getitems), 1)

    def test_oversized_input_fails_closed(self):
        """2I=12288 > 8192 (Qwen3-8B TP2 form): no fusion, chain untouched."""
        pm_pass = pm.PatternMatcherPass(pass_name="actq_big")
        _register(pm_pass)
        gm = _build_graph(_chain(), twoI=12288)
        count = pm_pass.apply(gm)
        self.assertEqual(count, 0)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 0)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _SWIGLU_OP]), 1)

    def test_odd_width_fails_closed(self):
        pm_pass = pm.PatternMatcherPass(pass_name="actq_odd")
        _register(pm_pass)
        gm = _build_graph(_chain(), twoI=15)
        count = pm_pass.apply(gm)
        self.assertEqual(count, 0)

    def test_no_swiglu_anchor_untouched(self):
        """quantize without a swiglu producer must not fuse."""
        pm_pass = pm.PatternMatcherPass(pass_name="actq_noanchor")
        _register(pm_pass)

        def fn(x, scale, recip, offset):
            return torch.ops.vllm.quantize(x, scale, recip, offset)

        gm = _build_graph(fn, twoI=16)
        count = pm_pass.apply(gm)
        self.assertEqual(count, 0)

    def test_pass_injection_gated_by_config(self):
        """ActQuantFusionPass with custom-op dispatch on carries the working
        pattern set (functional check on the direct chain)."""
        from vllm_ascend.compilation.passes.act_quant_fusion_pass import (
            ActQuantFusionPass,
        )

        cfg = SimpleNamespace(
            model_config=SimpleNamespace(dtype=_DTYPE),
            device_config=SimpleNamespace(device="npu"),
            compilation_config=SimpleNamespace(
                custom_ops=["all"],
                inductor_compile_config={},
                splitting_ops=[],
                use_inductor_graph_partition=False,
                pass_config=SimpleNamespace(),
            ),
        )
        p = ActQuantFusionPass(cfg)
        gm = _build_graph(_chain(), twoI=16)
        p(gm)  # __call__ applies the pattern set
        self.assertEqual(
            len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 1
        )

    def test_pass_not_registered_under_custom_ops_none(self):
        """M4 e2e lesson: the gate reads the vllm-level custom_ops (the
        npu_swiglu anchor needs CustomOp dispatch), not the nge availability."""
        from vllm_ascend.compilation.passes.act_quant_fusion_pass import (
            ActQuantFusionPass,
        )

        cfg = SimpleNamespace(
            model_config=SimpleNamespace(dtype=_DTYPE),
            device_config=SimpleNamespace(device="npu"),
            compilation_config=SimpleNamespace(
                custom_ops=["none"],
                inductor_compile_config={},
                splitting_ops=[],
                use_inductor_graph_partition=False,
                pass_config=SimpleNamespace(),
            ),
        )
        p = ActQuantFusionPass(cfg)
        gm = _build_graph(_chain(), twoI=16)
        p(gm)
        self.assertEqual(
            len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 0
        )
