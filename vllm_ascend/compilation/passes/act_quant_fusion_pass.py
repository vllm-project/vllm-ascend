#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Stage5 F4: SwiGLU + static W8A8 quant fusion pass (operator replacement).

Mirrors the upstream ``vllm/compilation/passes/fusion/act_quant_fusion.py``
replacement route (act + quant -> library fused op) with the NPU kernel
``torch_npu.npu_swiglu_quant`` (aclnnSwiGluQuantV2):

    pattern:     act = torch.ops.npu.npu_swiglu(x)              # [.., 2I] -> [.., I]
                 y   = torch.ops.vllm.quantize(act, scale, scale_reciprocal, offset)
    replacement: y   = torch.ops.npu.npu_swiglu_quant(
                     x, smooth_scales=scale_reciprocal.float(),
                     offsets=offset.float(), quant_mode=0, dst_type=torch.int8)[0]

Semantics locked by the M0 micro-probes
(stage5/_notes/t0_probe/probe_swiglu_{matrix,static_mapping,formula}.py):

* static mode (quant_mode=0): y = clamp(round(act * smooth_scales)) — so
  ``smooth_scales`` takes the *reciprocal* tensor of the w8a8_static chain
  (max 1-LSB rounding difference vs ``npu_quantize``);
* smooth_scales/offsets must be fp32 (bf16 scales fail kernel binary load,
  error 107000) — the replacement casts;
* the returned scale output is all-zeros in static mode (dropped);
* input last dim must be even and <= 8192 (aclnnSwiGluQuantV2 hard limit):
  Qwen3-8B TP1/TP2 (2I=24576/12288) exceeds it, so the pattern fails closed
  there — recorded as a known limitation; W8A8 TP4 (2I=6144) fits;
* dynamic mode (quant_mode=1) crashes the AIV kernel on this stack
  (910B3 / CANN 9.1.0-beta.1), so the dynamic-quant chain is NOT fused.

Default-off behind ``ascend_compilation_config.fuse_act_quant`` (proved-then-
flip, stage5 D1); the aten silu+mul (custom_ops=none) act form is a follow-up
pending the F1 ruling.
"""

import torch
from torch._inductor.pattern_matcher import (
    Match,
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import enable_custom_op, get_ascend_device_type

# aclnnSwiGluQuantV2 hard input-width limit (SwigluquantKernelNpuOpApi.cpp:66-74).
_SWIGLU_QUANT_MAX_LAST_DIM = 8192


def _swiglu_input_shape_check(match: Match) -> bool:
    """Fail-closed guard: the fused kernel needs an even input last dim <= 8192.

    Reads the tensor_meta of the npu_swiglu input node; symbolic shapes that
    cannot be decided locally fail closed (same lens as the stage4 W2
    ``_merged_leading_dims_view_check`` precedent).
    """
    for node in match.nodes:
        if node.target == torch.ops.npu.npu_swiglu.default:
            src = node.args[0] if node.args else None
            meta = getattr(src, "meta", None) if isinstance(src, torch.fx.Node) else None
            tensor_meta = (meta or {}).get("tensor_meta")
            if tensor_meta is None or not tensor_meta.shape:
                return False
            last = tensor_meta.shape[-1]
            try:
                even = last % 2 == 0
                within = last <= _SWIGLU_QUANT_MAX_LAST_DIM
            except Exception:  # symbolic comparison not locally decidable
                return False
            return bool(even and within)
    return False


class SwiGluQuantPattern(BasePattern):
    """npu_swiglu -> torch.ops.vllm.quantize  ==>  npu_swiglu_quant(static)."""

    def get_inputs(self):
        x = torch.randn(2, 8, device="npu", dtype=self.dtype)
        # w8a8_static runtime form: per-channel expanded [I] tensors.
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [x, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            act = torch.ops.npu.npu_swiglu(x)
            return torch.ops.vllm.quantize(act, scale, scale_reciprocal, offset)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            y, _ = torch.ops.npu.npu_swiglu_quant(
                x,
                smooth_scales=scale_reciprocal.to(torch.float32),
                offsets=offset.to(torch.float32),
                quant_mode=0,
                dst_type=torch.int8,
            )
            return y

        return replacement

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # BasePattern.register does not forward an extra_check; the shape guard
        # is specific to this pattern, so register to torch inductor directly
        # here (nge table skipped — the op pair is aclnn-specific already).
        register_replacement(
            self.get_pattern(),
            self.get_replacement(),
            self.get_inputs(),
            fwd_only,
            pm_pass,
            extra_check=_swiglu_input_shape_check,
        )


class ActQuantFusionPass(VllmInductorPass):
    """F4 pass wrapper: registers the SwiGluQuant pattern set."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(pass_name="act_quant_fusion_pass")
        self._uuid_factors: dict = {
            "dtype": str(vllm_config.model_config.dtype),
            "device": str(get_ascend_device_type()),
            "custom_op": str(enable_custom_op()),
        }

        if vllm_config.model_config.dtype not in (torch.bfloat16, torch.float16):
            logger.debug("ActQuant fusion not enabled: unsupported dtype")
            return
        if enable_custom_op():
            SwiGluQuantPattern(vllm_config).register(self.pattern_match_passes)
        else:
            # The npu_swiglu anchor only exists with custom-op dispatch enabled
            # (custom_ops=all / '+silu_and_mul'); the aten silu+mul form is a
            # follow-up pending the stage5 U5 ruling.
            logger.debug(
                "ActQuant fusion: custom op dispatch off — npu_swiglu anchor "
                "absent, pattern not registered."
            )

    def uuid(self) -> str:
        return InductorPass.hash_dict({"src": super().uuid(), **self._uuid_factors})

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        matched = self.pattern_match_passes.apply(graph)
        VllmPatternMatcherPass.match_table[self.pattern_match_passes.pass_name] += matched
        logger.debug("Replaced %s act_quant patterns", matched)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
