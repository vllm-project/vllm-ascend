#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
import torch
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg
from vllm_ascend.utils import enable_custom_op


class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormDynamicQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternNoBias(BasePattern):
    """add_rms_norm_bias(x, r, w, None, eps) + npu_dynamic_quant -> fused op.

    Qwen3.5/Qwen3-Next RMSNorm has no beta, so ops/layernorm.py forward_oot takes
    the ``beta=None`` branch of ``_C_ascend.npu_add_rms_norm_bias``. The existing
    AddRMSNormDynamicQuantPattern matches ``torch.ops.npu.npu_add_rms_norm``
    instead, and the WithBias variant requires a tensor bias; neither covers this
    call, so the fusion never applied to these models.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                beta=None,
                epsilon=self.eps,
                output_mask=[True, False],
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            # Pin the rule to INT8: npu_dynamic_quant is shared with the INT4
            # (torch.quint4x2) and FP8 paths, while the fused op below produces
            # INT8. Inductor's matcher ignores a keyword the pattern does not
            # declare, so without this the rule would also match those graphs
            # and replace them with the INT8 fused op.
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0, dst_type=torch.int8)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False], beta=bias
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


def _make_mx_quant_extra_check(expected_scale_alg: int):
    """Guard MXFP8 norm-quant fusion against silent keyword-argument drops.

    Inductor's pattern matcher ignores keyword arguments not declared in the
    pattern node. Because ``npu_dynamic_mx_quant`` defaults ``scale_alg`` to 0
    (and also accepts ``axis``, ``round_mode``, ``block_size``, ``dst_type_max``,
    and ``max_low_bound``), a pattern omitting ``scale_alg`` would otherwise
    match ``scale_alg=1`` (CEIL) or non-default quantizer arguments and silently
    rewrite them to schema defaults in the fused op.
    """

    def _check(match: Match) -> bool:
        quant_op = torch.ops.npu.npu_dynamic_mx_quant
        quant_nodes = [
            node
            for node in match.nodes
            if node.op == "call_function" and getattr(node.target, "overloadpacket", node.target) == quant_op
        ]
        if len(quant_nodes) != 1:
            return False
        kwargs = quant_nodes[0].kwargs
        raw_scale_alg = kwargs.get("scale_alg", 0)
        effective_scale_alg = 0 if raw_scale_alg is None else raw_scale_alg
        if isinstance(effective_scale_alg, bool) or effective_scale_alg != expected_scale_alg:
            return False
        if kwargs.get("axis", -1) != -1:
            return False
        if kwargs.get("round_mode", "rint") != "rint":
            return False
        if kwargs.get("block_size", 32) != 32:
            return False
        if kwargs.get("dst_type_max", 0.0) != 0.0:
            return False
        return kwargs.get("max_low_bound", 0.0) == 0.0

    return _check


class AddRMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6, scale_alg: int = 0):
        super().__init__(vllm_config, eps)
        self.scale_alg = int(scale_alg)

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}_scale_alg_{self.scale_alg}"

    def get_extra_check(self):
        return _make_mx_quant_extra_check(self.scale_alg)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            if self.scale_alg == 0:
                quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            else:
                quantized_output = torch.ops.npu.npu_dynamic_mx_quant(
                    out0, dst_type=torch.float8_e4m3fn, scale_alg=self.scale_alg
                )
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                scale_alg=self.scale_alg,
                dst_type=torch.float8_e4m3fn,
            )
            return (
                output[0],
                output[2],
                output[1],
            )

        return replacement


class RMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6, scale_alg: int = 0):
        super().__init__(vllm_config, eps)
        self.scale_alg = int(scale_alg)

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}_scale_alg_{self.scale_alg}"

    def get_extra_check(self):
        return _make_mx_quant_extra_check(self.scale_alg)

    def get_inputs(self):
        """
        Generate example inputs for the RMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm(rms_norm_input, rms_norm_weight, self.eps)
            out0 = output[0]
            if self.scale_alg == 0:
                quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            else:
                quantized_output = torch.ops.npu.npu_dynamic_mx_quant(
                    out0, dst_type=torch.float8_e4m3fn, scale_alg=self.scale_alg
                )
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                rms_norm_weight,
                epsilon=self.eps,
                scale_alg=self.scale_alg,
                dst_type=torch.float8_e4m3fn,
            )
            return output[0], output[1]

        return replacement


def _model_uses_w4a4_quant(vllm_config: VllmConfig | None) -> bool:
    """Check whether the model uses W4A4 int4 quantization for any layer.

    W4A4 int4 schemes (e.g. W4A4_DYNAMIC, W4A4_FLATQUANT_DYNAMIC)
    are incompatible with the fuse_norm_quant optimization,
    so callers use this to disable that fusion.
    """
    if vllm_config is None:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False
    quant_description = getattr(quant_config, "quant_description", None)
    if not quant_description:
        return False
    w4a4_int4_schemes = ["W4A4_DYNAMIC", "W4A4_FLATQUANT_DYNAMIC"]
    return any(
        isinstance(quant_type, str) and quant_type in w4a4_int4_schemes for quant_type in quant_description.values()
    )


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """Fuse AddRMSNorm with the INT8 dynamic activation quantize on Ascend.

    Scope: a quantize node is replaced only when its ``dst_type`` is
    ``torch.int8``, i.e. W8A8 and the int8-activation part of W4A8 - the weight
    width never appears in the graph, only the activation dtype does.
    ``_model_uses_w4a4_quant()`` disables the pass for W4A4 models, and other
    activation dtypes (fp8, int4) do not match the patterns, so those models
    keep the unfused path instead of being fused incorrectly.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        if _model_uses_w4a4_quant(vllm_config):
            logger.debug(
                "Quant fusion not enabled: the model contains "
                "W4A4 quantized weights, which are incompatible with the "
                "norm-quant fusion pass."
            )
            return

        common_epsilons = [1e-5, 1e-6]
        profile = get_current_hardware_profile()

        for eps in common_epsilons:
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if profile.supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION):
                mx_scale_algs = [DynamicMXQuantScaleAlg.MAX_EXPONENT]
                if profile.supports(HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE):
                    mx_scale_algs.append(DynamicMXQuantScaleAlg.CEIL)
                for scale_alg in mx_scale_algs:
                    AddRMSNormDynamicMXQuantPattern(vllm_config, eps=eps, scale_alg=scale_alg).register(
                        self.pattern_match_passes
                    )
                    RMSNormDynamicMXQuantPattern(vllm_config, eps=eps, scale_alg=scale_alg).register(
                        self.pattern_match_passes
                    )
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternNoBias(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
