#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import torch
import torch.fx as fx
import torch_npu
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.compilation.passes.inductor_pass import enable_fake_mode, get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import WeightLayoutPolicy, get_current_hardware_profile

# Imported for its registration side effect: defines
# torch.ops.vllm.ascend_sp_reduce_scatter_impl, matched by the reduce-scatter pattern.
from vllm_ascend.models.common.ops import sequence_parallel as _sequence_parallel  # noqa: F401
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg

_E8M0 = torch_npu.float8_e8m0fnu
# ACL dtype enum accepted by the fused matmul-comm kernels' ``y_dtype``, which takes the
# numeric ACL code rather than a torch.dtype. 15 == ACL_BF16.
_ACL_DTYPE_BF16 = 15
# Only bf16 output is wired up: the replacements hardcode _ACL_DTYPE_BF16, so the
# extra checks below must reject any graph whose matmul output is not bf16.
_SUPPORTED_OUTPUT_DTYPE = torch.bfloat16

_QUANT_MATMUL_OP = torch.ops.npu.npu_quant_matmul.default
_UNQUANTIZED_GEMM_OP = torch.ops.vllm.unquantized_gemm.default


def _get_hcomm_info(group) -> str:
    device_group = group.device_group
    rank = torch.distributed.get_rank(device_group)
    global_rank = torch.distributed.get_global_rank(device_group, rank)
    return device_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)


class _MatmulCommPattern(BasePattern):
    """Base for matmul + TP-collective fusions.

    Subclasses differ by matmul op identity, operand arity and collective
    topology -- not by dtype, which is constrained via ``get_extra_check``.
    """

    def __init__(self, vllm_config: VllmConfig, min_tokens: int, **kwargs):
        super().__init__(vllm_config, **kwargs)
        self.min_tokens = min_tokens

    @enable_fake_mode
    def register(self, pattern_match_passes: PatternMatcherPass) -> None:
        # Fake mode keeps the example operands below from allocating real NPU memory.
        super().register(pattern_match_passes)

    def _meets_min_tokens(self, node: fx.Node | None) -> bool:
        """Reject matches whose token count is below ``min_tokens``."""
        val = node.meta.get("val") if node is not None else None
        if val is None or val.dim() < 1:
            return False

        num_tokens = val.shape[0]
        if isinstance(num_tokens, int):
            return num_tokens >= self.min_tokens

        try:
            compile_range = get_pass_context().compile_range
        except AssertionError:
            logger.debug_once(
                "Matmul-comm fusion declined for a symbolic token dim: no compile range in scope."
            )
            return False
        return compile_range.start >= self.min_tokens


class _QuantMatmulCommPattern(_MatmulCommPattern):
    """
    A pattern that fuses a quantized matmul with a communication operation (reduce-scatter or all-reduce).
    """

    def __init__(self, vllm_config: VllmConfig, group_name: str, hcomm_info: str, group_size: int, min_tokens: int):
        super().__init__(vllm_config, min_tokens=min_tokens)
        self.group_name = group_name
        self.hcomm_info = hcomm_info
        self.group_size = group_size
        self.tp_size = get_tensor_model_parallel_world_size()

    @property
    def group_sizes(self) -> list[int]:
        return [1, 1, self.group_size]

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.group_size}_{self.tp_size}_{self.min_tokens}"

    def _example_mxfp8_operands(self, m: int, k: int, n: int) -> list[torch.Tensor]:
        k_dim = k // self.group_size
        return [
            torch.empty(m, k, device="npu", dtype=torch.float8_e4m3fn),
            torch.empty(m, k_dim // 2, 2, device="npu", dtype=torch.uint8),
            torch.empty(k, n, device="npu", dtype=torch.float8_e4m3fn),
            torch.empty(k_dim // 2, n, 2, device="npu", dtype=torch.uint8),
        ]

    def _quant_matmul(self, x, x_scale, weight, weight_scale) -> torch.Tensor:
        return torch.ops.npu.npu_quant_matmul(
            x,
            weight,
            weight_scale,
            scale_dtype=_E8M0,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=_E8M0,
            bias=None,
            output_dtype=_SUPPORTED_OUTPUT_DTYPE,
            group_sizes=self.group_sizes,
        )

    def get_extra_check(self):
        def supported(match: Match) -> bool:
            # The replacements hardcode a bf16 ACL output dtype; reject anything else
            # rather than registering one pattern variant per dtype.
            matmuls = [node for node in match.nodes if node.target is _QUANT_MATMUL_OP]
            if not matmuls:
                return False
            for node in matmuls:
                val = node.meta.get("val")
                if val is None or val.dtype is not _SUPPORTED_OUTPUT_DTYPE:
                    return False
            return self._meets_min_tokens(matmuls[0])

        return supported


class QuantMatmulReduceScatterPattern(_QuantMatmulCommPattern):
    """
    A pattern that fuses a quantized matmul with a reduce-scatter operation.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        return self._example_mxfp8_operands(m=64, k=256, n=128)

    def get_pattern(self):
        def pattern(
            quantized_x: torch.Tensor,
            pertoken_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            mm = self._quant_matmul(quantized_x, pertoken_scale, weight, weight_scale)
            return torch.ops.vllm.ascend_sp_reduce_scatter_impl.default(mm)

        return pattern

    def get_replacement(self):
        def replacement(
            quantized_x: torch.Tensor,
            pertoken_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            out = DeviceOperator.npu_quant_mm_reduce_scatter(
                quantized_x,
                weight,
                self.hcomm_info,
                self.tp_size,
                reduce_op="sum",
                x1_scale=pertoken_scale,
                x2_scale=weight_scale,
                group_sizes=self.group_sizes,
                x1_scale_dtype=_E8M0,
                x2_scale_dtype=_E8M0,
                y_dtype=_ACL_DTYPE_BF16,
            )
            return out[0]

        return replacement


class _AllGatherQuantMatmulPattern(_QuantMatmulCommPattern):
    """
    A pattern that fuses a quantized matmul with an all-gather operation.
    """
    _REAL_TOKENS = 57
    _SHARD_M = 32

    def __init__(self, *args, scale_alg: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_alg = scale_alg

    def pattern_key(self) -> str:
        return f"{super().pattern_key()}_{self.scale_alg}"

    def get_scalar_workaround(self) -> dict[str, float | int]:
        return {"real_tokens": self._REAL_TOKENS}

    def _example_shard(self, k: int) -> torch.Tensor:
        return torch.empty(self._SHARD_M, k, device="npu", dtype=self.dtype)

    def _example_mxfp8_weight(self, k: int, n: int) -> list[torch.Tensor]:
        return self._example_mxfp8_operands(m=self._SHARD_M, k=k, n=n)[2:]

    def _quantize(self, x: torch.Tensor):
        return torch.ops.npu.npu_dynamic_mx_quant(
            x,
            dst_type=torch.float8_e4m3fn,
            scale_alg=self.scale_alg,
        )

    def _gather(self, shard: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather(shard, 0, self.tp_size, group_name=self.group_name)

    def _fused(self, quantized_shard, shard_scale, weight, weight_scale, *, gather_output: bool):
        return DeviceOperator.npu_all_gather_quant_mm(
            quantized_shard,
            weight,
            self.hcomm_info,
            self.tp_size,
            x1_scale=shard_scale,
            x2_scale=weight_scale,
            group_sizes=self.group_sizes,
            x1_scale_dtype=_E8M0,
            x2_scale_dtype=_E8M0,
            y_dtype=_ACL_DTYPE_BF16,
            gather_output=gather_output,
        )


class AllGatherQuantMatmulPattern(_AllGatherQuantMatmulPattern):

    def get_inputs(self) -> list[torch.Tensor]:
        k, n = 256, 128
        return [self._example_shard(k), *self._example_mxfp8_weight(k, n)]

    def get_pattern(self):
        def pattern(shard, weight, weight_scale, real_tokens):
            gathered = self._gather(shard)[:real_tokens]
            quantized_x, pertoken_scale = self._quantize(gathered)
            return self._quant_matmul(quantized_x, pertoken_scale, weight, weight_scale)

        return pattern

    def get_replacement(self):
        def replacement(shard, weight, weight_scale, real_tokens):
            quantized_shard, shard_scale = self._quantize(shard)
            out = self._fused(
                quantized_shard,
                shard_scale,
                weight,
                weight_scale,
                gather_output=False,
            )
            return out[0][:real_tokens]

        return replacement


class AllGatherDualQuantMatmulPattern(_AllGatherQuantMatmulPattern):

    def get_inputs(self) -> list[torch.Tensor]:
        k, n1, n2 = 256, 128, 64
        return [
            self._example_shard(k),
            *self._example_mxfp8_weight(k, n1),
            *self._example_mxfp8_weight(k, n2),
        ]

    def get_pattern(self):
        def pattern(shard, weight1, weight_scale1, weight2, weight_scale2, real_tokens):
            gathered = self._gather(shard)[:real_tokens]
            quantized_1, scale_1 = self._quantize(gathered)
            out1 = self._quant_matmul(quantized_1, scale_1, weight1, weight_scale1)
            quantized_2, scale_2 = self._quantize(gathered)
            out2 = self._quant_matmul(quantized_2, scale_2, weight2, weight_scale2)
            return out1, out2

        return pattern

    def get_replacement(self):
        def replacement(shard, weight1, weight_scale1, weight2, weight_scale2, real_tokens):
            quantized_shard, shard_scale = self._quantize(shard)
            out = self._fused(
                quantized_shard,
                shard_scale,
                weight1,
                weight_scale1,
                gather_output=True,
            )
            gathered_x = out[1]
            gathered_scale = self._gather(shard_scale)
            out2 = self._quant_matmul(gathered_x, gathered_scale, weight2, weight_scale2)
            return out[0][:real_tokens], out2[:real_tokens]

        return replacement


class MatmulAllReducePattern(_MatmulCommPattern):
    _SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

    def __init__(self, vllm_config: VllmConfig, group_name: str, hcomm_info: str, min_tokens: int):
        super().__init__(vllm_config, min_tokens=min_tokens)
        self.group_name = group_name
        self.hcomm_info = hcomm_info
        self.tp_size = get_tensor_model_parallel_world_size()

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.tp_size}_{self.min_tokens}"

    def get_extra_check(self):
        def supported(match: Match) -> bool:
            gemm = next((node for node in match.nodes if node.target is _UNQUANTIZED_GEMM_OP), None)
            if gemm is None:
                return False
            x, weight = gemm.args[0], gemm.args[1]
            if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
                return False
            x_val, weight_val = x.meta.get("val"), weight.meta.get("val")
            if x_val is None or weight_val is None:
                return False
            if x_val.dim() != 2 or weight_val.dim() != 2:
                return False
            if x_val.dtype not in self._SUPPORTED_DTYPES or x_val.dtype is not weight_val.dtype:
                return False
            return self._meets_min_tokens(x)

        return supported

    def get_inputs(self) -> list[torch.Tensor]:
        m, k, n = 32, 256, 128
        return [
            torch.empty(m, k, device="npu", dtype=self.dtype),
            torch.empty(n, k, device="npu", dtype=self.dtype),
        ]

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            return torch.ops.vllm.all_reduce(mm, group_name=self.group_name)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return DeviceOperator.npu_mm_all_reduce_base(
                x,
                weight.t(),
                self.hcomm_info,
                reduce_op="sum",
            )

        return replacement


class MatmulCommFusionPass(VllmInductorPass):

    def __init__(
        self,
        vllm_config: VllmConfig,
        allow_quant_comm: bool = True,
        allow_all_reduce: bool = True,
    ):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(pass_name="matmul_comm_fusion_pass")
        self.enabled = False

        from vllm_ascend.ascend_config import get_ascend_config

        ascend_config = get_ascend_config()
        compilation_config = ascend_config.ascend_compilation_config
        self.min_tokens = compilation_config.matmul_comm_fusion_min_tokens
        self.fuse_reduce_scatter = compilation_config.fuse_matmul_reduce_scatter and allow_quant_comm
        self.fuse_all_gather = compilation_config.fuse_all_gather_matmul and allow_quant_comm
        self.fuse_all_reduce = compilation_config.fuse_matmul_all_reduce and allow_all_reduce
        if not (self.fuse_reduce_scatter or self.fuse_all_gather or self.fuse_all_reduce):
            return

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            logger.info("Matmul-comm fusion not enabled: TP size is 1")
            return

        if get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ:
            logger.info("Matmul-comm fusion not enabled: hardware forces NZ weight layout")
            return

        if (self.fuse_reduce_scatter or self.fuse_all_gather) and ascend_config.weight_nz_mode != 0:
            logger.info(
                "Reduce-scatter / all-gather fusion not enabled: weight_nz_mode=%s converts FP8 "
                "weights to NZ, which npu_quant_mm_reduce_scatter rejects. Set weight_nz_mode=0 "
                "to use them.",
                ascend_config.weight_nz_mode,
            )
            self.fuse_reduce_scatter = self.fuse_all_gather = False

        if self.fuse_all_reduce and ascend_config.weight_nz_mode == 2:
            logger.info(
                "Matmul-all-reduce fusion not enabled: weight_nz_mode=2 converts bf16 weights "
                "to NZ, which the transposing replacement cannot use. Set weight_nz_mode=0 or 1 "
                "to use it."
            )
            self.fuse_all_reduce = False

        group_size = None
        if self.fuse_reduce_scatter or self.fuse_all_gather:
            group_size = self._resolve_mx_group_size(vllm_config)
            if group_size is None:
                logger.info("Reduce-scatter / all-gather fusion not enabled: model is not MXFP8 quantized")
                self.fuse_reduce_scatter = self.fuse_all_gather = False

        if not (self.fuse_reduce_scatter or self.fuse_all_gather or self.fuse_all_reduce):
            return

        hcomm_info = _get_hcomm_info(tp_group)
        common = {
            "group_name": tp_group.unique_name,
            "hcomm_info": hcomm_info,
            "group_size": group_size,
            "min_tokens": self.min_tokens,
        }

        registered: list[str] = []

        if self.fuse_reduce_scatter:
            QuantMatmulReduceScatterPattern(vllm_config, **common).register(self.pattern_match_passes)
            registered.append("quant_matmul_reduce_scatter")

        if self.fuse_all_gather:
            scale_alg = get_dynamic_mx_quant_scale_alg(vllm_config)
            for pattern_cls in (AllGatherQuantMatmulPattern, AllGatherDualQuantMatmulPattern):
                pattern_cls(vllm_config, scale_alg=scale_alg, **common).register(self.pattern_match_passes)
            registered.append("all_gather_quant_matmul")

        if self.fuse_all_reduce:
            MatmulAllReducePattern(
                vllm_config,
                group_name=tp_group.unique_name,
                hcomm_info=hcomm_info,
                min_tokens=self.min_tokens,
            ).register(self.pattern_match_passes)
            registered.append("matmul_all_reduce")

        self.enabled = True
        logger.info("MatmulCommFusionPass: %s pattern(s) registered, enabled", ", ".join(registered))

    @staticmethod
    def _resolve_mx_group_size(vllm_config: VllmConfig) -> int | None:
        quant_description = getattr(vllm_config.quant_config, "quant_description", None) or {}
        if not isinstance(quant_description, dict):
            return None

        if not any(isinstance(v, str) and "MXFP8" in v for v in quant_description.values()):
            return None

        group_size = quant_description.get("group_size", 32)
        return group_size if isinstance(group_size, int) and group_size > 0 else None

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        if self.matched_count:
            logger.info("Fused %s matmul_comm patterns", self.matched_count)
        else:
            logger.info("Fused no matmul_comm patterns: graph has no matching collective tail")
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.start >= self.min_tokens