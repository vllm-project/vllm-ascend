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
import os

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from torch.fx.experimental.symbolic_shapes import statically_known_true
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# computation-communication tiling block is 512
ALLREDUCE_NORM_FUSE_THRESHOLD = 512
MATMUL_ALLREDUCE_RMSNORM_910C_MAX_M = 2
MATMUL_ALLREDUCE_RMSNORM_910C_K = 12800
MATMUL_ALLREDUCE_RMSNORM_910C_N = 5120


def _get_match_tensor(match, name: str) -> torch.Tensor | None:
    node = match.kwargs.get(name)
    if node is None:
        return None
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    return value if isinstance(value, torch.Tensor) else None


def _shape_matches(tensor: torch.Tensor | None, expected: tuple[int | None, ...]) -> bool:
    if tensor is None or len(tensor.shape) != len(expected):
        return False
    return all(
        wanted is None or statically_known_true(actual == wanted)
        for actual, wanted in zip(tensor.shape, expected)
    )


def _is_supported_910c_down_proj_match(match) -> bool:
    x = _get_match_tensor(match, "x")
    weight = _get_match_tensor(match, "weight")
    residual = _get_match_tensor(match, "residual")
    gamma = _get_match_tensor(match, "rms_norm_weight")
    tensors = (x, weight, residual, gamma)
    return (
        all(tensor is not None and tensor.dtype == torch.bfloat16 for tensor in tensors)
        and x is not None
        and len(x.shape) >= 2
        and statically_known_true(x.shape[-1] == MATMUL_ALLREDUCE_RMSNORM_910C_K)
        and _shape_matches(
            weight,
            (MATMUL_ALLREDUCE_RMSNORM_910C_N, MATMUL_ALLREDUCE_RMSNORM_910C_K),
        )
        and residual is not None
        and len(residual.shape) == len(x.shape)
        and all(
            statically_known_true(actual == expected)
            for actual, expected in zip(
                residual.shape,
                (*x.shape[:-1], MATMUL_ALLREDUCE_RMSNORM_910C_N),
            )
        )
        and _shape_matches(gamma, (MATMUL_ALLREDUCE_RMSNORM_910C_N,))
    )


def should_use_910c_op(vllm_config: VllmConfig, tp_size: int | None = None) -> bool:
    if tp_size is None:
        tp_size = vllm_config.parallel_config.tensor_parallel_size
    hf_config = vllm_config.model_config.hf_config
    return (
        get_ascend_device_type() == AscendDeviceType.A3
        and tp_size == 2
        and getattr(hf_config, "hidden_size", None) == 5120
    )


class MiddleLayerMatmulAllReduceAddRMSNormPattern(BasePattern):
    """
    recognizing the Matmul+AllReduce+AddRMSNorm computation pattern
    AllReduce is optimized in the fusion operator to a two-stage communication of ReduceScatter+AllGather
    """

    def __init__(self, vllm_config, eps=1e-6):
        self.vllm_config = vllm_config
        self.eps = eps
        tp_group = get_tp_group()
        device_group = tp_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.fallback_group_name = tp_group.unique_name
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_910c_op = should_use_910c_op(self.vllm_config, self.tp_size)

    def get_inputs(self):
        batch_size, seq_len = 2, 4
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            chunked_residual = torch.ops.vllm.maybe_chunk_residual(all_reduce_, residual)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_, chunked_residual, rms_norm_weight, None)
            out0 = output[0]
            out1 = output[2]
            return out0, out1

        return pattern

    def get_extra_check(self):
        return _is_supported_910c_down_proj_match if self.use_910c_op else super().get_extra_check()

    def get_replacement(self):
        def replacement(x, weight, residual, rms_norm_weight):
            args = (
                x,
                weight,
                residual,
                rms_norm_weight,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,
                False,
            )
            if self.use_910c_op:
                out0, out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm_910c(
                    *args, self.fallback_group_name
                )
            else:
                out0, out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(*args)
            return out0, out1

        return replacement


class LastLayerMatmulAllReduceAddRMSNormPattern(BasePattern):
    def __init__(self, vllm_config, eps=1e-6):
        super().__init__(vllm_config, eps)
        tp_group = get_tp_group()
        device_group = tp_group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.fallback_group_name = tp_group.unique_name
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_910c_op = should_use_910c_op(self.vllm_config, self.tp_size)

    def get_inputs(self):
        batch_size, seq_len = 2, 4
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            chunked_residual = torch.ops.vllm.maybe_chunk_residual(all_reduce_, residual)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_, chunked_residual, rms_norm_weight, None)
            return output[0]

        return pattern

    def get_extra_check(self):
        return _is_supported_910c_down_proj_match if self.use_910c_op else super().get_extra_check()

    def get_replacement(self):
        def replacement(x, weight, residual, rms_norm_weight):
            args = (
                x,
                weight,
                residual,
                rms_norm_weight,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,
                False,
            )
            if self.use_910c_op:
                out0, _ = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm_910c(
                    *args, self.fallback_group_name
                )
            else:
                out0, _ = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(*args)
            return out0

        return replacement


class MatmulAllReduceAddRMSNormPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.use_910c_op = should_use_910c_op(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="allreduce_rmsnorm_fusion_pass")

        middle_pattern = MiddleLayerMatmulAllReduceAddRMSNormPattern(vllm_config)
        last_pattern = LastLayerMatmulAllReduceAddRMSNormPattern(vllm_config)
        middle_pattern.register(self.pattern_match_passes, register_nge=not self.use_910c_op)
        last_pattern.register(self.pattern_match_passes, register_nge=not self.use_910c_op)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        limit_value = os.getenv("VLLM_ASCEND_MAR910C_FUSION_LIMIT")
        limit = int(limit_value) if limit_value is not None else None
        original_checks = []
        accepted = 0

        if limit is not None:
            if limit < 0:
                raise ValueError("VLLM_ASCEND_MAR910C_FUSION_LIMIT must be non-negative")

            for entries in self.pattern_match_passes.patterns.values():
                for entry in entries:
                    original_check = entry.extra_check
                    original_checks.append((entry, original_check))

                    def limited_check(match, check=original_check):
                        nonlocal accepted
                        if accepted >= limit or not check(match):
                            return False
                        accepted += 1
                        return True

                    entry.extra_check = limited_check

        try:
            self.matched_count = self.pattern_match_passes.apply(graph)
        finally:
            for entry, original_check in original_checks:
                entry.extra_check = original_check

        if limit is not None:
            logger.info(
                "Applied matmul-allreduce-rmsnorm fusion limit %d: replaced %d patterns",
                limit,
                self.matched_count,
            )
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        if self.use_910c_op:
            applicable = compile_range.end <= MATMUL_ALLREDUCE_RMSNORM_910C_MAX_M
        else:
            applicable = compile_range.start > ALLREDUCE_NORM_FUSE_THRESHOLD
        return applicable
