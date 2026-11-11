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

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern


class SwigluFusionPattern(BasePattern):
    """Pattern that matches the bare SwiGLU sequence:

        gate, up = x.chunk(2, dim=-1)
        out = F.silu(gate) * up

    and replaces it with a single ``torch_npu.npu_swiglu(x)`` call.

    The pattern is written with the high-level ``F.silu`` op (matching the
    source code form in e.g. ``DeepseekV4VisionMLP``). The pattern matcher
    traces the pattern function with the same decomposition table as the
    target graph, so ``F.silu`` decomposes identically on both sides and the
    structures line up. Writing the decomposed low-level form
    (``sigmoid + mul``) would *not* match under bf16/fp16, where ``silu``
    decomposes to ``convert(f32) -> sigmoid -> mul -> convert(dtype)`` for
    numerical stability. ``npu_swiglu`` accepts the concatenated
    ``[..., 2 * intermediate]`` tensor and internally performs the chunk +
    silu + mul.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        # Single concatenated gate_up tensor; the last dim only needs to be
        # even. Shapes provide meta info for pattern tracing only.
        x = torch.randn(2, 32, device="npu", dtype=self.dtype)
        return [x]

    def get_pattern(self):
        def pattern(x: torch.Tensor):
            gate, up = torch.split(x, x.shape[-1] // 2, dim=-1)
            return F.silu(gate) * up

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor):
            return torch_npu.npu_swiglu(x)

        return replacement


class SwigluFusionPass(VllmInductorPass):
    """A fusion pass that replaces the bare ``chunk(2, dim=-1) -> silu -> mul``
    SwiGLU pattern with ``torch_npu.npu_swiglu`` on Ascend.

    This benefits model paths that write the activation with raw aten ops
    (e.g. ``DeepseekV4VisionMLP``) instead of going through ``AscendSiluAndMul``.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="swiglu_fusion_pass")

        # npu_swiglu is a CANN operator that supports fp16 / bf16.
        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            logger.debug("SwiGLU fusion not enabled: unsupported dtype %s", dtype)
            return

        SwigluFusionPattern(vllm_config).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s swiglu patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Whether the pass is applicable for the current compile range.

        SwiGLU fusion is always allowed for the selected ranges for now; this
        hook exists so finer-grained range control can be added later.
        """
        return True
