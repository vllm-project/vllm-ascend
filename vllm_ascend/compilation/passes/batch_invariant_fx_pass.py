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
"""
FX Graph pass for batch-invariant mode.

This pass replaces aten matrix operations with batch-invariant implementations
at the FX graph level, making it compatible with npugraph_ex compilation.
"""

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)


class BatchInvariantFXPass(VllmInductorPass):
    """
    FX graph pass that replaces aten matrix operations with batch-invariant
    implementations.
    
    This pass uses pattern matching to find aten operations (mm, matmul, linear, etc.)
    and replaces them with calls to batch-invariant Triton kernels.
    """
    
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config
        self.pm_pass = PatternMatcherPass()
        
        if not HAS_TRITON:
            logger.warning("Triton not available, batch-invariant FX pass will not register patterns.")
            return
            
        # Import batch-invariant implementations
        from vllm_ascend.ops.triton.batch_invariant.matmul import (
            addmm_batch_invariant,
            bmm_batch_invariant,
            linear_batch_invariant,
            matmul_batch_invariant,
            mm_batch_invariant,
        )
        
        self.mm_batch_invariant = mm_batch_invariant
        self.matmul_batch_invariant = matmul_batch_invariant
        self.addmm_batch_invariant = addmm_batch_invariant
        self.bmm_batch_invariant = bmm_batch_invariant
        self.linear_batch_invariant = linear_batch_invariant
        
        # Register all patterns
        self._register_mm_patterns()
        self._register_matmul_patterns()
        self._register_addmm_patterns()
        self._register_bmm_patterns()
        self._register_linear_patterns()
    
    def _register_mm_patterns(self):
        """Register aten::mm pattern replacement."""
        
        def mm_pattern(a: torch.Tensor, b: torch.Tensor):
            return torch.mm(a, b)
        
        def mm_replacement(a: torch.Tensor, b: torch.Tensor):
            return self.mm_batch_invariant(a, b)
        
        # Example inputs for 2D matmul
        example_inputs = [
            torch.randn(128, 256, device="npu", dtype=torch.float16),
            torch.randn(256, 512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            mm_pattern,
            mm_replacement,
            example_inputs,
            self.pm_pass,
        )
    
    def _register_matmul_patterns(self):
        """Register aten::matmul pattern replacement."""
        
        def matmul_pattern(a: torch.Tensor, b: torch.Tensor):
            return torch.matmul(a, b)
        
        def matmul_replacement(a: torch.Tensor, b: torch.Tensor):
            return self.matmul_batch_invariant(a, b)
        
        # Register for 2D x 2D case
        example_inputs_2d = [
            torch.randn(128, 256, device="npu", dtype=torch.float16),
            torch.randn(256, 512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            matmul_pattern,
            matmul_replacement,
            example_inputs_2d,
            self.pm_pass,
        )
        
        # Register for 3D x 2D case (common in transformers)
        example_inputs_3d_2d = [
            torch.randn(4, 128, 256, device="npu", dtype=torch.float16),
            torch.randn(256, 512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            matmul_pattern,
            matmul_replacement,
            example_inputs_3d_2d,
            self.pm_pass,
        )
    
    def _register_addmm_patterns(self):
        """Register aten::addmm pattern replacement."""
        
        def addmm_pattern(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
            return torch.addmm(bias, a, b)
        
        def addmm_replacement(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
            return self.addmm_batch_invariant(bias, a, b)
        
        example_inputs = [
            torch.randn(512, device="npu", dtype=torch.float16),
            torch.randn(128, 256, device="npu", dtype=torch.float16),
            torch.randn(256, 512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            addmm_pattern,
            addmm_replacement,
            example_inputs,
            self.pm_pass,
        )
    
    def _register_bmm_patterns(self):
        """Register aten::bmm pattern replacement."""
        
        def bmm_pattern(a: torch.Tensor, b: torch.Tensor):
            return torch.bmm(a, b)
        
        def bmm_replacement(a: torch.Tensor, b: torch.Tensor):
            return self.bmm_batch_invariant(a, b)
        
        example_inputs = [
            torch.randn(4, 128, 256, device="npu", dtype=torch.float16),
            torch.randn(4, 256, 512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            bmm_pattern,
            bmm_replacement,
            example_inputs,
            self.pm_pass,
        )
    
    def _register_linear_patterns(self):
        """Register aten::linear pattern replacement."""
        
        def linear_pattern(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
            return torch.nn.functional.linear(input_, weight, bias)
        
        def linear_replacement(input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
            return self.linear_batch_invariant(input_, weight, bias)
        
        # With bias
        example_inputs_with_bias = [
            torch.randn(128, 256, device="npu", dtype=torch.float16),
            torch.randn(512, 256, device="npu", dtype=torch.float16),  # weight is (out_features, in_features)
            torch.randn(512, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            linear_pattern,
            linear_replacement,
            example_inputs_with_bias,
            self.pm_pass,
        )
        
        # Without bias (pass None)
        def linear_pattern_no_bias(input_: torch.Tensor, weight: torch.Tensor):
            return torch.nn.functional.linear(input_, weight, None)
        
        def linear_replacement_no_bias(input_: torch.Tensor, weight: torch.Tensor):
            return self.linear_batch_invariant(input_, weight, None)
        
        example_inputs_no_bias = [
            torch.randn(128, 256, device="npu", dtype=torch.float16),
            torch.randn(512, 256, device="npu", dtype=torch.float16),
        ]
        
        register_replacement(
            linear_pattern_no_bias,
            linear_replacement_no_bias,
            example_inputs_no_bias,
            self.pm_pass,
        )
    
    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        Batch-invariant pass is always applicable when enabled.
        """
        return True
    
    def __call__(self, graph: torch.fx.Graph):
        """Apply the batch-invariant pass to the FX graph."""
        if not HAS_TRITON:
            return
            
        logger.info("Applying batch-invariant FX pass to graph.")
        
        # Apply pattern matcher pass
        self.pm_pass.apply(graph)
        
        return


def apply_batch_invariant_to_fx_graph(graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Apply batch-invariant transformations to an FX graph module.
    
    This is a standalone function that can be called directly
    before torchair compilation.
    """
    if not HAS_TRITON:
        logger.warning("Triton not available, skipping batch-invariant FX pass.")
        return graph
    
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        linear_batch_invariant,
        matmul_batch_invariant,
        mm_batch_invariant,
    )
    
    modified = False
    
    # Iterate through all nodes in the graph
    for node in list(graph.graph.nodes):
        if node.op != "call_function":
            continue
            
        # Replace aten operations with batch-invariant versions
        if node.target == torch.mm:
            node.target = mm_batch_invariant
            modified = True
        elif node.target == torch.matmul:
            node.target = matmul_batch_invariant
            modified = True
        elif node.target == torch.addmm:
            node.target = addmm_batch_invariant
            modified = True
        elif node.target == torch.bmm:
            node.target = bmm_batch_invariant
            modified = True
        elif node.target == torch.nn.functional.linear:
            node.target = linear_batch_invariant
            modified = True
        # Also handle aten namespace operations
        elif isinstance(node.target, torch._ops.OpOverload):
            target_name = str(node.target)
            if target_name == "aten::mm":
                node.target = mm_batch_invariant
                modified = True
            elif target_name == "aten::matmul":
                node.target = matmul_batch_invariant
                modified = True
            elif target_name == "aten::addmm":
                node.target = addmm_batch_invariant
                modified = True
            elif target_name == "aten::bmm":
                node.target = bmm_batch_invariant
                modified = True
            elif target_name == "aten::linear":
                node.target = linear_batch_invariant
                modified = True
    
    if modified:
        logger.info("Batch-invariant FX pass modified the graph.")
        graph.recompile()
    
    return graph
