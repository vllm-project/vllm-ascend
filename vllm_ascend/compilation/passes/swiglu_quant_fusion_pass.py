import logging

import torch
import torch_npu
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import (PatternMatcherPass,
                                             PatternPrettyPrinter)
from vllm.attention.layer import Attention
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig

class SwigluQuantPattern:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        input = torch.randn(2, 4, device="npu", dtype=self.dtype)

        return (input,)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_swiglu(input)        
            output,scale = torch.ops.npu.npu_dynamic_quant(output)
            return output,scale

        def replacement(input: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output, scale = torch.ops.vllm.swiglu_quant(input,group_list=None,
                    group_list_type=2)
            return output,scale

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)

        
class SwigluQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing SwigluQuant and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="swiglu_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logging.debug("Quant fusion not enabled: unsupported dtype %s",
                         dtype)
            return
        SwigluQuantPattern(vllm_config).register(self.pattern_match_passes)
        

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logging.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable(self, runtime_shape: int | None = None) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
