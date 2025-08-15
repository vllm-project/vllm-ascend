import torch
from torch.fx.subgraph_rewriter import replace_pattern
import torch_npu
from typing import List, Tuple
from vllm.compilation.vllm_inductor_pass import VllmInductorPass



class AddRMSNormQuantPattern:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


    def register(self, patterns: List[Tuple[callable]]):

      def pattern(self, rms_norm_input, residual, rms_norm_weight, scale, offset):
          """
          Pattern for AddRMSNormQuant fusion.
          """
          output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, 1e-6)
          new_output = output[0]
          residual = output[2]
          quantized_output = torch.ops.npu.npu_quantize(new_output, scale, offset, torch.qint8, -1, False)
          return quantized_output, residual

      def replace(self, rms_norm_input, residual, rms_norm_weight, scale, offset):
          """
          Replacement for the AddRMSNormQuant fusion.
          """
          output = torch.ops.npu.npu_add_rms_norm_quant(
              rms_norm_input, 
              residual, 
              rms_norm_weight, 
              1. / scale, 
              offset, 
              epsilon=1e-6)
          quantized_output = output[0]
          residual = output[2]
          return quantized_output, residual

      patterns.append((pattern, replace))


class AscendQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.patterns = []
        # Register the AddRMSNormQuant fusion pattern into the graph rewriter pattern list
        AddRMSNormQuantPattern(vllm_config).register(self.patterns)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_ascend_quant_fusion")
        for pattern, replace in self.patterns:
          replace_pattern(graph, pattern, replace)
        self.dump_graph(graph, "after_ascend_quant_fusion")
        self.end_and_log()
