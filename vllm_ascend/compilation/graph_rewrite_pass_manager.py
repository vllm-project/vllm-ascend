from torch import fx as fx

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.logger import init_logger
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.compilation.inductor_pass import get_pass_context, InductorPass
from quant_fusion_pass import AscendQuantFusionPass


class GraphRewritePassManager:
    """
    A pass manager for graph rewriting passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu does not
    support inductor and triton for now, we choose to adopt the graph rewriter on
    fx graph rather than the inductor pass manager.
    """

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
        graph.recompile()
        return graph
    
    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)
  
    def configure(self, config: VllmConfig):
        self.pass_config = config.additional_config.ascend_pass_config
        if self.pass_config.enable_addrms_norm_quant_fusion:
            from .quant_fusion_pass import AscendQuantFusionPass
            self.passes.append(AscendQuantFusionPass(config))
        # Add more passes here as needed
