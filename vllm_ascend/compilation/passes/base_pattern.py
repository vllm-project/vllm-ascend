from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import torchair
import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig
from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        
    @abstractmethod
    def get_example_inputs(self) -> List[torch.Tensor]:
        pass
    
    @abstractmethod
    def get_pattern(self) -> Callable:
        pass
    
    @abstractmethod
    def get_replacement(self) -> Callable:
        pass
    
    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_example_inputs()
        
        pm.register_replacement(
            pattern_fn, replacement_fn, example_inputs, 
            pm.fwd_only, pm_pass
        )
        
        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=extra_stream_scope_check,
        )