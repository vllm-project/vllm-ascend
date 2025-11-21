from typing import Callable

import torch
import torch.fx as fx
from vllm.logger import init_logger

logger = init_logger(__name__)


class OmniBackend:

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        raise NotImplementedError("current not supported")
