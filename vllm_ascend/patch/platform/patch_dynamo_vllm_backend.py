# mypy: ignore-errors
from typing import Any, Dict

import torch.fx as fx
from vllm.compilation.backends import VllmBackend
from vllm.compilation.caching import VllmSerializableFunction

_original_vllmbackend_call = VllmBackend.__call__


def __patch_call__(self, graph: fx.GraphModule, example_inputs,
                   options: Dict[str, Any]) -> VllmSerializableFunction:
    return _original_vllmbackend_call(self, graph, example_inputs)


VllmBackend.__call__ = __patch_call__
