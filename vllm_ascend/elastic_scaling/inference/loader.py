import torch, os
import time
from vllm.config import ModelConfig
from typing import Optional
from torch import nn
from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading)

from vllm.config import ModelConfig, VllmConfig

def zero_copy_model(self, vllm_config: VllmConfig, ipc_engine, model,
            model_config: Optional[ModelConfig] = None) -> nn.Module:
    if int(os.getenv("INFER_STATUS", "0")) > 0:
        ipc_engine.zero_copy_model(model)
        process_weights_after_loading(model, model_config, torch.device("npu"))
        model = model.eval()
        return model
    else:
        return "no model"