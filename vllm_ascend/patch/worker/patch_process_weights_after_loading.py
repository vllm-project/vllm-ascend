import sys

import torch
from torch import nn
from vllm.config import ModelConfig
from vllm.model_executor.layers.attention import (
    Attention,
    MLAAttention,
    MMEncoderAttention,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader import base_loader, utils
from vllm.model_executor.model_loader.reload import set_torchao_reload_attrs
from vllm.model_executor.model_loader.utils import device_loading_context


def _is_dsa_attention(module: nn.Module) -> bool:
    module_cls = type(module)
    return module_cls.__module__ == "vllm_ascend.models.layer.attention.layer" and module_cls.__name__ == "DSAAttention"


def ascend_process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for Attention, MLA, and MM encoder.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if (isinstance(module, (Attention, MLAAttention, MMEncoderAttention)) or _is_dsa_attention(module)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


utils.process_weights_after_loading = ascend_process_weights_after_loading
base_loader.process_weights_after_loading = ascend_process_weights_after_loading

vllm_ascend_loaders = [
    "vllm_ascend.model_loader.netloader.netloader",
    "vllm_ascend.model_loader.rfork.rfork_loader",
]
for loader_module in vllm_ascend_loaders:
    loader = sys.modules.get(loader_module)
    if loader is not None:
        loader.__dict__["process_weights_after_loading"] = ascend_process_weights_after_loading
