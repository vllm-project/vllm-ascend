# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-facing meta model and synthetic GLM-5.3-Flash initialization."""

import torch
from transformers import AutoModel, PretrainedConfig


def build_meta_model(config: PretrainedConfig) -> torch.nn.Module:
    """Enumerate native HF text weights in the published checkpoint layout.

    AutoModelForCausalLM has no GLM-5.3 text entry, so attach a metadata-only
    LM head to AutoModel's text backbone. Normalize mHC/forget-gate namespaces
    and split the fused KDA convolution into checkpoint q/k/v modules. This
    adapter is only constructed on meta; its forward pass is never used.
    Routed experts are expanded by the shared weight source as for other cases.
    """
    model = torch.nn.Module()
    model.model = AutoModel.from_config(config, trust_remote_code=False)
    model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    for layer in model.model.layers:
        for branch in ("attn", "ffn"):
            connection = getattr(layer, f"{branch}_hc")
            for name, parameter in connection.named_parameters():
                layer.register_parameter(f"hc_{branch}_{name}", parameter)
            delattr(layer, f"{branch}_hc")
        attention = layer.self_attn
        if hasattr(attention, "forget_gate"):
            for name, module in attention.forget_gate.named_children():
                attention.add_module(name, module)
            for name, parameter in attention.forget_gate.named_parameters(recurse=False):
                attention.register_parameter(name, parameter)
            del attention.forget_gate
            for projection, weight in zip(("q", "k", "v"), attention.conv1d.weight.chunk(3, dim=0), strict=True):
                convolution = torch.nn.Module()
                convolution.register_parameter("weight", torch.nn.Parameter(weight))
                attention.add_module(f"{projection}_conv1d", convolution)
            del attention.conv1d
    return model


def initialize_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Adjust the source's seeded random tensor for KDA/mHC numerical stability.

    Both startup loading and live updates use these exact tensors. These values
    exercise the real modules but make no statement about checkpoint accuracy.
    """
    if name.endswith("A_log"):
        return tensor.zero_()
    if name.endswith("dt_bias"):
        return tensor.fill_(-2.0)
    if "hc_" in name:
        if name.endswith("scale"):
            return tensor.fill_(1.0)
        if name.endswith("base"):
            return tensor.zero_()
        return tensor.mul_(0.05)
    if "norm" in name and name.endswith("weight"):
        return tensor.fill_(1.0)
    if name.endswith("bias"):
        return tensor.zero_()
    return tensor.mul_(0.5)
