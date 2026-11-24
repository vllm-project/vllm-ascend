# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independently merge disk A/B into checkpoint tensors before normal loading."""

from dataclasses import replace

import torch
from bf16_scope import disk_module, effective_factors
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("m31_bf16_merged")
class BF16MergedModelLoader(DefaultModelLoader):
    def __init__(self, load_config):
        self.factors = {
            disk_module(name): pair
            for name, pair in effective_factors(load_config.model_loader_extra_config["adapter"]).items()
        }
        super().__init__(replace(load_config, load_format="auto", model_loader_extra_config={}))

    def get_all_weights(self, model_config, model):
        seen = set()
        for name, weight in super().get_all_weights(model_config, model):
            logical = disk_module("model." + name.removesuffix(".weight"))
            if logical in self.factors:
                if weight.dtype != torch.bfloat16:
                    raise ValueError(f"Expected a BF16 base matrix for {name}, got {weight.dtype}.")
                a, b = self.factors[logical]
                if tuple(weight.shape) != (b.shape[0], a.shape[1]):
                    raise ValueError(f"Merged LoRA shape mismatch for {name}.")
                weight = torch.addmm(weight.float(), b.float(), a.float()).bfloat16()
                seen.add(logical)
            yield name, weight
        missing = set(self.factors) - seen
        if missing:
            raise ValueError(f"Adapter matrices absent from the checkpoint: {sorted(missing)}")
        model._m31_merged_tensor_count = len(seen)


class MergedWorker:
    """Importing the extension registers the loader before model construction."""

    def m31_merged_identity(self):
        return {
            "merged_tensors": self.model_runner.model._m31_merged_tensor_count,
            "lora_enabled": self.model_runner.lora_config is not None,
        }
