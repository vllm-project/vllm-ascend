#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional, Tuple, Union

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, QuantizationConfig)


# func refers to vocabParallelEmbedding.__init__
def wrapper_vocab_parallel_embedding_init(func):

    def init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        func(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        # TODO: Contact vLLM maintainers to add a `params_dtype` attribute to the `VocabParallelEmbedding` class.
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

    return init


# func refers to RMSNorm.__init__
def wrapper_rmsnorm_init(func):

    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                       requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def wrapper_rmsnorm_forward_oot(func):

    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.ignore_anti:
            if residual is not None:
                residual += x
                out = torch_npu._npu_quant_rms_norm(
                    residual,
                    self.weight,
                    self.bias,
                    self.input_scale,
                    self.input_offset,
                    self.variance_epsilon,
                )
                return out, residual
            out = torch_npu._npu_quant_rms_norm(
                x,
                self.weight,
                self.bias,
                self.input_scale,
                self.input_offset,
                self.variance_epsilon,
            )
            return out

        if residual is not None:
            x, residual = func(self, x, residual)
            return x.add_(self.bias), residual

        return func(self, x).add_(self.bias)

    return _rmsnorm_forward_oot


MODEL_LAYER_MAPPING = {
    "LlamaModel": {
        "attn": {
            "layer_attr": "self_attn",
            "proj_attr": "qkv_proj",
            "norm_attr": "input_layernorm",
            "unquantized_type": UnquantizedLinearMethod,
        },
        "mlp": {
            "layer_attr": "mlp",
            "proj_attr": "gate_up_proj",
            "norm_attr": "post_attention_layernorm",
            "unquantized_type": UnquantizedLinearMethod,
        },
    },
}


def wrapper_load_model(func):

    def postprocess_loading(self) -> None:
        func(self)

        def process_layer(layer, idx, mapping):

            def process_module(module_cfg, layer_obj):
                if module_cfg is None:
                    return

                module_obj = getattr(layer_obj, module_cfg["layer_attr"], None)
                if module_obj is None:
                    return

                proj_attr = module_cfg["proj_attr"]
                if callable(proj_attr):
                    proj = proj_attr(module_obj, idx)
                else:
                    proj = getattr(module_obj, proj_attr, None)

                norm = getattr(layer_obj, module_cfg["norm_attr"], None)

                if proj is None or norm is None:
                    return

                norm.ignore_anti = isinstance(proj.quant_method,
                                              module_cfg["unquantized_type"])
                if not norm.ignore_anti:
                    for param_name in ["input_scale", "input_offset"]:
                        if hasattr(proj, param_name):
                            param = getattr(proj, param_name)
                            norm.register_parameter(
                                param_name,
                                torch.nn.Parameter(param.clone(),
                                                   requires_grad=False))

            process_module(mapping.get("attn"), layer)
            process_module(mapping.get("mlp"), layer)

        model_type = self.model.model.__class__.__name__
        mapping = MODEL_LAYER_MAPPING.get(model_type)

        if not mapping:
            logger.info(
                f"Warning: Model type '{model_type}' not found in MODEL_LAYER_MAPPING. Skipping layer mapping."
            )
            return

        for idx, layer in enumerate(self.model.model.layers):
            process_layer(layer, idx, mapping)

        if isinstance(self.model.model.norm, RMSNorm):
            self.model.model.norm.ignore_anti = True

    return postprocess_loading
