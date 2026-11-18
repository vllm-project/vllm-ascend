#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
Patch: fix target_layer_num for Eagle3 draft models under Pipeline Parallelism.

Upstream Eagle3 draft models (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
compute ``target_layer_num`` via ``model_config.get_num_layers(parallel_config)``
which, under PP, returns the **per-PP-stage** count. This value feeds into the
draft model's ``start_layer_id`` (used to build parameter name prefixes like
``model.layers.<start_layer_id + i>``). With PP>1 the prefixes collide with
the checkpoint (e.g. a 61-layer target + 2-way PP builds prefixes 31..34 while
the checkpoint expects 61..64), breaking weight loading. Additionally,
``config.target_layer_count`` (used to index ``layer_types`` for draft
attention) ends up wrong.

Fix: use ``get_total_num_hidden_layers()`` instead. This matches the
checkpoint's global layer indices and keeps ``target_layer_count`` correct.

Currently patches:
- Eagle3LlamaForCausalLM (Qwen, LLaMA-based Eagle3 targets)
- Eagle3DeepseekV2ForCausalLM / Eagle3DeepseekV3ForCausalLM (DeepSeek-V2/V3,
  Kimi K2/K2.6)

Constructor dependencies are resolved through their upstream modules at call
time so that vLLM overrides and test patch points remain effective.
"""

import logging

import torch
import torch.nn as nn
from vllm.model_executor.models import deepseek_eagle3, llama_eagle3
from vllm.model_executor.models.utils import maybe_prefix

logger = logging.getLogger(__name__)


def _patched_eagle3_llama_init(self, *, vllm_config, prefix: str = ""):
    nn.Module.__init__(self)
    self.config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(self.config, "draft_vocab_size", None) is None:
        base_vocab_size = getattr(self.config, "vocab_size", None)
        self.config.draft_vocab_size = base_vocab_size
    target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()

    self.config.target_layer_count = target_layer_num
    self.model = llama_eagle3.LlamaModel(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "model"),
        start_layer_id=target_layer_num,
    )

    logit_scale = getattr(self.config, "logit_scale", 1.0)
    self.lm_head = llama_eagle3.ParallelLMHead(
        self.config.draft_vocab_size,
        self.config.hidden_size,
        quant_config=llama_eagle3.get_draft_quant_config(vllm_config),
        prefix=maybe_prefix(prefix, "lm_head"),
    )
    self.logits_processor = llama_eagle3.LogitsProcessor(self.config.draft_vocab_size, scale=logit_scale)
    self.draft_id_to_target_id = nn.Parameter(
        torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
        requires_grad=False,
    )

    self.use_parallel_drafting = vllm_config.speculative_config.parallel_drafting

    if self.use_parallel_drafting:
        self.register_buffer(
            "mask_hidden",
            torch.zeros(1, self.model.fc_input_size),
            persistent=False,
        )


def _patched_eagle3_deepseek_v2_init(self, *, vllm_config, prefix: str = ""):
    nn.Module.__init__(self)
    self.config = vllm_config.speculative_config.draft_model_config.hf_config

    if getattr(self.config, "draft_vocab_size", None) is None:
        base_vocab_size = getattr(self.config, "vocab_size", None)
        self.config.draft_vocab_size = base_vocab_size

    target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()

    self.config.target_layer_count = target_layer_num

    self.model = deepseek_eagle3.DeepseekV2Eagle3Model(
        vllm_config=vllm_config,
        prefix=maybe_prefix(prefix, "model"),
        start_layer_id=target_layer_num,
    )

    logit_scale = getattr(self.config, "logit_scale", 1.0)
    self.lm_head = deepseek_eagle3.ParallelLMHead(
        self.config.draft_vocab_size,
        self.config.hidden_size,
        prefix=maybe_prefix(prefix, "lm_head"),
    )
    self.logits_processor = deepseek_eagle3.LogitsProcessor(self.config.draft_vocab_size, scale=logit_scale)
    self.draft_id_to_target_id = nn.Parameter(
        torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
        requires_grad=False,
    )


llama_eagle3.Eagle3LlamaForCausalLM.__init__ = _patched_eagle3_llama_init
deepseek_eagle3.Eagle3DeepseekV2ForCausalLM.__init__ = _patched_eagle3_deepseek_v2_init

logger.info(
    "Patched Eagle3LlamaForCausalLM and Eagle3DeepseekV2ForCausalLM "
    "__init__ to use get_total_num_hidden_layers() for target_layer_num."
)
