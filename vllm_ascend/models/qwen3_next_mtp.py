# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""
import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_next_mtp import (
    Qwen3NextMTP, Qwen3NextMultiTokenPredictor)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, maybe_prefix)
from vllm.transformers_utils.configs import Qwen3NextConfig

from vllm_ascend.models.qwen3_next import (CustomQwen3NextDecoderLayer,
                                           Qwen3NextRMSNorm)


@support_torch_compile
class CustomQwen3NextMultiTokenPredictor(Qwen3NextMultiTokenPredictor):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextMultiTokenPredictor, self).__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        config: Qwen3NextConfig = model_config.hf_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.fc = ColumnParallelLinear(self.config.hidden_size * 2,
                                       self.config.hidden_size,
                                       gather_output=True,
                                       bias=False,
                                       return_bias=False,
                                       quant_config=quant_config,
                                       prefix=f'{prefix}.fc')

        # use old version mtp layer name to avoid a exception in vllm
        self.layers = torch.nn.ModuleList(
            CustomQwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f'{prefix}.layers.{self.mtp_start_layer_idx + idx}',
            ) for idx in range(self.num_mtp_layers))

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)


@support_torch_compile
class CustomQwen3NextMTP(Qwen3NextMTP, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen3NextMTP currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super(Qwen3NextMTP, self).__init__()
        self.config = config
        self.model = CustomQwen3NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(self.unpadded_vocab_size,
                                      config.hidden_size,
                                      org_num_embeddings=config.vocab_size,
                                      padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
