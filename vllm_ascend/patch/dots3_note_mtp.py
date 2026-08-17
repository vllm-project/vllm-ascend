# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import deepseek_mtp
from vllm.model_executor.models.utils import _merge_multimodal_embeddings, maybe_prefix


def _get_mtp_hf_config(vllm_config: VllmConfig) -> PretrainedConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and speculative_config.draft_model_config is not None:
        return speculative_config.draft_model_config.hf_config
    return vllm_config.model_config.hf_config


class Dots3NoteMultiTokenPredictor(deepseek_mtp.DeepSeekMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = _get_mtp_hf_config(vllm_config)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.mtp_head_sharing = getattr(config, "mtp_head_sharing", "none")
        if self.mtp_head_sharing not in ("full", "none"):
            raise ValueError(f"MTP only supports mtp_head_sharing='full' or 'none', got {self.mtp_head_sharing!r}")
        self.num_physical_mtp_layers = 1 if self.mtp_head_sharing == "full" else self.num_mtp_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): deepseek_mtp.DeepSeekMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_physical_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def _get_physical_layer_idx(self, spec_step_idx: int) -> int:
        if self.mtp_head_sharing == "full":
            return self.mtp_start_layer_idx
        return self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        layer = self.layers[str(self._get_physical_layer_idx(spec_step_idx))]
        hidden_states = layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        )
        return layer.shared_head(hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        layer = self.layers[str(self._get_physical_layer_idx(spec_step_idx))]
        return self.logits_processor(layer.shared_head.head, hidden_states)


@support_torch_compile
class Dots3NoteMTPModel(deepseek_mtp.DeepSeekMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = _get_mtp_hf_config(vllm_config)
        self.quant_config = vllm_config.quant_config
        self.model = Dots3NoteMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        super().set_moe_parameters()
        self.num_moe_layers = len(self.model.layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def _prepare_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        first_layer = self.model.mtp_start_layer_idx
        share_layers = self.model.mtp_head_sharing == "full"
        dedicated_embeddings = getattr(self.config, "use_dedicated_mtp_embeddings", False)

        for name, weight in weights:
            if dedicated_embeddings and name == "model.embed_tokens.weight":
                continue
            if dedicated_embeddings and name == "model.mtp.embed_tokens.weight":
                name = f"model.layers.{first_layer}.embed_tokens.weight"
            elif share_layers and name == "lm_head.weight":
                name = f"model.layers.{first_layer}.shared_head.head.weight"

            spec_layer = deepseek_mtp.get_spec_layer_idx_from_weight_name(self.config, name)
            if share_layers and spec_layer is not None and spec_layer != first_layer:
                continue
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._prepare_weights(weights))
