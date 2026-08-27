from collections.abc import Iterable
from pathlib import Path

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.models.llama_eagle3 import (
    get_rotation_matrix,
    get_rotation_path,
    load_quarot_target_layer,
)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    if rotation_weight.dtype != torch.float32:
        rotation_weight = rotation_weight.to(torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None
        self.target_model_path = Path(vllm_config.model_config.model)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        all_weights = list(weights)
        includes_embed_tokens = any("embed_tokens" in name for name, _ in all_weights)
        includes_lm_head = any("lm_head" in name for name, _ in all_weights)
        rotation_weight = None
        if self.rotation_path is not None:
            processed_weights: list[tuple[str, torch.Tensor]] = []
            rotation_weight = get_rotation_matrix(self.rotation_path)
            for name, loaded_weight in all_weights:
                if "fc." in name:
                    loaded_weight = process_weight(loaded_weight, rotation_weight)
                processed_weights.append((name, loaded_weight))
            all_weights = processed_weights

        # Upstream load_weights already manages confidence_head (vllm#47808).
        result = super().load_weights(all_weights)

        if rotation_weight is not None:
            if not includes_embed_tokens:
                load_quarot_target_layer(
                    self.model.embed_tokens,
                    self.target_model_path,
                    TARGET_EMBED_WEIGHT_NAMES,
                    rotation_weight,
                    "draft embed_tokens.weight",
                )
                self.has_own_embed_tokens = True
            if not includes_lm_head:
                load_quarot_target_layer(
                    self.lm_head,
                    self.target_model_path,
                    TARGET_LM_HEAD_WEIGHT_NAMES,
                    rotation_weight,
                    "draft lm_head.weight",
                )
                self.has_own_lm_head = True

        return result
