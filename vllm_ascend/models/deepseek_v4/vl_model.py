# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend wrapper for DeepSeek-V4-Flash-Vision-Exp.

The processor, sentinel layout, ViT, and aligner are shared with the upstream
vLLM implementation from vllm-project/vllm#54566. This module contains only
the Ascend language-backbone integration and weight-loading boundary.
"""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.deepseek_v4.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    image_sentinel_mask,
)
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v4.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)


def _vision_parameter_name(name: str) -> str | None:
    """Map a checkpoint vision tensor to the wrapper parameter namespace."""
    if name.startswith("model."):
        name = name.removeprefix("model.")
    if name.startswith(("vision.", "aligner.", "image_")):
        return name
    return None


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class AscendDeepseekV4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    """DeepSeek-V4 vision entry point using the Ascend text backbone."""

    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        if getattr(config, "vision_n_layers", 0) > 0:
            config.mm_prefix_clamp_sliding_window = True
            config.mm_prefix_span_leading_pad_modulus = 4
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        image_enabled = config.vision_n_layers > 0 and self.multimodal_config.get_limit_per_prompt("image") > 0
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vision: DeepseekV4ViT | None = None
            self.aligner: DeepseekV4Aligner | None = None
            self.image_start: nn.Parameter | None = None
            self.image_end: nn.Parameter | None = None
            self.image_newline: nn.Parameter | None = None
            self.image_pad: nn.Parameter | None = None
            if image_enabled:
                self.vision = DeepseekV4ViT(config)
                self.aligner = DeepseekV4Aligner(config)
                for name in (
                    "image_start",
                    "image_end",
                    "image_newline",
                    "image_pad",
                ):
                    setattr(
                        self,
                        name,
                        nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32)),
                    )
                self.vision.to(dtype=model_config.dtype)
                self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = AscendDeepseekV4ForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        perm = kwargs.pop("perm", None)
        if vit_grid is None or llm_grid is None or perm is None:
            raise ValueError("DeepSeek-V4 vision input requires patches, vit_grid, llm_grid, and perm.")
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "perm": perm,
        }

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        perm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        assert self.vision is not None and self.aligner is not None
        patches = patches.to(self.aligner.w1.weight.dtype)

        embeds: list[torch.Tensor] = []
        vit_offset = 0
        llm_offset = 0
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(vit_grid.tolist(), llm_grid.tolist(), strict=True):
            n_vit = n_vit_h * n_vit_w
            n_llm = n_llm_h * n_llm_w
            image_embeds = self.aligner(
                self.vision(
                    patches[vit_offset : vit_offset + n_vit],
                    n_vit_h,
                    n_vit_w,
                ),
                n_vit_h,
                n_vit_w,
            )
            item_perm = perm[llm_offset : llm_offset + n_llm].to(image_embeds.device)
            embeds.append(image_embeds[item_perm])
            vit_offset += n_vit
            llm_offset += n_llm
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["perm"],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import (
            _merge_multimodal_embeddings,
        )

        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if self.image_start is not None:
            sentinel_mask = image_sentinel_mask(input_ids)
            if is_multimodal is not None:
                sentinel_mask = sentinel_mask & ~is_multimodal.to(input_ids.device)
            table = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            ).to(inputs_embeds.dtype)
            idx = (input_ids - IMAGE_SENTINEL_BASE_ID).clamp(0, 4)
            inputs_embeds = torch.where(sentinel_mask.unsqueeze(-1), table[idx], inputs_embeds)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required when merging image embeddings.")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded_vision: set[str] = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                vision_name = _vision_parameter_name(name)
                if vision_name is None:
                    yield name, loaded_weight
                    continue
                if vision_name not in params:
                    raise KeyError(f"Vision weight {name!r} has no parameter {vision_name!r}.")
                param = params[vision_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_vision.add(vision_name)

        loaded_language = self.language_model.load_weights(language_weights())
        return loaded_vision | {f"language_model.{name}" for name in loaded_language}

    def process_weights_after_loading(self) -> None:
        hook = getattr(
            self.language_model,
            "process_weights_after_loading",
            None,
        )
        if hook is not None:
            hook()
