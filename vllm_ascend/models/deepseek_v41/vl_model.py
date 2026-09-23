# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend multimodal wrapper for DeepSeek V4.1."""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsEagle3, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD_ID,
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .model import AscendDeepseekV41LLMForCausalLM
from .vision import DeepseekV41Aligner, DeepseekV41ViT


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
class AscendDeepseekV41ForCausalLM(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    """V4.1 image-span semantics with the shared Ascend vision tower."""

    # Engram history and vision MoE routing also consume the original token IDs.
    requires_raw_input_tokens = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    language_model_cls = AscendDeepseekV41LLMForCausalLM

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return IMAGE_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        if getattr(config, "vision_n_layers", 0) > 0:
            config.is_mm_prefix_lm = True
            config.mm_prefix_clamp_sliding_window = True
            config.mm_prefix_span_leading_pad_modulus = 2
        self.config = config
        self.multimodal_config = model_config.multimodal_config

        image_enabled = config.vision_n_layers > 0 and self.multimodal_config.get_limit_per_prompt("image") > 0
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vision: DeepseekV41ViT | None = None
            self.aligner: DeepseekV41Aligner | None = None
            self.image_start: nn.Parameter | None = None
            self.image_end: nn.Parameter | None = None
            self.image_newline: nn.Parameter | None = None
            if image_enabled:
                self.vision = DeepseekV41ViT(config)
                self.aligner = DeepseekV41Aligner(config)
                for name in ("image_start", "image_end", "image_newline"):
                    setattr(
                        self,
                        name,
                        nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32)),
                    )
                self.vision.to(dtype=model_config.dtype)
                self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = self.language_model_cls(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.moe_comm_methods = self.language_model.moe_comm_methods

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        types = kwargs.pop("types", None)
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "types": types,
        }

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        assert self.vision is not None and self.aligner is not None, "Image encoding requires an enabled vision tower"
        return self.aligner(
            self.vision(patches, n_vit_h, n_vit_w),
            n_vit_h,
            n_vit_w,
        )

    def _build_image_span(
        self,
        image_embeds: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        assert self.image_start is not None and self.image_end is not None and self.image_newline is not None
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = image_embeds
        return span

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        types: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        assert self.aligner is not None, "Image processing requires an enabled vision tower"
        patches = patches.to(self.aligner.w1.weight.dtype)
        embeds: list[torch.Tensor] = []
        vit_offset = 0
        span_offset = 0
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(
            vit_grid.tolist(),
            llm_grid.tolist(),
            strict=True,
        ):
            n_vit = n_vit_h * n_vit_w
            span_len = n_llm_h * (n_llm_w + 1) + 2
            image_embeds = self._encode_image(
                patches[vit_offset : vit_offset + n_vit],
                n_vit_h,
                n_vit_w,
            )
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    types[span_offset : span_offset + span_len],
                )
            )
            vit_offset += n_vit
            span_offset += span_len
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["types"],
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

        # The leading alignment row is not an image-feature position. It uses
        # the checkpoint's ordinary image-token embedding instead.
        embedding_ids = input_ids.masked_fill(input_ids == IMAGE_PAD_ID, IMAGE_SENTINEL_BASE_ID)
        inputs_embeds = self.language_model.embed_input_ids(embedding_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        return self.language_model.prepare_engram_graph_inputs(padded_tokens)

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        return self.language_model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )

    @property
    def token_lookback_depth(self) -> int:
        """What the runner sizes the prompt lookback buffer from."""
        return self.language_model.token_lookback_depth

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
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

    @property
    def engram_cache_layer_name(self) -> str | None:
        return self.language_model.engram_cache_layer_name
