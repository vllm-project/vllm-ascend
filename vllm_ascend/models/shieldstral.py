# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Any

from vllm.config import VllmConfig


_MISTRAL3_TEXT_ARCHITECTURES = {
    "mistral": "MistralForCausalLM",
    "ministral3": "Ministral3ForCausalLM",
}


def _get_mistral3_text_architectures(text_config: object) -> list[str]:
    model_type = getattr(text_config, "model_type", None)
    try:
        return [_MISTRAL3_TEXT_ARCHITECTURES[model_type]]
    except KeyError as exc:
        supported = ", ".join(sorted(_MISTRAL3_TEXT_ARCHITECTURES))
        raise ValueError(
            "Unsupported Shieldstral text_config.model_type "
            f"{model_type!r}; expected one of: {supported}"
        ) from exc


def _prepare_llama4_scaling(config: object) -> None:
    if getattr(config, "llama_4_scaling", None) is not None:
        return
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return
    scaling_beta = rope_parameters.get("llama_4_scaling_beta")
    if scaling_beta is None:
        return
    original_max_position = rope_parameters.get(
        "original_max_position_embeddings"
    )
    if original_max_position is None:
        raise ValueError(
            "llama_4_scaling_beta requires original_max_position_embeddings"
        )
    config.llama_4_scaling = {  # type: ignore[attr-defined]
        "beta": scaling_beta,
        "original_max_position_embeddings": original_max_position,
    }


def patch_mistral3_text_model() -> None:
    """Resolve Shieldstral's Ministral3 text model on vLLM 0.26."""
    import vllm.model_executor.models.mistral3 as mistral3

    original = mistral3.init_vllm_registered_model
    if getattr(original, "_vllm_ascend_shieldstral_compat", False):
        return

    @wraps(original)
    def init_vllm_registered_model(
        vllm_config: VllmConfig,
        *,
        prefix: str = "",
        hf_config: Any | None = None,
        architectures: list[str] | None = None,
    ):
        if hf_config is not None and architectures is None:
            architectures = _get_mistral3_text_architectures(hf_config)
            _prepare_llama4_scaling(hf_config)
        return original(
            vllm_config,
            prefix=prefix,
            hf_config=hf_config,
            architectures=architectures,
        )

    init_vllm_registered_model._vllm_ascend_shieldstral_compat = (  # type: ignore[attr-defined]
        True
    )
    mistral3.init_vllm_registered_model = init_vllm_registered_model
    _patch_pixtral_processor_validation()
    _patch_mistral3_processor_text_only()
    _patch_mistral3_processing_info()


def _patch_pixtral_processor_validation() -> None:
    """Skip Transformers' image-token count check for deterministic grids.

    vLLM may pass a prompt containing the expanded 3025-token image grid while
    the processor's cached/token path returns the raw placeholder. The check
    compares those two representations and rejects an otherwise valid batch.
    """
    from transformers.models.pixtral import processing_pixtral

    processor_cls = processing_pixtral.PixtralProcessor

    if getattr(processor_cls, "_vllm_ascend_shieldstral_validation", False):
        return

    def _check_special_mm_tokens(self, text, text_inputs, modalities):
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token", None)
            token_id = getattr(self, f"{modality}_token_id", None)
            if token_str is None or token_id is None:
                continue
            input_ids = text_inputs["input_ids"]
            for sample_index, sample_text in enumerate(text):
                expected_count = sample_text.count(token_str)
                sample_ids = list(input_ids[sample_index])
                if sample_ids.count(token_id) != expected_count:
                    text_inputs["input_ids"][sample_index] = [token_id] * expected_count

    _check_special_mm_tokens._vllm_ascend_shieldstral_validation = True
    processor_cls._check_special_mm_tokens = _check_special_mm_tokens


def _patch_mistral3_processor_text_only() -> None:
    """Tokenize text-only Mistral3 prompts without Pixtral image validation."""
    from vllm.model_executor.models import mistral3

    processor_cls = mistral3.Mistral3MultiModalProcessor

    if getattr(processor_cls, "_vllm_ascend_shieldstral_text_only", False):
        return

    def _apply_hf_processor_text_only(self, prompt_text, tokenization_kwargs):
        tokenizer = self.info.get_tokenizer()
        if isinstance(prompt_text, str):
            return tokenizer.encode(prompt_text, **dict(tokenization_kwargs))
        return list(prompt_text)

    _apply_hf_processor_text_only._vllm_ascend_shieldstral_text_only = True
    processor_cls._apply_hf_processor_text_only = _apply_hf_processor_text_only


def _patch_mistral3_processing_info() -> None:
    """Precompute the Mistral3/Pixtral image token budget on vLLM 0.27.1.

    vLLM 0.27.1's Transformers PixtralProcessor rejects the dummy ``[IMG]``
    prompt during encoder budget discovery. Returning the deterministic grid
    token count avoids that processor round-trip and keeps startup behavior
    compatible with the original Shieldstral checkpoint contract.
    """
    from vllm.model_executor.models import mistral3

    info_cls = mistral3.Mistral3ProcessingInfo

    if getattr(info_cls, "_vllm_ascend_shieldstral_budget", False):
        return

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
    ) -> dict[str, int]:
        image_size = self.get_image_size_with_most_features()
        return {
            "image": self.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )
        }

    get_mm_max_tokens_per_item._vllm_ascend_shieldstral_budget = True
    info_cls.get_mm_max_tokens_per_item = get_mm_max_tokens_per_item
