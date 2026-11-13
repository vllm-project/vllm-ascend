# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Bridge additional_config draft_lm_head_quantization onto the upstream field.

additional_config ``draft_lm_head_quantization="w8a8_dynamic"`` selects online
8-bit dynamic quantization for the drafter's copy of a shared target LM head
(EAGLE/EAGLE3/MTP). The format is adjudicated by hardware (MXFP8 on A5, INT8
on A2/A3). The copy itself is derived entirely by the upstream mechanism
driven from ``load_eagle_model``: shared weight storage, MTP per-layer fixup,
and RL re-derivation through the upstream ``_SOURCES`` registry.

The upstream ``SpeculativeConfig.draft_lm_head_quantization`` field is
validated at construction and only accepts the platform formats
("fp8"/"nvfp4"), so the Ascend format is resolved here and assigned to the
field around the draft model loading window only. When the target
``lm_head_quantization`` is also set, the option is absorbed instead: the
drafter shares the quantized target head directly, which achieves the same
effect without an extra copy, and the verifier runs on the quantized head.

Requires Model Runner V2 and a drafter that shares the target LM head
(upstream rejects the rest).
"""

from collections.abc import Iterator
from contextlib import contextmanager

from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

DRAFT_LM_HEAD_QUANTIZATION = "w8a8_dynamic"


def _resolve_format() -> str:
    device_family = get_current_hardware_profile().device_adaptor_family
    if device_family == DeviceAdaptorFamily.COMPATIBILITY:
        raise ValueError("draft_lm_head_quantization is not yet adapted for 310P.")
    return "mxfp8" if device_family == DeviceAdaptorFamily.FP8_OPTIMIZED else "int8"


def _register_lm_head_methods() -> None:
    # The Ascend schemes implement the scheme interface, not the vLLM
    # quant-method interface, so the factories wrap them in AscendEmbeddingMethod,
    # the adapter the upstream _quantize_copy instantiates through. The
    # factories are lazy: the schemes read the current vllm config at
    # construction, which is only valid inside _quantize_copy.
    from vllm.model_executor.layers.quantization.online import lm_head as online_lm_head_module

    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLmHeadMethod

    methods = online_lm_head_module._LM_HEAD_METHODS
    methods.setdefault("int8", lambda: AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod()))
    methods.setdefault("mxfp8", lambda: AscendEmbeddingMethod(AscendW8A8MXFP8DynamicLmHeadMethod()))


@contextmanager
def draft_lm_head_quantization_scope(vllm_config: VllmConfig) -> Iterator[None]:
    """Bridge the Ascend draft LM head format around draft model loading."""
    additional_config = vllm_config.additional_config or {}
    quantization = additional_config.get("draft_lm_head_quantization")
    if quantization is None:
        yield
        return
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_lm_head_quantization is not None:
        raise ValueError(
            "draft_lm_head_quantization in additional_config conflicts with the "
            "speculative_config field of the same name; select the Ascend "
            "feature through additional_config only."
        )
    if quantization != DRAFT_LM_HEAD_QUANTIZATION:
        raise ValueError(f"Unsupported draft_lm_head_quantization: {quantization!r}.")
    if additional_config.get("lm_head_quantization") is not None:
        # The target LM head is quantized online, so the upstream share branch
        # hands the drafter the quantized target head directly: the same
        # effect without an extra copy. Note the absorption also means the
        # verifier runs on the quantized head (no bf16 safety net).
        logger.info_once(
            "lm_head_quantization is enabled; the drafter will share the "
            "quantized target LM head and draft_lm_head_quantization is "
            "absorbed into it."
        )
        yield
        return
    if not vllm_config.use_v2_model_runner:
        raise ValueError("draft_lm_head_quantization requires Model Runner V2.")
    quant_format = _resolve_format()
    _register_lm_head_methods()
    logger.info_once("Quantizing the draft lm_head copy online with %s.", quant_format)
    # The field only accepts the platform formats at construction, so assign
    # the resolved format around the draft loading window only. RL weight
    # updates re-derive the copy through the upstream _SOURCES registry, not
    # this field, so restoring it after loading is safe.
    previous = speculative_config.draft_lm_head_quantization
    object.__setattr__(speculative_config, "draft_lm_head_quantization", quant_format)
    try:
        yield
    finally:
        object.__setattr__(speculative_config, "draft_lm_head_quantization", previous)
