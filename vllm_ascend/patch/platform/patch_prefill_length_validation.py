# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validate the proxy's original generation budget before submitting prefill."""

from functools import wraps
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.async_llm import AsyncLLM


def _is_target_model(model_config: Any) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    identifiers = [
        getattr(hf_config, "model_type", ""),
        *(getattr(hf_config, "architectures", None) or ()),
        getattr(model_config, "model", ""),
    ]
    # GLM-5.3 uses the GLM MoE DSA architecture. Match its config identifiers
    # even when the local model directory does not contain the model version.
    # Keep name matching for GLM-5.2/5.3 and DeepSeek-V4 variants.
    for identifier in identifiers:
        normalized = str(identifier).lower().replace("_", "").replace("-", "").replace(".", "")
        if normalized in ("glmmoedsa", "glmmoedsaforcausallm") or any(
            name in normalized for name in ("glm52", "glm53", "deepseekv4")
        ):
            return True
    return False


def _validate_prefill_request(engine: Any, request: Any) -> None:
    config = engine.vllm_config
    if getattr(getattr(config, "kv_transfer_config", None), "kv_role", None) != "kv_producer":
        return
    if not _is_target_model(config.model_config):
        return

    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    transfer_params = getattr(request, "kv_transfer_params", None) or extra_args.get("kv_transfer_params") or {}
    original_max_tokens = transfer_params.get("original_max_tokens")
    if original_max_tokens is None:
        # Regular requests and proxies without this metadata retain upstream
        # validation/default-budget behavior.
        return
    if isinstance(original_max_tokens, bool) or not isinstance(original_max_tokens, int) or original_max_tokens < 1:
        raise VLLMValidationError(
            "original_max_tokens must be a positive integer",
            parameter="kv_transfer_params.original_max_tokens",
            value=original_max_tokens,
        )

    prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if prompt_tokens is None:
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        else:
            prompt_embeds = getattr(request, "prompt_embeds", None)
            if prompt_embeds is None:
                return
            prompt_tokens = prompt_embeds.shape[0]

    max_model_len = config.model_config.max_model_len
    total_tokens = prompt_tokens + original_max_tokens
    if total_tokens > max_model_len:
        # Plain ValueError is wrapped as EngineGenerateError by vLLM 0.30.0.
        # VLLMValidationError instead follows its client-error (HTTP 400) path.
        raise VLLMValidationError(
            "Request exceeds model-len before prefill: "
            f"prompt tokens ({prompt_tokens}) + max_tokens ({original_max_tokens}) "
            f"= {total_tokens} > model-len ({max_model_len}).",
            parameter="max_tokens",
        )


def _install_patch() -> None:
    original = AsyncLLM._add_request
    if getattr(original, "_ascend_prefill_length_validation", False):
        return

    @wraps(original)
    async def add_request(self, request, *args, **kwargs):
        _validate_prefill_request(self, request)
        return await original(self, request, *args, **kwargs)

    add_request._ascend_prefill_length_validation = True
    AsyncLLM._add_request = add_request


_install_patch()
