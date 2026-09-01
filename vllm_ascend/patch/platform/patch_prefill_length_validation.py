from __future__ import annotations

from typing import Any

from vllm.logger import logger
from vllm.v1.engine.async_llm import AsyncLLM


_ORIGINAL_ADD_REQUEST = AsyncLLM._add_request


def _is_target_model(engine: AsyncLLM) -> bool:
    model_config = engine.vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    architectures = " ".join(getattr(hf_config, "architectures", ()) or ()).lower()
    model_name = str(getattr(model_config, "model", "")).lower()
    model_identifiers = f"{model_type} {architectures} {model_name}"
    return any(name in model_identifiers for name in ("glm-5.2", "glm5.2", "glm52", "deepseek_v4"))


def _prefill_max_tokens(request: Any) -> int:
    params = getattr(request, "kv_transfer_params", None) or {}
    if not params:
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        params = extra_args.get("kv_transfer_params", {}) or {}
    default_value = getattr(request, "max_tokens", None)
    if default_value is None:
        sampling_params = getattr(request, "sampling_params", None)
        default_value = getattr(sampling_params, "max_tokens", 0)
    value = params.get("original_max_tokens", default_value)
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return max(int(default_value or 0), 0)

def _validate_prefill_request(engine: AsyncLLM, request: Any) -> None:
    kv_config = getattr(engine.vllm_config, "kv_transfer_config", None)
    if kv_config is None or getattr(kv_config, "kv_role", None) != "kv_producer":
        return
    if not _is_target_model(engine):
        return
    max_tokens = _prefill_max_tokens(request)
    prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if prompt_tokens is None:
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return
        prompt_tokens = len(prompt_token_ids)
    total_tokens = prompt_tokens + max_tokens
    max_model_len = engine.vllm_config.model_config.max_model_len
    if total_tokens > max_model_len:
        raise ValueError(
            "Request exceeds model-len before prefill: "
            f"prompt tokens ({prompt_tokens}) + max_tokens ({max_tokens}) "
            f"= {total_tokens} > model-len ({max_model_len})."
        )
    logger.debug(
        "Prefill length validation passed: request_id=%s prompt_tokens=%d max_tokens=%d model_len=%d",
        request.request_id,
        prompt_tokens,
        max_tokens,
        max_model_len,
    )


async def _patched_add_request(
    self: AsyncLLM,
    request: Any,
    prompt: str | None,
    parent_req: Any,
    index: int,
    queue: Any,
) -> Any:
    # Validate before OutputProcessor and before crossing the EngineCore boundary.
    _validate_prefill_request(self, request)
    return await _ORIGINAL_ADD_REQUEST(self, request, prompt, parent_req, index, queue)


AsyncLLM._add_request = _patched_add_request  # type: ignore[method-assign]
