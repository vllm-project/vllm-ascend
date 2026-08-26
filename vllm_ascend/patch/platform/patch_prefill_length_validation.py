from __future__ import annotations

from typing import Any

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore


_ORIGINAL_PREPROCESS_ADD_REQUEST = EngineCore.preprocess_add_request


def _is_target_model(engine: EngineCore) -> bool:
    model_config = engine.vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    architectures = " ".join(getattr(hf_config, "architectures", ()) or ()).lower()
    model_name = str(getattr(model_config, "model", "")).lower()
    model_identifiers = f"{model_type} {architectures} {model_name}"
    return any(name in model_identifiers for name in ("glm-5.2", "glm5.2", "glm52", "deepseek_v4"))


def _prefill_max_tokens(request: Any) -> int:
    params = getattr(request, "kv_transfer_params", None) or {}
    value = params.get("original_max_tokens", request.max_tokens)
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return request.max_tokens


def _validate_prefill_request(engine: EngineCore, request: Any) -> None:
    kv_config = getattr(engine.vllm_config, "kv_transfer_config", None)
    if kv_config is None or getattr(kv_config, "kv_role", None) != "kv_producer":
        return
    if not _is_target_model(engine):
        return
    max_tokens = _prefill_max_tokens(request)
    total_tokens = request.num_prompt_tokens + max_tokens
    max_model_len = engine.scheduler.max_model_len
    if total_tokens > max_model_len:
        raise ValueError(
            "Request exceeds model-len before prefill: "
            f"prompt tokens ({request.num_prompt_tokens}) + max_tokens ({max_tokens}) "
            f"= {total_tokens} > model-len ({max_model_len})."
        )
    logger.debug(
        "Prefill length validation passed: request_id=%s prompt_tokens=%d max_tokens=%d model_len=%d",
        request.request_id,
        request.num_prompt_tokens,
        max_tokens,
        max_model_len,
    )


def _patched_preprocess_add_request(self: EngineCore, request: Any) -> Any:
    # Request.from_engine_core_request has already tokenized the prompt here.
    processed_request = _ORIGINAL_PREPROCESS_ADD_REQUEST(self, request)
    _validate_prefill_request(self, processed_request[0])
    return processed_request


EngineCore.preprocess_add_request = _patched_preprocess_add_request  # type: ignore[method-assign]
