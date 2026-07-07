# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output import StructuredOutputManager

_BACKEND_ATTR = "_vllm_ascend_structured_output_backend"
_ORIGINAL_GRAMMAR_INIT_ATTR = "_vllm_ascend_original_grammar_init"
_ORIGINAL_VALIDATE_ATTR = "_vllm_ascend_original_validate_structured_outputs"
_PATCHED_MANAGER_ATTR = "_vllm_ascend_backend_guard_patched"
_PATCHED_SAMPLING_PARAMS_ATTR = "_vllm_ascend_backend_validation_guard_patched"


def _request_backend(request: Any) -> str | None:
    if getattr(request, "structured_output_request", None) is None:
        return None

    sampling_params = getattr(request, "sampling_params", None)
    structured_outputs = getattr(sampling_params, "structured_outputs", None)
    backend = getattr(structured_outputs, "_backend", None)
    return backend if isinstance(backend, str) else None


def _backend_name_from_instance(backend: Any) -> str | None:
    if backend is None:
        return None

    backend_names = {
        "XgrammarBackend": "xgrammar",
        "GuidanceBackend": "guidance",
        "OutlinesBackend": "outlines",
        "LMFormatEnforcerBackend": "lm-format-enforcer",
    }
    for backend_cls in type(backend).__mro__:
        for class_name, backend_name in backend_names.items():
            if class_name in backend_cls.__name__:
                return backend_name
    return None


def _raise_mixed_backend(initialized_backend: str, request_backend: str) -> None:
    raise VLLMValidationError(
        "V1 structured outputs only supports one backend per engine. "
        f"The engine is already using '{initialized_backend}', but "
        f"this request resolved to '{request_backend}'. Configure "
        "`structured_outputs_config.backend` explicitly or use schemas "
        "supported by the initialized backend."
    )


def _sampling_params_backend(sampling_params: SamplingParams) -> str | None:
    structured_outputs = getattr(sampling_params, "structured_outputs", None)
    backend = getattr(structured_outputs, "_backend", None)
    return backend if isinstance(backend, str) else None


def _patch_sampling_params_validation() -> None:
    if getattr(SamplingParams, _PATCHED_SAMPLING_PARAMS_ATTR, False):
        return

    original_validate = SamplingParams._validate_structured_outputs
    setattr(SamplingParams, _ORIGINAL_VALIDATE_ATTR, original_validate)

    def _validate_structured_outputs(
        self: SamplingParams,
        structured_outputs_config: Any,
        tokenizer: Any,
    ) -> None:
        result = original_validate(self, structured_outputs_config, tokenizer)
        request_backend = _sampling_params_backend(self)
        if structured_outputs_config is None or request_backend is None:
            return result

        initialized_backend = getattr(structured_outputs_config, _BACKEND_ATTR, None)
        if initialized_backend is not None and request_backend != initialized_backend:
            _raise_mixed_backend(initialized_backend, request_backend)

        setattr(structured_outputs_config, _BACKEND_ATTR, request_backend)
        return result

    SamplingParams._validate_structured_outputs = _validate_structured_outputs
    setattr(SamplingParams, _PATCHED_SAMPLING_PARAMS_ATTR, True)


def _patch_structured_output_manager() -> None:
    if getattr(StructuredOutputManager, _PATCHED_MANAGER_ATTR, False):
        return

    original_grammar_init = StructuredOutputManager.grammar_init
    setattr(StructuredOutputManager, _ORIGINAL_GRAMMAR_INIT_ATTR, original_grammar_init)

    def grammar_init(self: StructuredOutputManager, request: Any) -> None:
        request_backend = _request_backend(request)
        if request_backend is None:
            return original_grammar_init(self, request)

        initialized_backend = getattr(self, _BACKEND_ATTR, None)
        if initialized_backend is None:
            initialized_backend = _backend_name_from_instance(getattr(self, "backend", None))
            if initialized_backend is not None:
                setattr(self, _BACKEND_ATTR, initialized_backend)

        if initialized_backend is not None and request_backend != initialized_backend:
            _raise_mixed_backend(initialized_backend, request_backend)

        result = original_grammar_init(self, request)
        if getattr(self, "backend", None) is not None:
            setattr(self, _BACKEND_ATTR, request_backend)
        return result

    StructuredOutputManager.grammar_init = grammar_init
    setattr(StructuredOutputManager, _PATCHED_MANAGER_ATTR, True)


_patch_sampling_params_validation()
_patch_structured_output_manager()
