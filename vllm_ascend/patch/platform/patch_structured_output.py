# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from inspect import Signature, signature
from typing import TYPE_CHECKING, Any

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

from vllm_ascend.utils import vllm_version_is

_BACKEND_ATTR = "_vllm_ascend_structured_output_backend"
_ORIGINAL_GRAMMAR_BITMASK_ATTR = "_vllm_ascend_original_grammar_bitmask"
_ORIGINAL_GRAMMAR_INIT_ATTR = "_vllm_ascend_original_grammar_init"
_ORIGINAL_VALIDATE_ATTR = "_vllm_ascend_original_validate_structured_outputs"


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


def _structured_outputs_config_from_call(
    validate_signature: Signature,
    sampling_params: SamplingParams,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    bound_arguments = validate_signature.bind_partial(
        sampling_params,
        *args,
        **kwargs,
    )
    return bound_arguments.arguments.get("structured_outputs_config")


def _patch_sampling_params_validation() -> None:
    original_validate = SamplingParams._validate_structured_outputs
    validate_signature = signature(original_validate)
    setattr(SamplingParams, _ORIGINAL_VALIDATE_ATTR, original_validate)

    def _validate_structured_outputs(
        self: SamplingParams,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        result = original_validate(self, *args, **kwargs)
        structured_outputs_config = _structured_outputs_config_from_call(
            validate_signature,
            self,
            args,
            kwargs,
        )
        request_backend = _sampling_params_backend(self)
        if structured_outputs_config is None or request_backend is None:
            return result

        initialized_backend = getattr(structured_outputs_config, _BACKEND_ATTR, None)
        if initialized_backend is not None and request_backend != initialized_backend:
            _raise_mixed_backend(initialized_backend, request_backend)

        setattr(structured_outputs_config, _BACKEND_ATTR, request_backend)
        return result

    SamplingParams._validate_structured_outputs = _validate_structured_outputs


def _patch_structured_output_manager() -> None:
    install_speculative_reasoning_backport = vllm_version_is("0.27.1")
    original_grammar_init = StructuredOutputManager.grammar_init
    setattr(StructuredOutputManager, _ORIGINAL_GRAMMAR_INIT_ATTR, original_grammar_init)

    def grammar_bitmask(
        self: StructuredOutputManager,
        requests: dict[str, Any],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Any:
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = self.vllm_config.num_speculative_tokens
        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            self._grammar_bitmask = self.backend.allocate_token_bitmask(max_batch_size * (1 + max_num_spec_tokens))

        cumulative_index = 0
        if len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold and max_num_spec_tokens == 0:
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []
                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))
            for promise in promises:
                promise.result()
        else:
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)
                apply_bitmask = self.should_fill_bitmask(request)

                reasoner = self._get_reasoner(request)
                detect_reasoning_end = not apply_bitmask and reasoner is not None and not self.enable_in_reasoning
                simulated_buf: list[int] | None = None
                history_len = 0
                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for i, token in enumerate(req_tokens):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    advance_grammar = apply_bitmask
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    elif detect_reasoning_end and reasoner is not None and not apply_bitmask:
                        if simulated_buf is None:
                            history = list(request.all_token_ids)
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated = simulated_buf[: history_len + i + 1]
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                    if advance_grammar and not grammar.is_terminated():
                        if post_reasoning_end_in_window:
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError((token, req_id, scheduled_spec_decode_tokens))
                    cumulative_index += 1

                if not (self.vllm_config.model_config.is_diffusion and req_tokens):
                    bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                    self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        return bitmask_tensor.numpy()

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
    if install_speculative_reasoning_backport:
        setattr(
            StructuredOutputManager,
            _ORIGINAL_GRAMMAR_BITMASK_ATTR,
            StructuredOutputManager.grammar_bitmask,
        )
        StructuredOutputManager.grammar_bitmask = grammar_bitmask


_patch_sampling_params_validation()
_patch_structured_output_manager()
