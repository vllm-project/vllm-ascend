# SPDX-License-Identifier: Apache-2.0

from inspect import signature
from types import SimpleNamespace

import pytest
import vllm.v1.structured_output as structured_output
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output import StructuredOutputManager, backend_guidance, backend_xgrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

from vllm_ascend.patch.platform import patch_structured_output  # noqa: F401

MODEL_CONFIG = SimpleNamespace(is_diffusion=False)


class FakeBackend:
    def __init__(self, vllm_config, tokenizer, vocab_size):
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def compile_grammar(self, request_type, grammar_spec):
        return (type(self).__name__, request_type, grammar_spec)


class FakeXgrammarBackend(FakeBackend):
    pass


class FakeGuidanceBackend(FakeBackend):
    pass


def make_manager() -> StructuredOutputManager:
    manager = object.__new__(StructuredOutputManager)
    manager.backend = None
    manager.vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 128))
    manager.tokenizer = object()
    manager._use_async_grammar_compilation = False
    return manager


def make_request(backend: str):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(structured_outputs=SimpleNamespace(_backend=backend)),
        structured_output_request=SimpleNamespace(
            structured_output_key=(StructuredOutputOptions.JSON, "{}"),
            grammar=None,
        ),
    )


def validate_structured_outputs(params, config):
    original_validate = getattr(
        SamplingParams,
        patch_structured_output._ORIGINAL_VALIDATE_ATTR,
    )
    if "model_config" in signature(original_validate).parameters:
        params._validate_structured_outputs(MODEL_CONFIG, config, tokenizer=object())
    else:
        params._validate_structured_outputs(config, tokenizer=object())


def test_sampling_params_rejects_mixed_structured_output_backends(monkeypatch):
    def fake_validate_xgrammar(sampling_params):
        schema = sampling_params.structured_outputs.json
        if schema.get("force_guidance"):
            raise ValueError("xgrammar unsupported")

    monkeypatch.setattr(
        backend_xgrammar,
        "validate_xgrammar_grammar",
        fake_validate_xgrammar,
    )
    monkeypatch.setattr(
        backend_guidance,
        "has_guidance_unsupported_json_features",
        lambda schema: False,
    )
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        lambda sampling_params, tokenizer=None: None,
    )

    config = StructuredOutputsConfig(backend="auto")
    xgrammar_params = SamplingParams(structured_outputs=StructuredOutputsParams(json={"type": "object"}))
    validate_structured_outputs(xgrammar_params, config)

    assert xgrammar_params.structured_outputs._backend == "xgrammar"
    assert getattr(config, patch_structured_output._BACKEND_ATTR) == "xgrammar"

    guidance_params = SamplingParams(structured_outputs=StructuredOutputsParams(json={"force_guidance": True}))
    with pytest.raises(ValueError, match="already using 'xgrammar'.*'guidance'"):
        validate_structured_outputs(guidance_params, config)


def test_sampling_params_allows_consistent_guidance_backend(monkeypatch):
    monkeypatch.setattr(
        backend_guidance,
        "has_guidance_unsupported_json_features",
        lambda schema: False,
    )
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        lambda sampling_params, tokenizer=None: None,
    )

    config = StructuredOutputsConfig(backend="guidance")
    for _ in range(2):
        params = SamplingParams(structured_outputs=StructuredOutputsParams(json={"type": "array"}))
        validate_structured_outputs(params, config)

        assert params.structured_outputs._backend == "guidance"
        assert getattr(config, patch_structured_output._BACKEND_ATTR) == "guidance"


def test_failed_first_validation_does_not_lock_config(monkeypatch):
    monkeypatch.setattr(
        backend_xgrammar,
        "validate_xgrammar_grammar",
        lambda sampling_params: (_ for _ in ()).throw(ValueError("xgrammar error")),
    )
    monkeypatch.setattr(
        backend_guidance,
        "has_guidance_unsupported_json_features",
        lambda schema: False,
    )
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        lambda sampling_params, tokenizer=None: (_ for _ in ()).throw(ValueError("guidance error")),
    )

    config = StructuredOutputsConfig(backend="auto")
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json={"force_guidance": True}))
    with pytest.raises(ValueError, match="guidance error"):
        validate_structured_outputs(params, config)

    assert not hasattr(config, patch_structured_output._BACKEND_ATTR)


def test_manager_rejects_mixed_structured_output_backends(monkeypatch):
    monkeypatch.setattr(structured_output, "XgrammarBackend", FakeXgrammarBackend)
    monkeypatch.setattr(structured_output, "GuidanceBackend", FakeGuidanceBackend)

    manager = make_manager()
    xgrammar_request = make_request("xgrammar")
    manager.grammar_init(xgrammar_request)

    assert isinstance(manager.backend, FakeXgrammarBackend)
    assert (
        getattr(
            manager,
            patch_structured_output._BACKEND_ATTR,
        )
        == "xgrammar"
    )
    assert xgrammar_request.structured_output_request.grammar == (
        "FakeXgrammarBackend",
        StructuredOutputOptions.JSON,
        "{}",
    )

    guidance_request = make_request("guidance")
    with pytest.raises(ValueError, match="already using 'xgrammar'.*'guidance'"):
        manager.grammar_init(guidance_request)


def test_manager_rejects_mixed_backend_after_subclassed_backend_is_initialized():
    manager = make_manager()
    manager.backend = FakeXgrammarBackend(
        manager.vllm_config,
        manager.tokenizer,
        manager.vllm_config.model_config.get_vocab_size(),
    )

    with pytest.raises(ValueError, match="already using 'xgrammar'.*'guidance'"):
        manager.grammar_init(make_request("guidance"))


def test_manager_allows_consistent_guidance_backend(monkeypatch):
    monkeypatch.setattr(structured_output, "GuidanceBackend", FakeGuidanceBackend)

    manager = make_manager()
    for _ in range(2):
        request = make_request("guidance")
        manager.grammar_init(request)

        assert isinstance(manager.backend, FakeGuidanceBackend)
        assert getattr(manager, patch_structured_output._BACKEND_ATTR) == "guidance"
        assert request.structured_output_request.grammar == (
            "FakeGuidanceBackend",
            StructuredOutputOptions.JSON,
            "{}",
        )


def test_failed_first_backend_does_not_lock_manager(monkeypatch):
    monkeypatch.setattr(structured_output, "XgrammarBackend", FakeXgrammarBackend)

    manager = make_manager()
    with pytest.raises(ValueError, match="Unsupported structured output backend"):
        manager.grammar_init(make_request("unsupported"))

    assert not hasattr(manager, patch_structured_output._BACKEND_ATTR)

    request = make_request("xgrammar")
    manager.grammar_init(request)

    assert isinstance(manager.backend, FakeXgrammarBackend)
    assert getattr(manager, patch_structured_output._BACKEND_ATTR) == "xgrammar"
