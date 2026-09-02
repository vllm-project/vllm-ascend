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


class FakeMaskRow:
    def __init__(self):
        self.value = None

    def fill_(self, value):
        self.value = value


class FakeMask:
    def __init__(self, size):
        self.rows = [FakeMaskRow() for _ in range(size)]

    @property
    def shape(self):
        return (len(self.rows),)

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = FakeMask(0)
            result.rows = self.rows[index]
            return result
        return self.rows[index]

    def numpy(self):
        return [row.value for row in self.rows]


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


def test_should_advance_uses_exact_speculative_window():
    marker = 9
    reasoner = SimpleNamespace(is_reasoning_end_streaming=lambda _all_ids, delta_ids: marker in list(delta_ids))
    manager = object.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    manager._get_reasoner = lambda _request: reasoner

    structured_request = SimpleNamespace(
        reasoning_ended=False,
        reasoning_end_token_index=None,
    )
    request = SimpleNamespace(
        use_structured_output=True,
        structured_output_request=structured_request,
        all_token_ids=[1, 2, marker, 7],
        num_computed_tokens=4,
        num_output_placeholders=1,
    )

    assert manager.should_advance(request, new_token_ids=[marker, 7])
    assert structured_request.reasoning_ended is True
    assert structured_request.reasoning_end_token_index == 2
    assert manager.trim_reasoning_for_advance(request, [marker, 7]) == [7]


def test_should_advance_preserves_legacy_delta_window():
    seen_deltas = []

    def record_delta(_all_ids, delta_ids):
        seen_deltas.append(list(delta_ids))
        return False

    reasoner = SimpleNamespace(is_reasoning_end_streaming=record_delta)
    manager = object.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    manager._get_reasoner = lambda _request: reasoner
    request = SimpleNamespace(
        use_structured_output=True,
        structured_output_request=SimpleNamespace(reasoning_ended=False),
        all_token_ids=[1, 2, 3, 4, 5],
        num_computed_tokens=5,
        num_output_placeholders=2,
    )

    assert manager.should_advance(request) is False
    assert seen_deltas == [[4, 5]]


def test_grammar_bitmask_validates_post_reasoning_drafts_before_accepting():
    marker = 9

    class FakeGrammar:
        def __init__(self):
            self.accepted = []
            self.rolled_back = None

        def fill_bitmask(self, bitmask, index):
            bitmask[index].fill_(index)

        def is_terminated(self):
            return False

        def validate_tokens(self, token_ids):
            return token_ids if token_ids != [7] else []

        def accept_tokens(self, _request_id, token_ids):
            self.accepted.extend(token_ids)
            return True

        def rollback(self, count):
            self.rolled_back = count

    grammar = FakeGrammar()
    reasoner = SimpleNamespace(is_reasoning_end_streaming=lambda _all_ids, delta_ids: list(delta_ids) == [marker])
    manager = object.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=3,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_config=SimpleNamespace(is_diffusion=False),
    )
    manager.backend = SimpleNamespace(allocate_token_bitmask=FakeMask)
    manager._grammar_bitmask = None
    manager._full_mask = -1
    manager.fill_bitmask_parallel_threshold = 100
    manager.should_fill_bitmask = lambda _request: False
    manager._get_reasoner = lambda _request: reasoner

    request = SimpleNamespace(
        all_token_ids=[1, 2],
        structured_output_request=SimpleNamespace(grammar=grammar),
    )
    result = manager.grammar_bitmask(
        {"req": request},
        ["req"],
        {"req": [marker, 7, 8]},
    )

    assert len(result) == 4
    assert grammar.accepted == [8]
    assert grammar.rolled_back == 1


def test_grammar_bitmask_preserves_normal_fsm_advance_path():
    class FakeGrammar:
        def __init__(self):
            self.accepted = []
            self.validate_calls = []

        def fill_bitmask(self, bitmask, index):
            bitmask[index].fill_(index)

        def is_terminated(self):
            return False

        def validate_tokens(self, token_ids):
            self.validate_calls.append(token_ids)
            return token_ids

        def accept_tokens(self, _request_id, token_ids):
            self.accepted.extend(token_ids)
            return True

        def rollback(self, _count):
            pass

    grammar = FakeGrammar()
    reasoner_lookups = []
    manager = object.__new__(StructuredOutputManager)
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=1,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_config=SimpleNamespace(is_diffusion=False),
    )
    manager.backend = SimpleNamespace(allocate_token_bitmask=FakeMask)
    manager._grammar_bitmask = None
    manager._full_mask = -1
    manager.fill_bitmask_parallel_threshold = 100
    manager.should_fill_bitmask = lambda _request: True
    manager._get_reasoner = lambda request: reasoner_lookups.append(request)

    request = SimpleNamespace(
        all_token_ids=[1, 2],
        structured_output_request=SimpleNamespace(grammar=grammar),
    )
    manager.grammar_bitmask({"req": request}, ["req"], {"req": [8]})

    assert reasoner_lookups == [request]
    assert grammar.accepted == [8]
    assert grammar.validate_calls == []


def test_vllm_release_already_has_structured_output_logger():
    assert structured_output.logger is not None
