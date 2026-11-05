# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
from vllm.v1.worker.gpu.spec_decode import speculator as upstream

from vllm_ascend import mrv2_utils
from vllm_ascend.worker.v2.spec_decode import suffix
from vllm_ascend.worker.v2.spec_decode.suffix import validate_suffix_runtime


def _with_hooks():
    return SimpleNamespace(
        add_request=lambda *args: None,
        on_requests_finished=lambda *args: None,
        reset_state=lambda *args: None,
    )


@pytest.mark.parametrize("missing", ["add_request", "on_requests_finished", "reset_state", "config"])
def test_incomplete_upstream_protocol_is_rejected(monkeypatch, missing):
    base = _with_hooks()
    config = SimpleNamespace(suffix_decoding_corpus_tokens=0)
    if missing == "config":
        del config.suffix_decoding_corpus_tokens
    else:
        delattr(base, missing)
    monkeypatch.setattr(upstream, "BaseSpeculator", base)
    with pytest.raises(ValueError, match="requires vLLM #60339"):
        validate_suffix_runtime(config)


def test_complete_upstream_protocol_is_accepted(monkeypatch):
    monkeypatch.setattr(upstream, "BaseSpeculator", _with_hooks())
    monkeypatch.setattr(suffix, "import_module", lambda name: SimpleNamespace(SuffixSpeculator=type("Suffix", (), {})))
    validate_suffix_runtime(SimpleNamespace(suffix_decoding_corpus_tokens=0))


def test_missing_concrete_upstream_class_is_rejected(monkeypatch):
    monkeypatch.setattr(upstream, "BaseSpeculator", _with_hooks())
    monkeypatch.setattr(suffix, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ValueError, match="requires vLLM #60339"):
        validate_suffix_runtime(SimpleNamespace(suffix_decoding_corpus_tokens=0))


@pytest.mark.parametrize("missing", ["vllm.v1.worker.gpu.spec_decode.suffix", "unrelated_dependency"])
def test_missing_upstream_module_does_not_hide_dependency_errors(monkeypatch, missing):
    monkeypatch.setattr(upstream, "BaseSpeculator", _with_hooks())

    def import_missing(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(suffix, "import_module", import_missing)
    expected_error = ValueError if missing.endswith(".suffix") else ModuleNotFoundError
    with pytest.raises(expected_error):
        validate_suffix_runtime(SimpleNamespace(suffix_decoding_corpus_tokens=0))


@pytest.mark.parametrize("env_value", [None, True])
def test_v2_fails_before_runner_construction_on_old_upstream(monkeypatch, env_value):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)
    monkeypatch.setattr(upstream, "BaseSpeculator", SimpleNamespace())
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix"))
    with pytest.raises(ValueError, match="requires vLLM #60339"):
        mrv2_utils._validate_v2_model_runner(config)


def test_explicit_v1_does_not_require_the_new_protocol(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setattr(upstream, "BaseSpeculator", SimpleNamespace())
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix"))
    mrv2_utils._validate_v2_model_runner(config)
    assert not mrv2_utils.use_v2_model_runner(config)


@pytest.mark.parametrize("env_value", [None, False, True])
def test_suffix_defaults_to_v2_and_honors_explicit_v1(monkeypatch, env_value):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix", suffix_decoding_corpus_tokens=0))
    assert mrv2_utils.use_v2_model_runner(config) is (env_value is not False)


def test_default_v2_validates_the_complete_protocol(monkeypatch):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(upstream, "BaseSpeculator", _with_hooks())
    monkeypatch.setattr(suffix, "import_module", lambda name: SimpleNamespace(SuffixSpeculator=type("Suffix", (), {})))
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix", suffix_decoding_corpus_tokens=0))
    mrv2_utils._validate_v2_model_runner(config)
