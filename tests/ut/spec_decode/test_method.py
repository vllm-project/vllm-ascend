# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.worker.spec_decode_family import (
    SpecDecodeFamily,
    resolve_spec_decode_family,
    uses_ascend_mtp_speculator,
)


def _config(
    method,
    *,
    architectures=None,
    gemma4=False,
    step3p5=False,
    model_type=None,
):
    draft_model_config = SimpleNamespace(
        architectures=architectures or [],
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures or []),
    )
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=draft_model_config,
        use_gemma4_mtp=lambda: gemma4,
        use_step3p5_mtp=lambda: step3p5,
    )
    return SimpleNamespace(speculative_config=speculative_config)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config("ngram"), SpecDecodeFamily.NGRAM),
        (_config("ngram_gpu"), SpecDecodeFamily.NGRAM_GPU),
        (_config("suffix"), SpecDecodeFamily.SUFFIX),
        (_config("medusa"), SpecDecodeFamily.MEDUSA),
        (_config("dspark"), SpecDecodeFamily.DSPARK),
        (_config("draft_model"), SpecDecodeFamily.DRAFT_MODEL),
        (_config("extract_hidden_states"), SpecDecodeFamily.EXTRACT_HIDDEN_STATES),
        (_config("eagle"), SpecDecodeFamily.EAGLE),
        (_config("eagle3"), SpecDecodeFamily.EAGLE),
        (_config("mtp"), SpecDecodeFamily.MTP),
        (_config("mtp", gemma4=True), SpecDecodeFamily.GEMMA4_MTP),
        (_config("mtp", step3p5=True), SpecDecodeFamily.STEP3P5_MTP),
        (
            _config("mtp", model_type="glm5_next_mtp"),
            SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP,
        ),
        (_config("dflash"), SpecDecodeFamily.DFLASH),
        (
            _config("dflash", architectures=["DFlash2DraftModel"]),
            SpecDecodeFamily.DFLASH2,
        ),
    ],
)
def test_resolve_spec_decode_family(config, expected):
    assert resolve_spec_decode_family(config) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config("mtp", gemma4=True, step3p5=True), SpecDecodeFamily.GEMMA4_MTP),
        (
            _config("mtp", step3p5=True, model_type="glm5_next_mtp"),
            SpecDecodeFamily.STEP3P5_MTP,
        ),
    ],
)
def test_resolve_spec_decode_family_preserves_mtp_precedence(config, expected):
    assert resolve_spec_decode_family(config) is expected


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (SpecDecodeFamily.MTP, True),
        (SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP, True),
        (SpecDecodeFamily.GEMMA4_MTP, False),
        (SpecDecodeFamily.STEP3P5_MTP, False),
        (SpecDecodeFamily.EAGLE, False),
    ],
)
def test_uses_ascend_mtp_speculator(family, expected):
    assert uses_ascend_mtp_speculator(family) is expected


def test_310p_mtp_routing_uses_family_helper(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode as base_spec_decode

    original_init = base_spec_decode.init_speculator
    patch_module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_spec_decode_310")
    monkeypatch.setattr(base_spec_decode, "init_speculator", original_init)

    module = ModuleType("vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator")
    speculator_cls = MagicMock()
    module.__dict__["AscendMTPSpeculator310"] = speculator_cls
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        patch_module,
        "resolve_spec_decode_family",
        lambda _config: SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP,
    )
    monkeypatch.setattr(
        patch_module,
        "uses_ascend_mtp_speculator",
        lambda family: family is SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP,
    )

    config = SimpleNamespace(speculative_config=object())
    result = patch_module._init_speculator_310p(config, "npu")

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, "npu")


def test_310p_non_mtp_routing_uses_parent_factory(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode as base_spec_decode

    original_init = base_spec_decode.init_speculator
    patch_module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_spec_decode_310")
    monkeypatch.setattr(base_spec_decode, "init_speculator", original_init)
    monkeypatch.setattr(patch_module, "uses_ascend_mtp_speculator", lambda _family: False)
    fallback = MagicMock()
    monkeypatch.setattr(patch_module, "_orig_init_speculator", fallback)

    config = SimpleNamespace(speculative_config=object())
    result = patch_module._init_speculator_310p(config, "npu")

    assert result is fallback.return_value
    fallback.assert_called_once_with(config, "npu")


def test_resolve_spec_decode_family_uses_dispatch_method_override():
    config = _config("mtp")

    assert resolve_spec_decode_family(config, method="dspark") is SpecDecodeFamily.DSPARK


@pytest.mark.parametrize(
    ("owner_name", "attribute", "value"),
    [
        ("draft", "architecture", "DFlash2DraftModel"),
        ("draft", "architectures", ["DFlash2DraftModel"]),
        ("hf_config", "architectures", ["DFlash2DraftModel"]),
        ("hf_text_config", "architectures", ["DFlash2DraftModel"]),
        ("model_arch_config", "architectures", ["DFlash2DraftModel"]),
    ],
)
def test_dflash2_architecture_sources(owner_name, attribute, value):
    draft_model_config = SimpleNamespace(
        architecture=None,
        architectures=[],
        hf_config=SimpleNamespace(architectures=[]),
        hf_text_config=SimpleNamespace(architectures=[]),
        model_arch_config=SimpleNamespace(architectures=[]),
    )
    setattr(draft_model_config if owner_name == "draft" else getattr(draft_model_config, owner_name), attribute, value)
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="dflash", draft_model_config=draft_model_config))

    assert resolve_spec_decode_family(config) is SpecDecodeFamily.DFLASH2


@pytest.mark.parametrize(
    "owner_name",
    ["draft", "hf_config", "hf_text_config", "model_arch_config"],
)
def test_multi_kv_mtp_architecture_sources(owner_name):
    draft_model_config = SimpleNamespace(
        architecture=None,
        architectures=[],
        hf_config=SimpleNamespace(model_type=None, architectures=[]),
        hf_text_config=SimpleNamespace(architectures=[]),
        model_arch_config=SimpleNamespace(architectures=[]),
    )
    target = draft_model_config if owner_name == "draft" else getattr(draft_model_config, owner_name)
    target.architectures = "Glm5NextMTPModel"
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="mtp", draft_model_config=draft_model_config))

    assert resolve_spec_decode_family(config) is SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(speculative_config=None),
        _config("unsupported"),
    ],
)
def test_resolve_spec_decode_family_returns_unknown(config):
    assert resolve_spec_decode_family(config) is SpecDecodeFamily.UNKNOWN


def test_mrv1_dispatch_uses_shared_family_selector(monkeypatch):
    import vllm_ascend.worker.v1.spec_decode as mrv1_spec_decode

    config = _config("configured_method")
    runner = object()
    proposer = object()
    seen = []

    def resolve(vllm_config, method=None):
        seen.append((vllm_config, method))
        return SpecDecodeFamily.NGRAM

    monkeypatch.setattr(mrv1_spec_decode, "resolve_spec_decode_family", resolve)
    monkeypatch.setattr(mrv1_spec_decode, "AscendNgramProposer", lambda *args: proposer)

    assert mrv1_spec_decode.get_spec_decode_method("requested_method", config, "npu", runner) is proposer
    assert seen == [(config, "requested_method")]


def test_mrv2_dispatch_uses_shared_family_selector(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode as mrv2_spec_decode
    from vllm_ascend.worker.v2.spec_decode.dspark import speculator as dspark_speculator

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="configured_method", enforce_eager=False),
        compilation_config=SimpleNamespace(),
    )
    device = object()
    speculator = object()
    seen = []

    def resolve(vllm_config):
        seen.append(vllm_config)
        return SpecDecodeFamily.DSPARK

    monkeypatch.setattr(mrv2_spec_decode, "resolve_spec_decode_family", resolve)
    monkeypatch.setattr(dspark_speculator, "AscendDSparkSpeculator", lambda *args: speculator)

    assert mrv2_spec_decode.init_speculator(config, device) is speculator
    assert seen == [config]
