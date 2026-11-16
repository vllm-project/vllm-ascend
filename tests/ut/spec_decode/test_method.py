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
)


def _config(
    method,
    *,
    architectures=None,
    gemma4=False,
    step3p5=False,
    model_type=None,
    multi_module=False,
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
        use_multi_module_mtp=lambda: multi_module,
    )
    return SimpleNamespace(speculative_config=speculative_config)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config("ngram"), SpecDecodeFamily.NGRAM),
        (_config("ngram_gpu"), SpecDecodeFamily.NGRAM_GPU),
        (_config("suffix"), SpecDecodeFamily.UNKNOWN),
        (_config("medusa"), SpecDecodeFamily.UNKNOWN),
        (_config("dspark"), SpecDecodeFamily.DSPARK),
        (_config("draft_model"), SpecDecodeFamily.DRAFT_MODEL),
        (_config("extract_hidden_states"), SpecDecodeFamily.EXTRACT_HIDDEN_STATES),
        (_config("eagle"), SpecDecodeFamily.EAGLE),
        (_config("eagle3"), SpecDecodeFamily.EAGLE),
        (_config("mtp"), SpecDecodeFamily.MTP),
        (_config("eagle", multi_module=True), SpecDecodeFamily.MULTI_MODULE_MTP),
        (_config("mtp", step3p5=True, multi_module=True), SpecDecodeFamily.MULTI_MODULE_MTP),
        (_config("mtp", multi_module=True), SpecDecodeFamily.MTP),
        (_config("dflash", architectures=["LiLiCorrDraftModel"]), SpecDecodeFamily.LILICORR),
        (_config("mtp", gemma4=True), SpecDecodeFamily.GEMMA4_MTP),
        (_config("mtp", step3p5=True), SpecDecodeFamily.UNKNOWN),
        (
            _config("mtp", model_type="glm5_next_mtp"),
            SpecDecodeFamily.MTP,
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
            SpecDecodeFamily.UNKNOWN,
        ),
    ],
)
def test_resolve_spec_decode_family_preserves_mtp_precedence(config, expected):
    assert resolve_spec_decode_family(config) is expected


def test_single_step_step3p5_preserves_eagle_fallback():
    config = _config("mtp", step3p5=True)
    config.speculative_config.use_eagle = lambda: True
    assert resolve_spec_decode_family(config) is SpecDecodeFamily.EAGLE


def test_310p_mtp_routing_uses_family_helper(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode as base_spec_decode
    from vllm_ascend.worker.v2 import model_runner as base_model_runner

    original_init = base_spec_decode.init_speculator
    original_model_runner_init = base_model_runner.init_speculator
    patch_module = importlib.import_module("vllm_ascend.patch.worker.patch_spec_decode_310")
    monkeypatch.setattr(base_spec_decode, "init_speculator", original_init)
    monkeypatch.setattr(base_model_runner, "init_speculator", original_model_runner_init)

    module = ModuleType("vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator")
    speculator_cls = MagicMock()
    module.__dict__["AscendMTPSpeculator310"] = speculator_cls
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        patch_module,
        "resolve_spec_decode_family",
        lambda _config: SpecDecodeFamily.MTP,
    )

    config = SimpleNamespace(speculative_config=object())
    result = patch_module._init_speculator_310p(config, "npu")

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, "npu")


def test_310p_non_mtp_routing_uses_parent_factory(monkeypatch):
    import vllm_ascend.worker.v2.spec_decode as base_spec_decode
    from vllm_ascend.worker.v2 import model_runner as base_model_runner

    original_init = base_spec_decode.init_speculator
    original_model_runner_init = base_model_runner.init_speculator
    patch_module = importlib.import_module("vllm_ascend.patch.worker.patch_spec_decode_310")
    monkeypatch.setattr(base_spec_decode, "init_speculator", original_init)
    monkeypatch.setattr(base_model_runner, "init_speculator", original_model_runner_init)
    monkeypatch.setattr(patch_module, "resolve_spec_decode_family", lambda _config: SpecDecodeFamily.EAGLE)
    fallback = MagicMock()
    monkeypatch.setattr(patch_module, "_orig_init_speculator", fallback)

    config = SimpleNamespace(speculative_config=object())
    req_states = object()
    result = patch_module._init_speculator_310p(config, "npu", req_states)

    assert result is fallback.return_value
    fallback.assert_called_once_with(config, "npu", req_states)


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
    "config",
    [
        SimpleNamespace(speculative_config=None),
        _config("unsupported"),
    ],
)
def test_resolve_spec_decode_family_returns_unknown(config):
    assert resolve_spec_decode_family(config) is SpecDecodeFamily.UNKNOWN


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
