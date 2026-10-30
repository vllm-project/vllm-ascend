# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the actual factory source while isolating device constructors."""

import ast
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def factory(monkeypatch):
    spec_dir = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"
    # Execute the actual Ascend model-boundary predicate, not a test replica.
    utils_tree = ast.parse((spec_dir.parents[2] / "utils.py").read_text())
    predicate = next(
        node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_deepseek_v41"
    )
    utils = ModuleType("vllm_ascend.utils")
    utils.__dict__["Any"] = object
    exec(compile(ast.Module(body=[predicate], type_ignores=[]), "utils.py", "exec"), utils.__dict__)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    logging = ModuleType("vllm.logger")
    logging.__dict__["logger"] = Mock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logging)
    config = ModuleType("vllm.config")
    CompilationMode = Enum("CompilationMode", ["NONE", "VLLM_COMPILE"])
    config.__dict__["CompilationMode"] = CompilationMode
    config.__dict__["VllmConfig"] = object
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    ascend_config = ModuleType("vllm_ascend.ascend_config")
    effective = SimpleNamespace(draft_window_size=None)
    ascend_config.__dict__["get_ascend_config"] = lambda: effective
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", ascend_config)
    name = "vllm_ascend.worker.v2.spec_decode.draft_window_policy"
    spec = importlib.util.spec_from_file_location(name, spec_dir / "draft_window_policy.py")
    assert spec is not None and spec.loader is not None
    policy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, policy)
    spec.loader.exec_module(policy)
    spec = importlib.util.spec_from_file_location("_draft_window_factory_under_test", spec_dir / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, effective, CompilationMode


def _config(method="dspark", *, eager=True, model_type="qwen3", draft_arch="Qwen3DSparkModel"):
    target = SimpleNamespace(
        model="target-weights", model_weights=None, revision="r1", hf_config=SimpleNamespace(model_type=model_type)
    )
    draft = SimpleNamespace(
        model="independent-draft",
        model_weights=None,
        revision="r1",
        hf_config=SimpleNamespace(model_type=model_type),
        architectures=[draft_arch],
    )
    spec = SimpleNamespace(
        method=method,
        enforce_eager=eager,
        draft_model_config=draft,
        uses_dynamic_speculative_decoding=Mock(return_value=False),
        use_dspark=lambda: method == "dspark",
        use_dflash=lambda: method == "dflash",
        use_eagle=lambda: method in {"eagle", "eagle3", "mtp"},
        use_gemma4_mtp=lambda: False,
        use_step3p5_mtp=lambda: False,
    )
    return SimpleNamespace(
        model_config=target,
        speculative_config=spec,
        compilation_config=SimpleNamespace(mode="target-graph", cudagraph_mode="target-graph", context=object()),
    )


def _constructor(monkeypatch, route="dspark", class_name="AscendDSparkSpeculator"):
    name = f"vllm_ascend.worker.v2.spec_decode.{route}.speculator"
    module = ModuleType(name)
    ctor = Mock(return_value=SimpleNamespace())
    setattr(module, class_name, ctor)
    monkeypatch.setitem(sys.modules, name, module)
    return ctor


@pytest.mark.parametrize("eager", [None, False, True])
def test_off_has_no_window_state_and_preserves_target(factory, monkeypatch, eager):
    module, effective, modes = factory
    config = _config(eager=eager)
    config.speculative_config.uses_dynamic_speculative_decoding.side_effect = AssertionError("OFF inspected dynamic K")
    ctor = _constructor(monkeypatch)
    result = module.init_speculator(config, "cpu")
    assert vars(result) == {}
    module.logger.warning.assert_not_called()
    draft_config = ctor.call_args.args[0]
    assert config.compilation_config.mode == "target-graph"
    assert draft_config.compilation_config.context is config.compilation_config.context
    if eager:
        assert draft_config is not config
        assert draft_config.compilation_config.mode is modes.NONE
    else:
        assert draft_config is config


def test_on_attaches_policy_using_effective_config(factory, monkeypatch):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    config = _config()
    config.additional_config = {"draft_window_size": None}
    ctor = _constructor(monkeypatch)
    result = module.init_speculator(config, "cpu")
    assert result is ctor.return_value
    assert result.draft_window_policy.enabled
    assert result.draft_window_policy.effective_window_size == 1024
    assert config.compilation_config.mode == "target-graph"
    config.speculative_config.uses_dynamic_speculative_decoding.assert_called_once_with()


@pytest.mark.parametrize("eager", [False, None])
def test_on_requires_only_draft_eager(factory, monkeypatch, eager):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    ctor = _constructor(monkeypatch)
    with pytest.raises(NotImplementedError, match="enforce_eager"):
        module.init_speculator(_config(eager=eager), "cpu")
    ctor.assert_not_called()


@pytest.mark.parametrize("value", [-1, "invalid"])
def test_mtp_forced_off_before_numeric_or_dynamic_checks(factory, monkeypatch, value):
    module, effective, _ = factory
    effective.draft_window_size = value
    config = _config(method="mtp", eager=False)
    config.speculative_config.uses_dynamic_speculative_decoding.side_effect = AssertionError("MTP inspected dynamic K")
    ctor = _constructor(monkeypatch, "mtp", "AscendMTPSpeculator")
    result = module.init_speculator(config, "cpu")
    assert result is ctor.return_value
    assert vars(result) == {}
    module.logger.warning.assert_called_once()
    assert module.logger.warning.call_args.args[1] == "MTP"


@pytest.mark.parametrize(
    "model_type,arch", [("deepseek_v4", "DSparkDraftModel"), ("deepseek_v41", "DSparkV41DraftModel")]
)
def test_builtin_v4_category_is_forced_off(factory, monkeypatch, model_type, arch):
    module, effective, _ = factory
    effective.draft_window_size = -1
    config = _config(model_type=model_type, draft_arch=arch, eager=False)
    config.speculative_config.draft_model_config.model = config.model_config.model
    config.speculative_config.uses_dynamic_speculative_decoding.side_effect = AssertionError("forced OFF inspected K")
    ctor = _constructor(monkeypatch)
    assert module.init_speculator(config, "cpu") is ctor.return_value
    assert vars(ctor.return_value) == {}
    module.logger.warning.assert_called_once()


@pytest.mark.parametrize("difference", ["path", "weights", "revision", "commit"])
def test_builtin_v4_copies_and_packaging_do_not_change_forced_off(factory, monkeypatch, difference):
    module, effective, _ = factory
    effective.draft_window_size = -1
    config = _config(model_type="deepseek_v4", draft_arch="DSparkDraftModel", eager=False)
    draft = config.speculative_config.draft_model_config
    if difference == "path":
        draft.model = "separately-copied-draft"
    elif difference == "weights":
        draft.model_weights = "separately-packaged-weights"
    elif difference == "revision":
        draft.revision = "r2"
    else:
        config.model_config.hf_config._commit_hash = "target-revision"
        draft.hf_config._commit_hash = "draft-package-revision"
    ctor = _constructor(monkeypatch)
    assert module.init_speculator(config, "cpu") is ctor.return_value
    assert vars(ctor.return_value) == {}
    module.logger.warning.assert_called_once()


@pytest.mark.parametrize("difference", ["prefix", "unknown_architecture", "draft_type"])
def test_v4_identity_requires_actual_model_category(factory, monkeypatch, difference):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    config = _config(model_type="deepseek_v4", draft_arch="DSparkDraftModel")
    draft = config.speculative_config.draft_model_config
    if difference == "prefix":
        config.model_config.hf_config.model_type = "deepseek_v4_independent"
    elif difference == "unknown_architecture":
        draft.architectures = ["UnregisteredDraftModel"]
    else:
        draft.hf_config.model_type = "qwen3"
    ctor = _constructor(monkeypatch)
    with pytest.raises(ValueError, match="model-route identity"):
        module.init_speculator(config, "cpu")
    ctor.assert_not_called()


@pytest.mark.parametrize(
    "method,arch,dynamic",
    [
        ("eagle", "EagleDraftModel", False),
        ("dflash", "UnknownDFlashDraftModel", False),
        ("eagle3", "EagleDraftModel", True),
    ],
)
def test_unsupported_window_rejected_before_constructor(factory, monkeypatch, method, arch, dynamic):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    config = _config(method=method, draft_arch=arch)
    config.speculative_config.uses_dynamic_speculative_decoding.return_value = dynamic
    route, cls = ("dflash", "AscendDFlashSpeculator") if method == "dflash" else ("eagle", "AscendEagleSpeculator")
    ctor = _constructor(monkeypatch, route, cls)
    with pytest.raises(NotImplementedError):
        module.init_speculator(config, "cpu")
    ctor.assert_not_called()


def test_dflash2_eager_window_keeps_its_own_constructor_and_target_graph(factory, monkeypatch):
    module, effective, modes = factory
    effective.draft_window_size = 1024
    config = _config(method="dflash", draft_arch="DFlash2DraftModel")
    ctor = _constructor(monkeypatch, "dflash2", "AscendDFlash2Speculator")
    result = module.init_speculator(config, "cpu")
    assert result is ctor.return_value
    assert result.draft_window_policy.enabled
    assert result.draft_window_policy.normalized_method == "dflash"
    assert ctor.call_args.args[0].compilation_config.mode is modes.NONE
    assert config.compilation_config.mode == "target-graph"


@pytest.mark.parametrize("architectures", [[], ["UnknownDFlashDraftModel"], ["DFlashDraftModel", "DFlash2DraftModel"]])
def test_unknown_or_ambiguous_dflash_window_architecture_is_rejected_early(factory, monkeypatch, architectures):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    config = _config(method="dflash")
    config.speculative_config.draft_model_config.architectures = architectures
    ctor = _constructor(monkeypatch, "dflash", "AscendDFlashSpeculator")
    dflash2 = _constructor(monkeypatch, "dflash2", "AscendDFlash2Speculator")
    with pytest.raises(NotImplementedError, match="architecture"):
        module.init_speculator(config, "cpu")
    ctor.assert_not_called()
    # OFF still delegates to the selected baseline's own capability checks.
    effective.draft_window_size = None
    expected = dflash2 if "DFlash2DraftModel" in architectures else ctor
    assert module.init_speculator(config, "cpu") is expected.return_value
    assert not hasattr(expected.return_value, "draft_window_policy")


def test_ascend_final_v41_architecture_forces_only_window_off(factory, monkeypatch):
    module, effective, _ = factory
    effective.draft_window_size = "invalid"
    config = _config(model_type="deepseek_v41", draft_arch="DeepseekV41DSparkModel", eager=False)
    config.speculative_config.uses_dynamic_speculative_decoding.side_effect = AssertionError("forced OFF inspected K")
    ctor = _constructor(monkeypatch)
    assert module.init_speculator(config, "cpu") is ctor.return_value
    assert vars(ctor.return_value) == {}
    module.logger.warning.assert_called_once()


def test_v41_nested_text_identity_uses_existing_model_boundary(factory, monkeypatch):
    module, effective, _ = factory
    effective.draft_window_size = -1
    config = _config(model_type="deepseek_v41", draft_arch="DeepseekV41DSparkModel", eager=False)
    config.model_config.hf_config = SimpleNamespace(
        model_type="released_multimodal_wrapper", text_config=SimpleNamespace(model_type="deepseek_v41_text")
    )
    ctor = _constructor(monkeypatch)
    assert module.init_speculator(config, "cpu") is ctor.return_value
    assert vars(ctor.return_value) == {}


def test_recognized_external_qwen_draft_with_v4_target_follows_switch(factory, monkeypatch):
    module, effective, _ = factory
    effective.draft_window_size = 1024
    config = _config(model_type="deepseek_v4", draft_arch="Qwen3DSparkModel")
    config.speculative_config.draft_model_config.hf_config.model_type = "qwen3"
    ctor = _constructor(monkeypatch)
    result = module.init_speculator(config, "cpu")
    assert result is ctor.return_value
    assert result.draft_window_policy.enabled
    module.logger.warning.assert_not_called()
