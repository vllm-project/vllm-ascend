# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramSpeculatorConfig

from vllm_ascend.worker.v2 import spec_decode as v2_spec_decode
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


def test_rejection_sample_rejects_block_with_synthetic_rates():
    """Reject incompatible verification modes before allocating or launching on NPU."""
    with pytest.raises(AssertionError, match="incompatible with synthetic"):
        rejection_sample(
            target_logits=torch.zeros(2, 1),
            draft_logits=None,
            draft_sampled=torch.zeros(2, dtype=torch.int64),
            cu_num_logits=torch.tensor([0, 2], dtype=torch.int32),
            pos=torch.arange(2, dtype=torch.int32),
            idx_mapping=torch.zeros(1, dtype=torch.int32),
            expanded_idx_mapping=torch.zeros(2, dtype=torch.int32),
            expanded_local_pos=torch.arange(2, dtype=torch.int32),
            temperature=torch.ones(1),
            seed=torch.zeros(1, dtype=torch.int64),
            num_speculative_steps=1,
            synthetic_conditional_rates=torch.ones(1),
            use_block_verification=True,
        )


def _make_config(method: str, speculator_class: str | None, *, custom_method: str | None = None):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            enforce_eager=None,
            custom_method=custom_method,
            speculator_config=SimpleNamespace(speculator_class=speculator_class),
        ),
    )


def _mock_speculator_module(monkeypatch, module_name, class_name):
    """Mock a dynamically imported speculator module and return its class."""
    module = ModuleType(module_name)
    speculator_cls = MagicMock(name=class_name)
    setattr(module, class_name, speculator_cls)
    monkeypatch.setitem(sys.modules, module_name, module)
    return speculator_cls


def test_eager_ngram_forwards_current_request_state(monkeypatch):
    config = _make_config("ngram", NgramSpeculatorConfig.speculator_class)
    config.speculative_config.enforce_eager = True
    config.compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)
    device = torch.device("cpu")
    req_states = object()
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm.v1.worker.gpu.spec_decode.ngram.speculator",
        "NgramGPUSpeculator",
    )
    result = init_speculator(config, device, req_states)
    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once()
    draft_config, draft_device, draft_states = speculator_cls.call_args.args
    assert draft_device == device
    assert draft_states is req_states
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert draft_config is not config
    assert draft_config.compilation_config is not config.compilation_config
    assert draft_config.compilation_config.mode == CompilationMode.NONE


def test_init_speculator_ngram_requires_request_state():
    with pytest.raises(ValueError, match="requires request states"):
        init_speculator(_make_config("ngram_gpu", NgramSpeculatorConfig.speculator_class), torch.device("cpu"))


@pytest.mark.parametrize("enforce_eager", [True, False, None])
def test_draft_compilation_config_respects_enforce_eager(monkeypatch, enforce_eager):
    config = _make_config("mtp", MTPSpeculatorConfig.speculator_class)
    config.speculative_config.enforce_eager = enforce_eager
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[6, 12, 24],
        max_cudagraph_capture_size=24,
    )
    target_compilation = config.compilation_config
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.mtp.speculator",
        "AscendMTPSpeculator",
    )

    init_speculator(config, torch.device("cpu"))

    draft_config = speculator_cls.call_args.args[0]
    draft_compilation = draft_config.compilation_config
    if enforce_eager:
        assert draft_config is not config
        assert draft_compilation.mode == CompilationMode.NONE
    else:
        assert draft_config is config
    assert config.compilation_config is target_compilation
    assert target_compilation.mode == CompilationMode.VLLM_COMPILE
    assert draft_compilation.cudagraph_mode == target_compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert draft_compilation.cudagraph_capture_sizes == target_compilation.cudagraph_capture_sizes == [6, 12, 24]
    assert draft_compilation.static_forward_context is target_compilation.static_forward_context
    assert draft_compilation.static_all_moe_layers is target_compilation.static_all_moe_layers


def test_init_speculator_requires_speculative_config():
    """Test that init_speculator requires speculative_config to be present."""
    config = SimpleNamespace(speculative_config=None)

    with pytest.raises(AssertionError):
        init_speculator(config, torch.device("cpu"))


def _install_310p_spec_decode_patch(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import target_dependent_ar

    # Isolate the three CPU input helpers and the dispatch map between cases.
    monkeypatch.setattr(target_dependent_ar, "speculator", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        v2_spec_decode, "_UPSTREAM_TO_ASCEND_SPECULATOR", v2_spec_decode._UPSTREAM_TO_ASCEND_SPECULATOR.copy()
    )
    path = Path(v2_spec_decode.__file__).resolve().parents[3] / "patch/worker/patch_spec_decode_310.py"
    runpy.run_path(str(path))


def test_310p_plain_mtp_uses_shared_factory(monkeypatch):
    from vllm_ascend.worker.v2 import model_runner

    _install_310p_spec_decode_patch(monkeypatch)
    config = _make_config("mtp", MTPSpeculatorConfig.speculator_class)
    config.speculative_config.enforce_eager = True
    config.compilation_config = SimpleNamespace(mode=CompilationMode.VLLM_COMPILE)
    constructor = _mock_speculator_module(
        monkeypatch, "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator", "AscendMTPSpeculator310"
    )
    assert model_runner.init_speculator is v2_spec_decode.init_speculator is init_speculator
    device = torch.device("cpu")
    assert model_runner.init_speculator(config, device, object()) is constructor.return_value
    draft_config, draft_device = constructor.call_args.args
    assert draft_device == device
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert draft_config.compilation_config.mode == CompilationMode.NONE
    assert draft_config.compilation_config is not config.compilation_config


@pytest.mark.parametrize("use_310p", [False, True])
@pytest.mark.parametrize("num_mtp_modules", [1, 2])
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
@pytest.mark.parametrize("model_type", ["qwen3_5_mtp", "step3p5_mtp", "gemma4_mtp"])
def test_mtp_routing_uses_actual_native_config(
    monkeypatch, use_310p, model_type, num_mtp_modules, num_speculative_tokens
):
    # Bypass config model loading, while retaining the real native predicates.
    speculative = object.__new__(SpeculativeConfig)
    speculative.method = "mtp"
    speculative.custom_method = None
    speculative.enforce_eager = False
    speculative.num_speculative_tokens = num_speculative_tokens
    speculative.index_share_for_mtp_iteration = None
    speculative.draft_model_config = cast(
        Any,
        SimpleNamespace(
            speculator_config_class=(
                "vllm.v1.worker.gpu.spec_decode.gemma4.config.Gemma4MTPSpeculatorConfig"
                if model_type == "gemma4_mtp"
                else None
            ),
            hf_config=SimpleNamespace(
                model_type=model_type,
                num_nextn_predict_layers=num_mtp_modules,
                n_predict=num_mtp_modules,
                index_share_for_mtp_iteration=False,
            ),
        ),
    )
    config = SimpleNamespace(speculative_config=speculative)
    if use_310p:
        _install_310p_spec_decode_patch(monkeypatch)
    if model_type == "gemma4_mtp":
        expected_upstream = "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.Gemma4Speculator"
        module, name = "vllm_ascend.worker.v2.spec_decode.gemma4.speculator", "AscendGemma4Speculator"
    elif min(num_mtp_modules, num_speculative_tokens) > 1:
        expected_upstream = "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator"
        module = "vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator"
        name = "AscendMultiModuleMTPSpeculator"
    elif use_310p:
        expected_upstream = "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator"
        module, name = "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator", "AscendMTPSpeculator310"
    else:
        expected_upstream = "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator"
        module, name = "vllm_ascend.worker.v2.spec_decode.mtp.speculator", "AscendMTPSpeculator"
    assert speculative.speculator_config.speculator_class == expected_upstream
    constructor = _mock_speculator_module(monkeypatch, module, name)
    device = torch.device("cpu")
    assert init_speculator(config, device) is constructor.return_value
    constructor.assert_called_once_with(config, device)


@pytest.mark.parametrize(
    "method,upstream_class,target_module,target_class,with_req_states",
    [
        (
            "ngram_gpu",
            "vllm.v1.worker.gpu.spec_decode.ngram.speculator.NgramGPUSpeculator",
            "vllm.v1.worker.gpu.spec_decode.ngram.speculator",
            "NgramGPUSpeculator",
            True,
        ),
        (
            "extract_hidden_states",
            "vllm.v1.worker.gpu.spec_decode.extract_hidden_states.ExtractHiddenStatesSpeculator",
            "vllm.v1.worker.gpu.spec_decode.extract_hidden_states",
            "ExtractHiddenStatesSpeculator",
            False,
        ),
        (
            "dspark",
            "vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator",
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
            "AscendDSparkSpeculator",
            False,
        ),
        (
            "dflash",
            "vllm.v1.worker.gpu.spec_decode.lilicorr.speculator.LiLiCorrSpeculator",
            "vllm_ascend.worker.v2.spec_decode.lilicorr.speculator",
            "AscendLiLiCorrSpeculator",
            False,
        ),
        (
            "dflash",
            "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator",
            "vllm_ascend.worker.v2.spec_decode.dflash2.speculator",
            "AscendDFlash2Speculator",
            False,
        ),
        (
            "dflash",
            "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator",
            "vllm_ascend.worker.v2.spec_decode.dflash.speculator",
            "AscendDFlashSpeculator",
            False,
        ),
        (
            "mtp",
            "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.Gemma4Speculator",
            "vllm_ascend.worker.v2.spec_decode.gemma4.speculator",
            "AscendGemma4Speculator",
            False,
        ),
        (
            "mtp",
            "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator.MultiModuleMTPSpeculator",
            "vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator",
            "AscendMultiModuleMTPSpeculator",
            False,
        ),
        (
            "mtp",
            "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator",
            "vllm_ascend.worker.v2.spec_decode.mtp.speculator",
            "AscendMTPSpeculator",
            False,
        ),
        (
            "eagle",
            "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator",
            "vllm_ascend.worker.v2.spec_decode.eagle.speculator",
            "AscendEagleSpeculator",
            False,
        ),
        (
            "draft_model",
            "vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator.StandaloneARSpeculator",
            "vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator",
            "AscendStandaloneARSpeculator",
            False,
        ),
    ],
)
def test_init_speculator_main_dispatches_via_speculator_class(
    monkeypatch,
    method,
    upstream_class,
    target_module,
    target_class,
    with_req_states,
):
    """Main path selects the Ascend/upstream speculator solely from speculator_class."""
    config = _make_config(method, upstream_class)
    constructor = _mock_speculator_module(monkeypatch, target_module, target_class)
    device = torch.device("cpu")
    req_states = object() if with_req_states else None

    result = init_speculator(config, device, req_states)

    assert result is constructor.return_value
    if with_req_states:
        constructor.assert_called_once_with(config, device, req_states)
    else:
        constructor.assert_called_once_with(config, device)


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
def test_init_speculator_main_actual_native_config_eagle(monkeypatch, method):
    """Verify real upstream SpeculativeConfig.speculator_config resolves EAGLE."""
    speculative = object.__new__(SpeculativeConfig)
    speculative.method = method
    speculative.custom_method = None
    speculative.enforce_eager = False
    speculative.draft_model_config = cast(Any, SimpleNamespace(speculator_config_class=None))
    assert (
        speculative.speculator_config.speculator_class
        == "vllm.v1.worker.gpu.spec_decode.eagle.speculator.EagleSpeculator"
    )
    config = SimpleNamespace(speculative_config=speculative)
    constructor = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.eagle.speculator",
        "AscendEagleSpeculator",
    )
    device = torch.device("cpu")

    assert init_speculator(config, device) is constructor.return_value
    constructor.assert_called_once_with(config, device)


def test_init_speculator_main_rejects_unadapted_upstream_speculator():
    """Unadapted concrete speculator classes fail with an explicit class-level NotImplementedError."""
    speculator_class = "vllm.v1.worker.gpu.spec_decode.suffix.speculator.SuffixSpeculator"
    config = _make_config("suffix", speculator_class=speculator_class)

    with pytest.raises(NotImplementedError) as exc_info:
        init_speculator(config, torch.device("cpu"))

    assert speculator_class in str(exc_info.value)
    assert "suffix is not supported yet" not in str(exc_info.value)


def test_init_speculator_main_rejects_none_speculator_class():
    """Speculator configs with speculator_class=None keep the method-level NotImplementedError."""
    config = _make_config("unsupported", speculator_class=None)

    with pytest.raises(NotImplementedError, match="unsupported is not supported yet"):
        init_speculator(config, torch.device("cpu"))


def test_init_speculator_main_actual_native_config_xpress_rejected_vs_dflash(monkeypatch):
    """Real XPressSpeculatorConfig (method='dflash') rejects XPress explicitly instead of generic DFlash."""
    from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
    from vllm.v1.worker.gpu.spec_decode.xpress.config import XPressSpeculatorConfig

    assert issubclass(XPressSpeculatorConfig, DFlashSpeculatorConfig)
    assert XPressSpeculatorConfig.base_method == "dflash"

    dflash_ctor = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator",
        "AscendDFlashSpeculator",
    )
    device = torch.device("cpu")

    xpress_spec = object.__new__(SpeculativeConfig)
    xpress_spec.method = "dflash"
    xpress_spec.custom_method = None
    xpress_spec.enforce_eager = False
    xpress_spec.draft_model_config = cast(
        Any,
        SimpleNamespace(speculator_config_class="vllm.v1.worker.gpu.spec_decode.xpress.config.XPressSpeculatorConfig"),
    )
    assert isinstance(xpress_spec.speculator_config, XPressSpeculatorConfig)
    expected_xpress_cls = "vllm.v1.worker.gpu.spec_decode.xpress.speculator.XPressSpeculator"
    assert xpress_spec.speculator_config.speculator_class == expected_xpress_cls

    with pytest.raises(NotImplementedError) as exc_info:
        init_speculator(SimpleNamespace(speculative_config=xpress_spec), device)

    assert expected_xpress_cls in str(exc_info.value)
    assert "dflash is not supported yet" not in str(exc_info.value)
    dflash_ctor.assert_not_called()

    dflash_spec = object.__new__(SpeculativeConfig)
    dflash_spec.method = "dflash"
    dflash_spec.custom_method = None
    dflash_spec.enforce_eager = False
    dflash_spec.draft_model_config = cast(Any, SimpleNamespace(speculator_config_class=None))
    assert type(dflash_spec.speculator_config) is DFlashSpeculatorConfig

    dflash_cfg = SimpleNamespace(speculative_config=dflash_spec)
    assert init_speculator(dflash_cfg, device) is dflash_ctor.return_value
    dflash_ctor.assert_called_once_with(dflash_cfg, device)


@pytest.fixture
def custom_speculators(monkeypatch):
    base_mod: Any = ModuleType("vllm.v1.worker.gpu.spec_decode.speculator")

    class DummyBaseSpeculator:
        pass

    base_mod.BaseSpeculator = DummyBaseSpeculator
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.spec_decode.speculator", base_mod)

    class CustomNoReqStates(DummyBaseSpeculator):
        def __init__(self, vllm_config, device):
            self.args = (vllm_config, device)

    class CustomWithReqStates(DummyBaseSpeculator):
        def __init__(self, vllm_config, device, req_states):
            self.args = (vllm_config, device, req_states)

    class InvalidSpeculator:
        def __init__(self, vllm_config, device):
            pass

    custom_mod: Any = ModuleType("tests.fake_custom_speculator")
    custom_mod.CustomNoReqStates = CustomNoReqStates
    custom_mod.CustomWithReqStates = CustomWithReqStates
    custom_mod.InvalidSpeculator = InvalidSpeculator
    monkeypatch.setitem(sys.modules, "tests.fake_custom_speculator", custom_mod)
    return custom_mod


def test_init_speculator_main_supports_custom_speculator(custom_speculators):
    device = torch.device("cpu")
    cfg_no_req = _make_config(
        "eagle",
        custom_method="my_custom",
        speculator_class="tests.fake_custom_speculator.CustomNoReqStates",
    )
    inst_no_req = init_speculator(cfg_no_req, device)
    assert isinstance(inst_no_req, custom_speculators.CustomNoReqStates)
    assert inst_no_req.args == (cfg_no_req, device)

    req_states = object()
    cfg_with_req = _make_config(
        "ngram",
        custom_method="my_custom_req",
        speculator_class="tests.fake_custom_speculator.CustomWithReqStates",
    )
    inst_with_req = init_speculator(cfg_with_req, device, req_states)
    assert isinstance(inst_with_req, custom_speculators.CustomWithReqStates)
    assert inst_with_req.args == (cfg_with_req, device, req_states)

    with pytest.raises(ValueError, match="requires request states"):
        init_speculator(cfg_with_req, device, None)

    cfg_invalid = _make_config(
        "eagle",
        custom_method="my_invalid",
        speculator_class="tests.fake_custom_speculator.InvalidSpeculator",
    )
    with pytest.raises(TypeError, match="must be a subclass of"):
        init_speculator(cfg_invalid, device)


def test_310p_custom_mtp_receives_request_state(monkeypatch, custom_speculators):
    _install_310p_spec_decode_patch(monkeypatch)
    config = _make_config("mtp", "tests.fake_custom_speculator.CustomWithReqStates", custom_method="my_custom_mtp")
    req_states = object()
    draft = init_speculator(config, torch.device("cpu"), req_states)
    assert isinstance(draft, custom_speculators.CustomWithReqStates)
    assert draft.args[2] is req_states
