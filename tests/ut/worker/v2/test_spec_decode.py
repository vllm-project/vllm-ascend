# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, SpeculativeConfig

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


def _make_config(method, declared_config=None):
    """Skip checkpoint I/O while using vLLM's real speculator selection."""
    spec = object.__new__(SpeculativeConfig)
    spec.method = method
    spec.custom_method = None
    spec.enforce_eager = False
    spec.target_model_config = None
    spec.num_speculative_tokens = 1
    spec.index_share_for_mtp_iteration = None
    spec.draft_model_config = SimpleNamespace(
        speculator_config_class=declared_config,
        hf_config=SimpleNamespace(model_type="qwen3_5_mtp"),
    )
    return SimpleNamespace(speculative_config=spec)


def _mock_speculator_module(monkeypatch, module_name, class_name):
    """Mock a dynamically imported speculator module and return its class."""
    module = ModuleType(module_name)
    speculator_cls = MagicMock(name=class_name)
    setattr(module, class_name, speculator_cls)
    monkeypatch.setitem(sys.modules, module_name, module)
    return speculator_cls


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_init_speculator_ngram_uses_current_request_state(monkeypatch, method, enforce_eager):
    config = _make_config(method)
    config.speculative_config.enforce_eager = enforce_eager
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
    if enforce_eager:
        assert draft_config is not config
        assert draft_config.compilation_config is not config.compilation_config
        assert draft_config.compilation_config.mode == CompilationMode.NONE
    else:
        assert draft_config is config


def test_init_speculator_ngram_requires_request_state():
    with pytest.raises(ValueError, match="requires request states"):
        init_speculator(_make_config("ngram_gpu"), torch.device("cpu"))


def test_init_speculator_extract_hidden_states(monkeypatch):
    """Test dispatching extract_hidden_states to the upstream speculator."""
    config = _make_config("extract_hidden_states")
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm.v1.worker.gpu.spec_decode.extract_hidden_states",
        "ExtractHiddenStatesSpeculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


@pytest.mark.parametrize(
    "method,declared_config,module,name",
    [
        ("dspark", None, "dspark", "AscendDSparkSpeculator"),
        ("dflash", None, "dflash", "AscendDFlashSpeculator"),
        ("dflash", "dflash2.config.DFlash2SpeculatorConfig", "dflash2", "AscendDFlash2Speculator"),
        ("dflash", "lilicorr.config.LiLiCorrSpeculatorConfig", "lilicorr", "AscendLiLiCorrSpeculator"),
        ("eagle", None, "eagle", "AscendEagleSpeculator"),
        ("eagle3", None, "eagle", "AscendEagleSpeculator"),
        ("draft_model", None, "standalone_ar", "AscendStandaloneARSpeculator"),
    ],
)
def test_init_speculator_uses_upstream_selection(monkeypatch, method, declared_config, module, name):
    if declared_config is not None:
        declared_config = f"vllm.v1.worker.gpu.spec_decode.{declared_config}"
    config = _make_config(method, declared_config)
    constructor = _mock_speculator_module(monkeypatch, f"{v2_spec_decode.__name__}.{module}.speculator", name)
    device = torch.device("cpu")
    assert init_speculator(config, device) is constructor.return_value
    constructor.assert_called_once_with(config, device)


@pytest.mark.parametrize("enforce_eager", [True, False, None])
def test_draft_compilation_config_respects_enforce_eager(monkeypatch, enforce_eager, use_310p_dispatch):
    config = _make_config("mtp")
    config.speculative_config.enforce_eager = enforce_eager
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[6, 12, 24],
        max_cudagraph_capture_size=24,
    )
    target_compilation = config.compilation_config
    if use_310p_dispatch:
        module, name = "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator", "AscendMTPSpeculator310"
    else:
        module, name = "vllm_ascend.worker.v2.spec_decode.mtp.speculator", "AscendMTPSpeculator"
    speculator_cls = _mock_speculator_module(monkeypatch, module, name)

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


@pytest.mark.parametrize(
    "config,expected",
    [
        (_make_config("suffix"), "SuffixSpeculator"),
        (
            _make_config("dflash", "vllm.v1.worker.gpu.spec_decode.xpress.config.XPressSpeculatorConfig"),
            "XPressSpeculator",
        ),
    ],
)
def test_init_speculator_rejects_unmapped_upstream_implementation(config, expected):
    with pytest.raises(NotImplementedError, match=expected):
        init_speculator(config, torch.device("cpu"))


@pytest.fixture(params=[False, True], ids=["default", "310p"])
def use_310p_dispatch(request, monkeypatch):
    if request.param:
        from vllm.v1.worker.gpu.spec_decode import target_dependent_ar

        # Keep both the routing table and the unrelated CPU helper patches local to the test.
        monkeypatch.setattr(v2_spec_decode, "_ASCEND_SPECULATORS", v2_spec_decode._ASCEND_SPECULATORS.copy())
        monkeypatch.setattr(target_dependent_ar, "speculator", SimpleNamespace(), raising=False)
        path = Path(v2_spec_decode.__file__).resolve().parents[3] / "patch/worker/patch_spec_decode_310.py"
        runpy.run_path(str(path))
    return request.param


@pytest.mark.parametrize("num_mtp_modules", [1, 2])
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
@pytest.mark.parametrize("model_type", ["qwen3_5_mtp", "step3p5_mtp", "gemma4_mtp"])
def test_mtp_routing_uses_actual_native_config(
    monkeypatch, use_310p_dispatch, model_type, num_mtp_modules, num_speculative_tokens
):
    declared_config = "vllm.v1.worker.gpu.spec_decode.gemma4.config.Gemma4MTPSpeculatorConfig"
    config = _make_config("mtp", declared_config if model_type == "gemma4_mtp" else None)
    speculative = config.speculative_config
    speculative.num_speculative_tokens = num_speculative_tokens
    speculative.draft_model_config.hf_config = SimpleNamespace(
        model_type=model_type,
        num_nextn_predict_layers=num_mtp_modules,
        n_predict=num_mtp_modules,
        index_share_for_mtp_iteration=False,
    )
    if model_type == "gemma4_mtp":
        module, name = "vllm_ascend.worker.v2.spec_decode.gemma4.speculator", "AscendGemma4Speculator"
    elif min(num_mtp_modules, num_speculative_tokens) > 1:
        module = "vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator"
        name = "AscendMultiModuleMTPSpeculator"
    elif use_310p_dispatch:
        module, name = "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator", "AscendMTPSpeculator310"
    else:
        module, name = "vllm_ascend.worker.v2.spec_decode.mtp.speculator", "AscendMTPSpeculator"
    constructor = _mock_speculator_module(monkeypatch, module, name)
    device = torch.device("cpu")
    assert init_speculator(config, device) is constructor.return_value
    constructor.assert_called_once_with(config, device)
