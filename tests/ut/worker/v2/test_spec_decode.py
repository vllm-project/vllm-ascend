# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

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


def _make_config(
    method,
    use_dspark=False,
    use_dflash=False,
    use_eagle=False,
    architectures=None,
    use_gemma4_mtp=False,
    use_step3p5_mtp=False,
):
    """Create a minimal speculative decoding config for init_speculator tests."""
    speculative_config = SimpleNamespace(
        method=method,
        enforce_eager=None,
        use_dspark=MagicMock(return_value=use_dspark),
        use_dflash=MagicMock(return_value=use_dflash),
        use_eagle=MagicMock(return_value=use_eagle),
        uses_draft_model=MagicMock(return_value=False),
        use_gemma4_mtp=MagicMock(return_value=use_gemma4_mtp),
        use_step3p5_mtp=MagicMock(return_value=use_step3p5_mtp),
        use_multi_module_mtp=MagicMock(return_value=False),
        draft_model_config=SimpleNamespace(architectures=architectures or []),
    )
    return SimpleNamespace(speculative_config=speculative_config)


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


def test_init_speculator_dspark(monkeypatch):
    """Test dispatching DSpark speculative decoding."""
    config = _make_config(
        "dspark",
        use_dspark=True,
    )
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
        "AscendDSparkSpeculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_init_speculator_dflash2(monkeypatch):
    """Test dispatching DFlash2DraftModel to AscendDFlash2Speculator."""
    config = _make_config(
        "dflash",
        use_dflash=True,
        architectures=["DFlash2DraftModel"],
    )
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator",
        "AscendDFlash2Speculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_init_speculator_dflash(monkeypatch):
    """Test dispatching the regular DFlash draft model."""
    config = _make_config(
        "dflash",
        use_dflash=True,
        architectures=["DFlashDraftModel"],
    )
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator",
        "AscendDFlashSpeculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_init_speculator_mtp(monkeypatch):
    """Test dispatching the regular MTP speculative decoding path."""
    config = _make_config("mtp")
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.mtp.speculator",
        "AscendMTPSpeculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


@pytest.mark.parametrize("enforce_eager", [True, False, None])
def test_draft_compilation_config_respects_enforce_eager(monkeypatch, enforce_eager):
    config = _make_config("mtp")
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


def test_init_speculator_eagle(monkeypatch):
    """Test dispatching Eagle speculative decoding."""
    config = _make_config(
        "eagle",
        use_eagle=True,
    )
    device = torch.device("cpu")
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.eagle.speculator",
        "AscendEagleSpeculator",
    )

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_init_speculator_requires_speculative_config():
    """Test that init_speculator requires speculative_config to be present."""
    config = SimpleNamespace(speculative_config=None)

    with pytest.raises(AssertionError):
        init_speculator(config, torch.device("cpu"))


def test_init_speculator_unsupported_method():
    """Test that an unsupported speculative method raises NotImplementedError."""
    config = _make_config("unsupported")

    with pytest.raises(
        NotImplementedError,
        match="unsupported is not supported yet",
    ):
        init_speculator(config, torch.device("cpu"))


def test_init_speculator_routes_gemma4_mtp(monkeypatch):
    """Gemma4 MTP routes to AscendGemma4Speculator, not the regular MTP path."""
    config = _make_config("mtp", use_gemma4_mtp=True)
    speculator_cls = _mock_speculator_module(
        monkeypatch,
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator",
        "AscendGemma4Speculator",
    )
    device = torch.device("cpu")

    result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_init_speculator_skips_special_mtp():
    """Step3.5 MTP bypasses the regular MTP speculator and stays unsupported."""
    config = _make_config("mtp", use_step3p5_mtp=True)

    with pytest.raises(
        NotImplementedError,
        match="mtp is not supported yet",
    ):
        init_speculator(config, torch.device("cpu"))
