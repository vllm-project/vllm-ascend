# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import init_speculator


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
        use_dspark=MagicMock(return_value=use_dspark),
        use_dflash=MagicMock(return_value=use_dflash),
        use_eagle=MagicMock(return_value=use_eagle),
        use_gemma4_mtp=MagicMock(return_value=use_gemma4_mtp),
        use_step3p5_mtp=MagicMock(return_value=use_step3p5_mtp),
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


@pytest.mark.parametrize(
    "use_gemma4_mtp,use_step3p5_mtp",
    [
        (True, False),
        (False, True),
    ],
)
def test_init_speculator_skips_special_mtp(
    use_gemma4_mtp,
    use_step3p5_mtp,
):
    """Test that Gemma4 and Step3.5 MTP bypass the regular MTP speculator."""
    config = _make_config(
        "mtp",
        use_gemma4_mtp=use_gemma4_mtp,
        use_step3p5_mtp=use_step3p5_mtp,
    )

    with pytest.raises(
        NotImplementedError,
        match="mtp is not supported yet",
    ):
        init_speculator(config, torch.device("cpu"))
