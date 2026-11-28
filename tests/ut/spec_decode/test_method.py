# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import SpeculativeConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.worker.gpu.spec_decode import speculator_config as upstream_config
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.worker.v2 import spec_decode


@pytest.mark.parametrize(("upstream", "ascend"), spec_decode._NPU_SPECULATORS.items())
def test_npu_adapters_inherit_the_selected_upstream_implementation(upstream, ascend):
    assert issubclass(resolve_obj_by_qualname(ascend), resolve_obj_by_qualname(upstream))


@pytest.mark.parametrize(
    ("method", "config_class", "module", "name"),
    [
        ("dspark", "dspark.config.DSparkSpeculatorConfig", "dspark", "DSpark"),
        ("dflash", "dflash.config.DFlashSpeculatorConfig", "dflash", "DFlash"),
        ("dflash", "dflash2.config.DFlash2SpeculatorConfig", "dflash2", "DFlash2"),
        ("dflash", "lilicorr.config.LiLiCorrSpeculatorConfig", "lilicorr", "LiLiCorr"),
        ("mtp", "gemma4.config.Gemma4MTPSpeculatorConfig", "gemma4", "Gemma4"),
        ("eagle", "eagle.config.EagleSpeculatorConfig", "eagle", "Eagle"),
        ("eagle3", "eagle.config.Eagle3SpeculatorConfig", "eagle", "Eagle"),
        ("draft_model", "standalone_ar.config.StandaloneARSpeculatorConfig", "standalone_ar", "StandaloneAR"),
    ],
)
def test_native_config_selects_npu_adapter(monkeypatch, method, config_class, module, name):
    spec = object.__new__(SpeculativeConfig)
    spec.method = method
    spec.custom_method = None
    spec.enforce_eager = False
    spec.draft_model_config = SimpleNamespace(speculator_config_class=f"vllm.v1.worker.gpu.spec_decode.{config_class}")
    config = SimpleNamespace(speculative_config=spec)
    constructor = MagicMock()
    monkeypatch.setattr(f"vllm_ascend.worker.v2.spec_decode.{module}.speculator.Ascend{name}Speculator", constructor)

    assert spec_decode.init_speculator(config, torch.device("cpu")) is constructor.return_value
    constructor.assert_called_once_with(config, torch.device("cpu"))


@pytest.mark.parametrize("takes_states", [False, True])
@pytest.mark.parametrize("use_310p", [False, True])
@pytest.mark.parametrize("class_name", ["Speculator", "MTPSpeculator"])
def test_custom_speculator_is_constructed_by_upstream(monkeypatch, takes_states, use_310p, class_name):
    calls = []

    class CustomSpeculator(BaseSpeculator):
        def __new__(cls, config, device):
            calls.append((config, device))
            return calls

    class CustomStatefulSpeculator(BaseSpeculator):
        def __new__(cls, config, device, *, req_states):
            calls.append((config, device, req_states))
            return calls

    # A custom class can derive its config from MTP without being the built-in
    # MTP speculator. The exact qualified class, not base_method, is authoritative.
    class CustomConfig(MTPSpeculatorConfig):
        speculator_class = f"custom_test.{class_name}"

    monkeypatch.setattr(upstream_config, "_CUSTOM_CONFIGS", dict(upstream_config._CUSTOM_CONFIGS))
    upstream_config.register_speculative_method("ascend_factory_test", CustomConfig)
    module = ModuleType("custom_test")
    setattr(module, class_name, CustomStatefulSpeculator if takes_states else CustomSpeculator)
    monkeypatch.setitem(sys.modules, "custom_test", module)
    spec = object.__new__(SpeculativeConfig)
    spec.method = "mtp"
    spec.custom_method = "ascend_factory_test"
    spec.enforce_eager = False
    config = SimpleNamespace(speculative_config=spec)
    profile = SimpleNamespace(supports=lambda capability: not use_310p)
    monkeypatch.setattr(spec_decode, "get_current_hardware_profile", lambda: profile)
    states = object()
    device = torch.device("cpu")

    assert spec_decode.init_speculator(config, device, states) is calls
    assert calls == [(config, device, states) if takes_states else (config, device)]
    assert isinstance(config.speculative_config.speculator_config, CustomConfig)
    assert config.speculative_config.speculator_config.speculator_class == f"custom_test.{class_name}"


def test_custom_speculator_retains_upstream_type_validation(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            enforce_eager=False,
            speculator_config=SimpleNamespace(speculator_class="custom_test.NotASpeculator"),
        )
    )
    monkeypatch.setattr("vllm.v1.worker.gpu.spec_decode.resolve_obj_by_qualname", lambda qualname: object)

    with pytest.raises(TypeError, match="must be a subclass"):
        spec_decode.init_speculator(config, torch.device("cpu"))


def test_unported_builtin_does_not_silently_run_gpu_implementation():
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            enforce_eager=False,
            speculator_config=SimpleNamespace(
                speculator_class="vllm.v1.worker.gpu.spec_decode.suffix.speculator.SuffixSpeculator"
            ),
        )
    )

    with pytest.raises(NotImplementedError, match="No Ascend adapter"):
        spec_decode.init_speculator(config, torch.device("cpu"))
