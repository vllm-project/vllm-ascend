# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend import utils
from vllm_ascend.compilation import graph_fusion_pass_manager as manager_module
from vllm_ascend.compilation.passes import norm_quant_fusion_pass
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize(
    ("hf_config", "device_type", "expected_passes"),
    [
        (SimpleNamespace(model_type="mistral4"), AscendDeviceType.A2, 0),
        (
            SimpleNamespace(
                model_type="mistral3",
                text_config=SimpleNamespace(model_type="mistral4"),
            ),
            AscendDeviceType.A2,
            0,
        ),
        (SimpleNamespace(model_type="ministral3"), AscendDeviceType.A2, 0),
        (
            SimpleNamespace(
                model_type="mistral3",
                text_config=SimpleNamespace(model_type="ministral3"),
            ),
            AscendDeviceType.A2,
            0,
        ),
        (SimpleNamespace(model_type="mistral4"), AscendDeviceType.A3, 1),
        (SimpleNamespace(model_type="deepseek_v3"), AscendDeviceType.A2, 1),
    ],
)
def test_mistral4_a2_skips_unsupported_add_rms_norm_quant_patterns(
    monkeypatch,
    hf_config,
    device_type,
    expected_passes,
):
    config = SimpleNamespace(
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": True,
                "fuse_qknorm_rope": False,
                "fuse_muls_add": False,
            }
        },
        model_config=SimpleNamespace(hf_config=hf_config),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
    )
    fake_pass = object()
    pass_factory = MagicMock(return_value=fake_pass)
    monkeypatch.setattr(
        norm_quant_fusion_pass,
        "AddRMSNormQuantFusionPass",
        pass_factory,
    )
    monkeypatch.setattr(utils, "is_310p", lambda: False)
    monkeypatch.setattr(
        utils,
        "get_ascend_device_type",
        lambda: device_type,
    )

    manager = manager_module.GraphFusionPassManager()
    manager.configure(config)

    assert len(manager.passes) == expected_passes
    if expected_passes:
        pass_factory.assert_called_once_with(config)
    else:
        pass_factory.assert_not_called()
