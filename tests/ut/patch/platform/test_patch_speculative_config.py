# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override


def _make_cfg(model_type: str, *, n_predict: int = 1, architectures: list[str] | None = None):
    cfg = SimpleNamespace(
        model_type=model_type,
        architectures=list(architectures or ["DummyForCausalLM"]),
        num_nextn_predict_layers=n_predict,
    )

    def update(mapping=None, **kwargs):
        data = dict(mapping or {})
        data.update(kwargs)
        for key, value in data.items():
            setattr(cfg, key, value)

    cfg.update = update
    return cfg


@pytest.mark.parametrize(
    ("model_type", "expected_arch"),
    [
        ("deepseek_v3", "DeepSeekMTPModel"),
        ("deepseek_v32", "DeepSeekMTPModel"),
        ("glm_moe_dsa", "DeepSeekMTPModel"),
        ("deepseek_v4", "DeepSeekV4MTPModel"),
        ("deepseek_mtp", "DeepSeekMTPModel"),
    ],
)
def test_hf_config_override_deepseek_family(model_type: str, expected_arch: str):
    cfg = _make_cfg(model_type, n_predict=2)

    out = hf_config_override(cfg)

    assert out.model_type == "deepseek_mtp"
    assert out.architectures == [expected_arch]
    assert out.n_predict == 2
