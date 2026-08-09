# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.quantization.fp8_config import _uses_dsv4_310p_adapter


def test_dsv4_adapter_requires_deepseek_v4_model() -> None:
    qwen_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3")))
    dsv4_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4"))
    )

    with (
        patch("vllm_ascend.quantization.fp8_config.is_310p", return_value=True),
        patch("vllm_ascend.quantization.fp8_config.get_current_vllm_config", return_value=qwen_config),
    ):
        assert not _uses_dsv4_310p_adapter()

    with (
        patch("vllm_ascend.quantization.fp8_config.is_310p", return_value=True),
        patch("vllm_ascend.quantization.fp8_config.get_current_vllm_config", return_value=dsv4_config),
    ):
        assert _uses_dsv4_310p_adapter()
