# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend.models.deepseek_v4.mtp import DeepSeekMultiTokenPredictorLayer
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod


class _MockTPGroup:
    """Minimal TP=1 group stand-in for get_tp_group() on CPU runners."""

    rank_in_group = 0
    world_size = 1


@pytest.fixture(autouse=True)
def _mock_tp_group():
    """ReplicatedLinear/ParallelLMHead query the TP group during __init__,
    but parallel state is not initialized on CPU test runners."""
    mock = _MockTPGroup()
    with (
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=mock),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=mock),
    ):
        yield


def _make_mtp_layer(prefix: str) -> DeepSeekMultiTokenPredictorLayer:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=128,
                    rms_norm_eps=1e-6,
                    hc_eps=1e-6,
                    hc_mult=2,
                    vocab_size=100,
                )
            )
        ),
        use_v2_model_runner=False,
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    return DeepSeekMultiTokenPredictorLayer(vllm_config, prefix=prefix)


def test_mtp_projections_propagate_prefix() -> None:
    with patch("vllm_ascend.models.deepseek_v4.mtp.DeepseekV2DecoderLayer") as mock_decoder_layer:
        mock_decoder_layer.return_value = MagicMock()
        layer = _make_mtp_layer("model.mtp.0")

    assert layer.e_proj.prefix == "model.mtp.0.e_proj"
    assert layer.h_proj.prefix == "model.mtp.0.h_proj"


def test_mtp_projection_prefixes_match_compressed_tensors_ignore_rules() -> None:
    quant_config = AscendCompressedTensorsConfig.from_config(
        {
            **COMPRESSED_TENSORS_W8A8_CONFIG,
            "ignore": ["re:^model\\.mtp\\..*"],
        }
    )
    layer_prefix = "model.mtp.0.e_proj"

    prefixed_method = quant_config.get_quant_method(MagicMock(spec=LinearBase), layer_prefix)
    empty_prefix_method = quant_config.get_quant_method(MagicMock(spec=LinearBase), "")

    assert isinstance(prefixed_method, AscendUnquantizedLinearMethod)
    assert isinstance(empty_prefix_method, AscendLinearMethod)
    assert not isinstance(empty_prefix_method, UnquantizedLinearMethod)


def test_mtp_projection_prefixes_are_not_empty() -> None:
    with patch("vllm_ascend.models.deepseek_v4.mtp.DeepseekV2DecoderLayer") as mock_decoder_layer:
        mock_decoder_layer.return_value = MagicMock()
        layer = _make_mtp_layer("model.mtp.1")

    for projection in (layer.e_proj, layer.h_proj):
        assert projection.prefix
        assert projection.prefix.startswith("model.mtp.1.")
