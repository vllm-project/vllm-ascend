import pytest
from transformers import Qwen2Config

from vllm_ascend.ops.qwen2_decoder import AscendQwen2DecoderLayer


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_qwen2_decoder_layer_uses_configured_attention_type(layer_idx: int):
    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    decoder_layer = AscendQwen2DecoderLayer(config, layer_idx)

    assert decoder_layer.attention_type == config.layer_types[layer_idx]
