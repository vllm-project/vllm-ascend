"""CPU checks for Qwen4Exp ModelSlim PLE INT8 table loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.models.qwen4_exp.common import ple as ple_utils
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

from vllm_ascend.models.qwen4_exp.ple import patch_upstream_ple_short_conv
from vllm_ascend.models.qwen4_exp.ple_int8 import (
    AscendQwen4ExpPLEInt8EmbeddingMethod,
    load_ple_int8_scales,
)
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


class FakePLEEmbedding(ple_utils.PLEVocabParallelEmbedding):
    supports_prefetch = False

    def __init__(self):
        nn.Module.__init__(self)
        self.org_vocab_size = 10
        self.embedding_dim = 160
        self.shard_indices = SimpleNamespace(org_vocab_start_index=2, org_vocab_end_index=7)

    def allocate_embedding_weight(self, rows, width, dtype):
        return torch.empty(rows, width, dtype=dtype)


def test_select_and_dequantize_int8_ple_rows():
    patch_upstream_ple_short_conv()
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    config = AscendModelSlimConfig({prefix + ".shard_0.quant_type": "INT8_SYM"})
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, prefix)
    assert isinstance(method, AscendQwen4ExpPLEInt8EmbeddingMethod)
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(AscendModelSlimConfig({}), prefix),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )
    layer = FakePLEEmbedding()
    method.create_weights(layer, 160, [5], 160, 5, torch.bfloat16, weight_loader=lambda *_: None)
    layer.weight.data.fill_(10)
    layer.weight_scale.data.fill_(0.01)
    layer.weight_scale.data[:, 1] = 0.02
    actual = method.embedding(layer, torch.tensor([0, 3]))
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual[:, :32].float(), torch.full((2, 32), 0.1), atol=0.001, rtol=0)
    torch.testing.assert_close(actual[:, 32:64].float(), torch.full((2, 32), 0.2), atol=0.001, rtol=0)
    layer.supports_prefetch = True
    with pytest.raises(NotImplementedError, match="pinned-host"):
        method.create_weights(layer, 160, [5], 160, 5, torch.bfloat16)


def test_scale_shards_follow_weight_row_overlap():
    method = AscendQwen4ExpPLEInt8EmbeddingMethod()
    embedding = FakePLEEmbedding()
    embedding.embedding_method = method
    method.create_weights(embedding, 160, [5], 160, 5, torch.bfloat16, weight_loader=lambda *_: None)
    embedding.weight_scale.data.zero_()
    owner = SimpleNamespace(ngram_embedding=embedding, split_ngram_parts=2)
    weights = [
        ("ngram_embedding.shard_0.scale", torch.arange(25, dtype=torch.float32).reshape(5, 5)),
        ("ngram_embedding.shard_1.scale", torch.arange(25, 50, dtype=torch.float32).reshape(5, 5)),
    ]
    loaded = load_ple_int8_scales(owner, weights, lambda _, remaining: set(remaining))
    assert loaded == {"ngram_embedding.weight_scale"}
    method.process_weights_after_loading(embedding)
    torch.testing.assert_close(embedding.weight_scale[:3], weights[0][1][2:5])
    torch.testing.assert_close(embedding.weight_scale[3:], weights[1][1][:2])
    with pytest.raises(ValueError, match="Duplicate PLE scale shard"):
        load_ple_int8_scales(owner, weights + weights[:1], lambda _, remaining: set(remaining))
    with pytest.raises(ValueError, match="expected FP32"):
        load_ple_int8_scales(
            owner,
            [("ngram_embedding.shard_0.scale", weights[0][1].to(torch.bfloat16))],
            lambda _, remaining: set(remaining),
        )


def test_scale_shards_across_weight_file_batches():
    method = AscendQwen4ExpPLEInt8EmbeddingMethod()
    embedding = FakePLEEmbedding()
    embedding.embedding_method = method
    method.create_weights(embedding, 160, [5], 160, 5, torch.bfloat16, weight_loader=lambda *_: None)
    owner = SimpleNamespace(ngram_embedding=embedding, split_ngram_parts=2)
    shard_0 = ("ngram_embedding.shard_0.scale", torch.arange(25, dtype=torch.float32).reshape(5, 5))
    shard_1 = ("ngram_embedding.shard_1.scale", torch.arange(25, 50, dtype=torch.float32).reshape(5, 5))

    assert load_ple_int8_scales(owner, [shard_0], lambda _, remaining: set(remaining)) == {
        "ngram_embedding.weight_scale"
    }
    with pytest.raises(ValueError, match="Missing PLE scale shards after loading: \\[1\\]"):
        method.process_weights_after_loading(embedding)

    assert load_ple_int8_scales(owner, [shard_1], lambda _, remaining: set(remaining)) == {
        "ngram_embedding.weight_scale"
    }
    method.process_weights_after_loading(embedding)
    torch.testing.assert_close(embedding.weight_scale[:3], shard_0[1][2:5])
    torch.testing.assert_close(embedding.weight_scale[3:], shard_1[1][:2])

    with pytest.raises(ValueError, match="Duplicate PLE scale shard across batches: 0"):
        load_ple_int8_scales(owner, [shard_0], lambda _, remaining: set(remaining))
