# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ple_quantization import load_modelslim_ple_weights


def make_layer(start=1, end=5):
    weight = torch.nn.Parameter(torch.zeros(end - start, 8, dtype=torch.bfloat16), requires_grad=False)

    def copy(param, value, checkpoint_start):
        param[checkpoint_start - start : checkpoint_start - start + len(value)].copy_(value)

    weight.weight_loader = copy
    return SimpleNamespace(
        split_ngram_parts=2,
        ngram_embedding=SimpleNamespace(
            org_vocab_size=6,
            embedding_dim=8,
            shard_indices=SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end),
            weight=weight,
        ),
    )


@pytest.mark.parametrize("scale_first", [False, True])
def test_split_int8_ple_loads_tp_intersection_and_regular_weights(scale_first):
    layer = make_layer()
    codes = torch.arange(-24, 24, dtype=torch.int8).reshape(6, 8)
    scales = torch.arange(1, 13, dtype=torch.float32).reshape(6, 2) / 10
    tensors = []
    for shard in range(2):
        pair = [
            (f"ngram_embedding.shard_{shard}.weight", codes[shard * 3 : (shard + 1) * 3]),
            (f"ngram_embedding.shard_{shard}.scale", scales[shard * 3 : (shard + 1) * 3]),
        ]
        tensors.extend(pair[::-1] if scale_first else pair)
    regular = ("layer_multipliers", torch.tensor([7]))
    tensors.append(regular)

    def original(_layer, weights):
        assert list(weights) == [regular]
        return {regular[0]}

    loaded = load_modelslim_ple_weights(layer, iter(tensors), original)
    expected = (codes.float().reshape(6, 2, 4) * scales[..., None]).reshape(6, 8).to(torch.bfloat16)
    torch.testing.assert_close(layer.ngram_embedding.weight, expected[1:5], rtol=0, atol=0)
    assert loaded == {"ngram_embedding.weight", "layer_multipliers"}


def test_bf16_shard_is_delegated_unchanged():
    tensors = [("ngram_embedding.shard_0.weight", torch.ones(3, 8, dtype=torch.bfloat16))]

    def original(_layer, weights):
        assert list(weights) == tensors
        return {"ngram_embedding.weight"}

    assert load_modelslim_ple_weights(make_layer(), tensors, original) == {"ngram_embedding.weight"}


@pytest.mark.parametrize("failure", ["missing", "shape", "negative", "nan", "index"])
def test_invalid_int8_ple_checkpoint_is_rejected(failure):
    codes = torch.ones(3, 8, dtype=torch.int8)
    scales = torch.ones(3, 2)
    if failure == "shape":
        scales = scales[:, :0]
    elif failure == "negative":
        scales[0, 0] = -1
    elif failure == "nan":
        scales[0, 0] = float("nan")
    index = 2 if failure == "index" else 0
    tensors = [(f"ngram_embedding.shard_{index}.weight", codes)]
    if failure != "missing":
        tensors.append((f"ngram_embedding.shard_{index}.scale", scales))

    def original(_layer, weights):
        list(weights)
        return set()

    with pytest.raises(ValueError, match="PLE"):
        load_modelslim_ple_weights(make_layer(), tensors, original)
