# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig

from vllm_ascend.worker.v2.spec_decode.dmtp.native_mtp import NativeMTP, _NativeMTPWeights


def _config():
    return Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )


@torch.inference_mode()
def test_incremental_mtp_matches_full_causal_prefixes_after_slot_reuse():
    torch.manual_seed(23)
    cfg = _config()
    weights = _NativeMTPWeights(cfg)
    embedding = torch.nn.Embedding(29, cfg.hidden_size)
    mtp = NativeMTP(weights, cfg, embedding, 2, 132)
    token_ids = torch.arange(130) % 29
    previous = torch.randn(130, cfg.hidden_size)

    # Compute prefix+bonus pairs with a single teacher-forced HF call, with
    # all prefix rows visible up to and including the current shifted token.
    positions = torch.arange(1, 131).view(1, 1, -1)
    position_embeddings = mtp.rope(previous, positions)
    expected = weights(
        embedding(token_ids), previous, torch.full((130, 130), float("-inf")).triu(1), position_embeddings, None
    )
    cache, prefill = mtp.append(mtp.empty_cache(0), token_ids[:129], previous[:129])
    cache, decode = mtp.append(cache, token_ids[-1:], previous[-1:])
    assert cache.length == 130
    torch.testing.assert_close(torch.cat([prefill, decode]), expected)

    # A new request on that slot and another request on an independent slot
    # must both ignore the old length and all stale values beyond their prefix.
    new_inputs = token_ids.flip(0)[:7]
    new_hidden = previous.flip(0)[:7]
    new_cache, reused = mtp.append(mtp.empty_cache(0), new_inputs, new_hidden)
    separate, expected_new = mtp.append(mtp.empty_cache(1), new_inputs, new_hidden)
    assert new_cache.keys.data_ptr() == cache.keys.data_ptr() != separate.keys.data_ptr()
    torch.testing.assert_close(reused, expected_new)


def test_from_target_loads_only_native_weights_and_validates_missing_keys(tmp_path):
    cfg = _config()
    weights = _NativeMTPWeights(cfg).bfloat16()
    tensors = {f"mtp.{name}": tensor for name, tensor in weights.state_dict().items()}
    shards = ["mtp1.safetensors", "mtp2.safetensors"]
    weight_map = {}
    for i, shard in enumerate(shards):
        keys = list(tensors)[i::2]
        save_file({key: tensors[key] for key in keys}, tmp_path / shard)
        weight_map.update(dict.fromkeys(keys, shard))
    weight_map["model.layers.0.mlp.up_proj.weight"] = "target-shard-not-read.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    target_embedding = torch.nn.Embedding(29, cfg.hidden_size).bfloat16()
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embedding), embed_input_ids=target_embedding)
    target_config = SimpleNamespace(
        hf_text_config=cfg, model=tmp_path, quantization=None, dtype=torch.bfloat16, revision=None, max_model_len=128
    )
    vllm_config = SimpleNamespace(
        model_config=target_config,
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        load_config=SimpleNamespace(download_dir=None, load_format="auto"),
    )
    native = NativeMTP.from_target(vllm_config, target)
    assert native.embed is target_embedding
    assert native.cache_bytes_per_token == 2 * cfg.num_key_value_heads * cfg.head_dim * 2
    torch.testing.assert_close(native.weights.state_dict(), weights.state_dict(), rtol=0, atol=0)
    del weight_map["mtp.norm.weight"]
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="missing=.*norm.weight"):
        NativeMTP.from_target(vllm_config, target)
