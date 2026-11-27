# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig

from vllm_ascend.worker.v2.spec_decode.dmtp.batch import NativeMTPBatch
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


@pytest.mark.parametrize("max_tokens", [1, 121, 127, 128, 129, 3071, 3072])
def test_paged_native_reserves_room_for_rejected_queries_at_context_limit(max_tokens):
    cfg = _config()
    native = NativeMTP(_NativeMTPWeights(cfg), cfg, torch.nn.Embedding(29, cfg.hidden_size), 2, max_tokens, paged=True)
    capacity = native.blocks_per_req * native.kv_pool.shape[2]
    assert capacity >= max_tokens + 7
    # Two live slots and an isolated padding slot must have equal capacity.
    assert native.kv_pool.shape[1] == 3 * native.blocks_per_req


@pytest.mark.parametrize("rejected", range(8))
@torch.inference_mode()
def test_paged_native_matches_hf_with_rejection_reordering_and_slot_reuse(monkeypatch, rejected):
    """Compare the batched path against independent, unpadded HF causal calls."""
    from vllm_ascend.device.device_op import DeviceOperator

    def write_cache(key, value, key_cache, value_cache, slot_mapping, **kwargs):
        for row, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                key_cache.flatten(0, 1)[slot].copy_(key[row])
                value_cache.flatten(0, 1)[slot].copy_(value[row])

    def attention(query, keys, values, cache_seqlens, page_table, cu_seqlens_q, softmax_scale, **kwargs):
        outputs = []
        for i, length in enumerate(cache_seqlens.tolist()):
            start, end = cu_seqlens_q[i : i + 2].tolist()
            k = keys[page_table[i].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1)
            v = values[page_table[i].long()].flatten(0, 1)[:length].repeat_interleave(2, dim=1)
            q = query[start:end]
            scores = torch.einsum("qhd,khd->hqk", q, k) * softmax_scale
            mask = torch.ones(end - start, length, dtype=torch.bool).triu(length - (end - start) + 1)
            scores.masked_fill_(mask, float("-inf"))
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        return torch.cat(outputs)

    monkeypatch.setattr(DeviceOperator, "reshape_and_cache", write_cache)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn_npu_3",
        SimpleNamespace(flash_attn_with_kvcache=attention, get_scheduler_metadata=lambda **_: torch.zeros(4)),
    )
    torch.manual_seed(41)
    cfg = _config()
    weights = _NativeMTPWeights(cfg)
    embedding = torch.nn.Embedding(29, cfg.hidden_size)
    native = NativeMTP(weights, cfg, embedding, 3, 132, paged=True)
    native.kv_pool.zero_()
    reference = NativeMTP(weights, cfg, embedding, 3, 132)
    staged = NativeMTPBatch(native, 3, 24)
    caches = {}
    # Partial prefill, reordered verification, then slot reuse with graph
    # padding. Slot 2 stays untouched, even when padded outputs are computed.
    for slots, lengths, starts, rejections, padded in [
        ([1, 0], [3, 2], [0, 0], [0, 0], 2),
        ([0, 1], [8, 8], [2, 3], [rejected, 7 - rejected], 3),
        ([1], [8], [0], [0], 2),
    ]:
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        t = sum(lengths)
        ids = torch.randint(0, 29, (t,))
        hidden = torch.randn(t, cfg.hidden_size)
        pos = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths)])
        batch = SimpleNamespace(
            num_reqs=len(slots),
            num_tokens=t,
            query_start_loc=offsets,
            idx_mapping=torch.tensor(slots),
            input_ids=ids,
            positions=pos,
        )
        sampled = torch.tensor([i % 2 for i in range(len(slots))])
        last = torch.tensor([[15], [16], [17]])
        next_prompt = torch.tensor([[20, 21, 22]])
        staged.prepare(batch, hidden, sampled, torch.tensor(rejections), last, next_prompt, padded)
        total = t + (padded - len(slots)) * 8
        metadata = staged.attention_metadata(padded, total, max(lengths))
        actual = native.forward_paged(
            staged.input_ids[:total], staged.hidden[:total], staged.positions[:total], metadata
        )
        for row, slot in enumerate(slots):
            start = offsets[row].item()
            end = offsets[row + 1].item() - rejections[row]
            anchor = last[slot, 0] if sampled[row] else next_prompt[0, slot]
            if starts[row] == 0:
                caches[slot] = reference.empty_cache(slot)
            committed = torch.cat([ids[start + 1 : end], anchor[None]])
            caches[slot], expected = reference.append(caches[slot], committed, hidden[start:end], pos[start:end] + 1)
            torch.testing.assert_close(actual[start:end], expected, rtol=2e-5, atol=2e-5)
            assert staged.last_indices[row] == end - 1
            assert (staged.slot_mapping[end : offsets[row + 1]] == -1).all()
        assert (staged.slot_mapping[t:total] == -1).all()
        assert native.kv_pool[:, 2 * native.blocks_per_req : 3 * native.blocks_per_req].count_nonzero() == 0
