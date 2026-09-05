"""Unit tests for global-layer addressing of layerwise pool keys under PP.

Covers the PP adaptation of the layerwise KV pool key space:

- ``PoolKey.split_layers(num_layers, layer_offset)`` emits global layer ids
  so pipeline stages write disjoint, model-global layer keys.
- A PP=2 producer layout (stage 0: layers 0..20, stage 1: layers 21..43
  including the MTP tail) produces exactly the union of a PP=1 decode
  layout's keys (layers 0..43), enabling cross-stage lookup.
"""

from typing import Any

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    PoolKey,
)


def _meta() -> KeyMetadata:
    return KeyMetadata(
        model_name="DSV4",
        head_or_tp_rank=0,
        pcp_rank=0,
        dcp_rank=0,
        pp_rank=0,
        kv_cache_group_id=0,
    )


def _key() -> PoolKey:
    return PoolKey(_meta(), "hash_x")


def _layer_ids(pool_key: PoolKey, num_layers: int, offset: int = 0) -> list[int]:
    return [k.layer_id for k in pool_key.split_layers(num_layers, offset)]


def _layer_key_strings(pool_key: PoolKey, num_layers: int, offset: int = 0) -> list[str]:
    return [k.to_string() for k in pool_key.split_layers(num_layers, offset)]


class TestSplitLayersOffset:
    def test_default_offset_is_backward_compatible(self):
        assert _layer_ids(_key(), 44) == list(range(44))

    def test_offset_shifts_layer_ids(self):
        # stage 1 of PP=2 over a 43-layer model: 22 local layers, offset 21
        assert _layer_ids(_key(), 22, 21) == list(range(21, 43))

    def test_producer_stages_cover_decode_keyspace_exactly(self):
        # PP=2 producers: stage0 (21 layers, offset 0) + stage1 (22 layers,
        # offset 21) + MTP (1 layer at global id 43, folded into stage1's
        # effective num_layers=23) must equal the PP=1 decode key space 0..43.
        stage0 = _layer_key_strings(_key(), 21, 0)
        stage1 = _layer_key_strings(_key(), 23, 21)  # 22 attn + MTP@43
        decode = _layer_key_strings(_key(), 44, 0)

        produced = set(stage0) | set(stage1)
        assert produced == set(decode)
        assert len(stage0) + len(stage1) == len(decode) == 44

    def test_no_overlap_between_stages(self):
        stage0 = set(_layer_key_strings(_key(), 21, 0))
        stage1 = set(_layer_key_strings(_key(), 22, 21))
        assert not (stage0 & stage1)

    def test_key_string_contains_global_layer_id(self):
        keys = _key().split_layers(1, 43)
        assert "@layer_id:43@" in keys[0].to_string()

    def test_replace_layer_id(self):
        k = _key().split_layers(5, 10)[3]
        assert k.layer_id == 13
        k2 = k.replace_layer_id(7)
        assert k2.layer_id == 7
        assert k2.chunk_hash == k.chunk_hash
        assert k2.key_metadata == k.key_metadata

    def test_hash_distinguishes_layers_and_offsets(self):
        base = _key()
        # same local index, different offset -> different global key
        k_a = base.split_layers(22, 0)[5]
        k_b = base.split_layers(22, 21)[5]
        assert hash(k_a) != hash(k_b)
        assert k_a != k_b

    def test_equality_across_constructions(self):
        # a key written by stage1 (local 0 + offset 21) equals a decode-side
        # key for global layer 21
        stage1_key = _key().split_layers(22, 21)[0]
        decode_key = _key().split_layers(44, 0)[21]
        assert stage1_key == decode_key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
