# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import unittest
from unittest.mock import patch

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager, MambaManager


class TestCachedMambaTail(unittest.TestCase):
    def setUp(self):
        init_none_hash(sha256)

    def manager(self, block_size=1280, hash_block_size=1280):
        attention = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32)
        mamba = MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float32,),
            page_size_padded=attention.page_size_bytes,
            mamba_cache_mode="align",
        )
        config = KVCacheConfig(
            num_blocks=256,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention),
                KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba),
            ],
        )
        return KVCacheManager(
            config,
            max_model_len=32768,
            enable_caching=True,
            hash_block_size=hash_block_size,
            scheduler_block_size=block_size,
            use_eagle=False,
        )

    def request(self, request_id, length, hash_block_size=1280):
        return Request(
            request_id=request_id,
            prompt_token_ids=list(range(length)),
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_block_size, sha256),
        )

    def test_warm_siblings_are_admitted_in_one_step(self):
        for length in (12799, 12800, 16383, 16384):
            with self.subTest(length=length):
                manager = self.manager()
                warm = self.request("warm", length)
                boundary = (length - 1) // 1280 * 1280
                self.assertIsNotNone(manager.allocate_slots(warm, boundary))
                warm.num_computed_tokens = boundary
                manager.new_step_starts()
                self.assertIsNotNone(manager.allocate_slots(warm, length - boundary))
                warm.num_computed_tokens = length
                manager.new_step_starts()
                manager.free(warm)
                for index in range(4):
                    request = self.request(f"sibling-{index}", length)
                    blocks, hit, _ = manager.get_computed_blocks(request)
                    self.assertEqual(hit, boundary)
                    self.assertIsNotNone(
                        manager.allocate_slots(
                            request,
                            length - hit,
                            num_new_computed_tokens=hit,
                            new_computed_blocks=blocks,
                        )
                    )

    def test_pending_state_is_not_read_before_next_step(self):
        manager = self.manager()
        producer = self.request("producer", 2561)
        self.assertIsNotNone(manager.allocate_slots(producer, 2560))
        sibling = self.request("sibling", 2561)
        blocks, hit, _ = manager.get_computed_blocks(sibling)
        self.assertEqual(hit, 2560)
        self.assertIsNone(manager.allocate_slots(sibling, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks))
        manager.new_step_starts()
        blocks, hit, _ = manager.get_computed_blocks(sibling)
        self.assertIsNotNone(
            manager.allocate_slots(sibling, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        )

    def test_partial_hit_still_gets_private_copy(self):
        manager = self.manager(block_size=8, hash_block_size=4)
        producer = self.request("producer", 13, hash_block_size=4)
        self.assertIsNotNone(manager.allocate_slots(producer, 12))
        producer.num_computed_tokens = 12
        manager.new_step_starts()
        sibling = self.request("sibling", 13, hash_block_size=4)
        blocks, hit, _ = manager.get_computed_blocks(sibling)
        self.assertEqual(hit, 12)
        source = blocks.blocks[1][-1]
        source_hash = source.block_hash
        self.assertIsNotNone(
            manager.allocate_slots(sibling, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks)
        )
        state_manager = manager.coordinator.single_type_managers[1]
        self.assertIsNot(state_manager.req_to_blocks[sibling.request_id][1], source)
        self.assertEqual(source.block_hash, source_hash)
        self.assertTrue(any(src is source for src, _ in state_manager._pending_cow_copies))

    def test_reserved_checkpoint_is_delegated(self):
        manager = self.manager()
        request = self.request("checkpoint", 2561)
        state_manager = manager.coordinator.single_type_managers[1]
        state_manager._checkpoints[request.request_id] = (2560, 1)
        with patch.object(MambaManager, "_cache_partial_tail_block", return_value=None) as original:
            state_manager._cache_partial_tail_block(request, 2560, 0)
        original.assert_called_once_with(request, 2560, 0)

    def test_override_keeps_upstream_signature(self):
        self.assertEqual(
            inspect.signature(AscendMambaManager._cache_partial_tail_block),
            inspect.signature(MambaManager._cache_partial_tail_block),
        )
