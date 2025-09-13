import unittest
from unittest.mock import MagicMock, call, patch
from vllm.v1.core.block_pool import BlockPool
from vllm_ascend.distributed.cpu_offload_manager.cpu_kv_cache_manager import CPUCacheStats,CPUKVCacheManager
from collections import defaultdict
from typing import Optional
import torch

from vllm.utils import logger, sha256
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         PrefixCachingMetrics)
from vllm.v1.core.single_type_kv_cache_manager import \
    get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheSpec,AttentionSpec,FullAttentionSpec
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

original_init = CPUKVCacheManager.__init__

class TestCPUCacheStats(unittest.TestCase):
    def setUp(self):
        self.stats = CPUCacheStats(enable_prefix_caching=True, log_stats=True)
    
    @patch('time.time')
    def test_initialization_with_log_stats_true(self, mock_time):
        mock_time.return_value = 1620000000.0

        stats = self.stats

        self.assertTrue(stats.enable_prefix_caching)
        self.assertTrue(stats.log_stats)
        
        self.assertIsInstance(stats.prefix_cache_stats, PrefixCacheStats)
        self.assertIsInstance(stats.cpu_prefix_cache_metrics, PrefixCachingMetrics)

    @patch('time.time')
    @patch('vllm.utils.logger.info')
    def test_log(self, mock_logger, mock_time):
        self.mock_metrics = MagicMock(spec=PrefixCachingMetrics)
        self.mock_metrics.hit_rate = 0.75
        self.stats.cpu_prefix_cache_metrics = self.mock_metrics

        self.stats.time_sec = 1000  
        mock_time.return_value = 1010  
        self.stats.log()

        mock_logger.assert_called_once_with(
            "CPU Prefix cache hit rate: %.1f%%", 75.0
        )

    def test_make_prefix_cache_stats(self):
        self.stats_with_log = CPUCacheStats(enable_prefix_caching=True, log_stats=True)
        self.stats_without_log = CPUCacheStats(enable_prefix_caching=True, log_stats=False)
        
        self.original_stats = self.stats_with_log.prefix_cache_stats

        result = self.stats_with_log.make_prefix_cache_stats()
        
        self.assertIs(result, self.original_stats)
        self.assertIsInstance(result, PrefixCacheStats)
        self.assertIsNot(self.stats_with_log.prefix_cache_stats, self.original_stats)
        self.assertIsInstance(self.stats_with_log.prefix_cache_stats, PrefixCacheStats)

        result = self.stats_without_log.make_prefix_cache_stats()        
        self.assertIsNone(result)
        self.assertIsNone(self.stats_without_log.prefix_cache_stats)
    
    def test_update(self):
        self.enabled_both = CPUCacheStats(enable_prefix_caching=True, log_stats=True)
        self.disabled_log = CPUCacheStats(enable_prefix_caching=True, log_stats=False)
        self.disabled_caching = CPUCacheStats(enable_prefix_caching=False, log_stats=True)
        
        self.initial_requests = self.enabled_both.prefix_cache_stats.requests
        self.initial_queries = self.enabled_both.prefix_cache_stats.queries
        self.initial_hits = self.enabled_both.prefix_cache_stats.hits

        num_tokens = 100
        num_computed_tokens = 30
        
        self.enabled_both.update(num_tokens, num_computed_tokens)
        
        self.assertEqual(
            self.enabled_both.prefix_cache_stats.requests,
            self.initial_requests + 1
        )
        self.assertEqual(
            self.enabled_both.prefix_cache_stats.queries,
            self.initial_queries + num_tokens
        )
        self.assertEqual(
            self.enabled_both.prefix_cache_stats.hits,
            self.initial_hits + num_computed_tokens
        )

class TestCPUKVCacheManager(unittest.TestCase):
    
    def setUp(self):
        self.kv_cache_spec = FullAttentionSpec(
            block_size=64,
            num_kv_heads=8,      
            head_size=64,        
            dtype=torch.float32, 
            use_mla=False        
        )
        
        self.mock_block_pool = MagicMock()
        self.mock_block_pool.get_num_free_blocks.return_value = 10
        # self.mock_cpu_cache_stats = MagicMock(spec=CPUCacheStats)
        # self.mock_cpu_cache_stats.prefix_cache_stats = MagicMock()

        self.mock_cpu_cache_stats = MagicMock()
        self.mock_cpu_cache_stats.prefix_cache_stats = MagicMock()
        self.mock_cpu_cache_stats.cpu_prefix_cache_metrics = MagicMock()
        self.mock_cpu_cache_stats.cpu_prefix_cache_metrics.observe = MagicMock()
        self.mock_cpu_cache_stats.set_cache_stats = MagicMock()
        self.mock_cpu_cache_stats.log = MagicMock()

        self.mock_manager = MagicMock()

        self.manager_patcher = patch('vllm.v1.core.single_type_kv_cache_manager.get_manager_for_kv_cache_spec', return_value=self.mock_manager)
        self.mock_get_manager = self.manager_patcher.start()
        
        self.stats_patcher = patch('vllm_ascend.distributed.cpu_offload_manager.cpu_kv_cache_manager.CPUCacheStats', return_value=self.mock_cpu_cache_stats)
        self.mock_stats_class = self.stats_patcher.start()
        
        self.block_pool_patcher = patch('vllm.v1.core.block_pool.BlockPool', return_value=self.mock_block_pool)
        self.mock_block_pool_class = self.block_pool_patcher.start()

        def patched_init(instance, kv_cache_spec, num_cpu_blocks, caching_hash_algo="builtin",
                        use_eagle=False, enable_kv_cache_events=False):
            instance.block_size = kv_cache_spec.block_size
            instance.num_cpu_blocks = num_cpu_blocks
            instance.caching_hash_fn = hash if caching_hash_algo == "builtin" else None
            instance.use_eagle = use_eagle
            
            instance.block_pool = self.mock_block_pool
            instance.cpu_cache_stats = self.mock_cpu_cache_stats
            
            instance.single_type_manager = self.mock_manager
            
            instance.req_to_block_hashes = defaultdict(list)
            instance.req_to_computed_blocks = defaultdict(list)
            instance.req_failed_to_allocate = defaultdict(bool)
            instance.req_to_num_tokens = defaultdict(int)
            instance.req_to_free = defaultdict(lambda: Request())

        CPUKVCacheManager.__init__ = patched_init

        self.cache_manager = CPUKVCacheManager(
            kv_cache_spec=self.kv_cache_spec,  # 使用真实实例而不是Mock
            num_cpu_blocks=100,
            caching_hash_algo="sha256",
            use_eagle=True,
            enable_kv_cache_events=True
        )

    def create_mock_blocks(self, count, start_id=1):
        blocks = []
        for i in range(count):
            block = MagicMock()
            block.block_id = start_id + i  
            blocks.append(block)
        return blocks
    
    def tearDown(self):
        self.manager_patcher.stop()
        self.stats_patcher.stop()
        self.block_pool_patcher.stop()
        CPUKVCacheManager.__init__ = original_init

    
    def test_initialization(self):
       
        self.assertEqual(self.cache_manager.block_size, 64)
        self.assertEqual(self.cache_manager.num_cpu_blocks, 100)
        self.assertTrue(self.cache_manager.use_eagle)
        self.assertEqual(self.cache_manager.block_pool, self.mock_block_pool)
        self.assertEqual(self.cache_manager.single_type_manager, self.mock_manager)
        
        self.assertIsInstance(self.cache_manager.req_to_block_hashes, defaultdict)
        self.assertIsInstance(self.cache_manager.req_to_computed_blocks, defaultdict)
        self.assertIsInstance(self.cache_manager.req_failed_to_allocate, defaultdict)
        self.assertIsInstance(self.cache_manager.req_to_num_tokens, defaultdict)
        self.assertIsInstance(self.cache_manager.req_to_free, defaultdict)
        self.assertEqual(self.cache_manager.cpu_cache_stats, self.mock_cpu_cache_stats)

    def test_get_matched_num_and_touch(self):
        # case 1
        mock_request = MagicMock(spec=Request)
        mock_request.request_id = "test_request"
        mock_sampling_params = MagicMock()
        mock_sampling_params.prompt_logprobs = 1  
        mock_request.sampling_params = mock_sampling_params
        mock_request.num_tokens = 10
        mock_request.block_hashes = []  

        result = self.cache_manager.get_matched_num_and_touch(mock_request)
        
        self.assertEqual(result, (0, False))
        self.mock_manager.find_longest_cache_hit.assert_not_called()
        self.mock_block_pool.touch.assert_not_called()

        # case2 
        mock_sampling_params.prompt_logprobs = None
        mock_computed_blocks = [MagicMock(spec=KVCacheBlock), MagicMock(spec=KVCacheBlock)]
        mock_block_hashes = [MagicMock(spec=BlockHash), MagicMock(spec=BlockHash)]

        self.cache_manager.req_to_block_hashes["test_request"] = mock_block_hashes
        self.mock_manager.find_longest_cache_hit.return_value = [mock_computed_blocks]
        result = self.cache_manager.get_matched_num_and_touch(mock_request)

        expected_num_tokens = len(mock_computed_blocks) * self.cache_manager.block_size
        self.assertEqual(result, (expected_num_tokens, False))

        
        self.mock_manager.find_longest_cache_hit.assert_called_once_with(
            block_hashes=mock_block_hashes,
            max_length=9,  
            kv_cache_group_ids=[0],
            block_pool=self.mock_block_pool,
            kv_cache_spec=self.mock_manager.kv_cache_spec,
            use_eagle=True,
        )
        self.assertEqual(self.cache_manager.req_to_computed_blocks["test_request"], mock_computed_blocks)
        self.mock_block_pool.touch.assert_called_once_with([mock_computed_blocks])
        self.mock_cpu_cache_stats.set_cache_stats.assert_called_once_with(
            mock_request.num_tokens, expected_num_tokens
        )
        self.mock_cpu_cache_stats.cpu_prefix_cache_metrics.observe.assert_called_once_with(
            self.mock_cpu_cache_stats.prefix_cache_stats
        )
        self.mock_cpu_cache_stats.log.assert_called_once()

    def test__release_ahead_touch(self):
        request_id = "test_request"
        mock_computed_blocks = [MagicMock(spec=KVCacheBlock), MagicMock(spec=KVCacheBlock), MagicMock(spec=KVCacheBlock)]
        self.cache_manager.req_to_computed_blocks[request_id] = mock_computed_blocks
        
        self.cache_manager._release_ahead_touch(request_id)
        
        self.mock_manager.block_pool.free_blocks.assert_called_once()
        
        args, _ = self.mock_manager.block_pool.free_blocks.call_args
        actual_blocks = list(args[0])
        
        expected_blocks = list(reversed(mock_computed_blocks))
        self.assertEqual(actual_blocks, expected_blocks)
        
        self.assertNotIn(request_id, self.cache_manager.req_to_computed_blocks)

    def test_allocate_slots_normal(self):
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 0

        req_to_num_tokens = {"req1": 100, "req2": 200}
        unallocated_req_ids = {"req1", "req2"}
        
        self.mock_manager.single_type_manager.get_num_blocks_to_allocate.side_effect = [2, 3]
        self.mock_block_pool.get_num_free_blocks.return_value = 10  
        
        self.cache_manager.single_type_manager.allocate_new_blocks.side_effect = [
            self.create_mock_blocks(2, start_id=1),  
            self.create_mock_blocks(3, start_id=3)  
        ]
        
        result = self.cache_manager.allocate_slots(req_to_num_tokens, unallocated_req_ids)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["req1"], [1, 2]) 
        self.assertEqual(result["req2"], [3, 4, 5])

    def test_allocate_slots_insufficient_blocks(self):
        self.cache_manager._free_slots = MagicMock()
        self.cache_manager._release_ahead_touch = MagicMock()
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 0
        req_to_num_tokens = {"req1": 100}
        unallocated_req_ids = {"req1"}
        
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 5
        self.mock_block_pool.get_num_free_blocks.return_value = 3  
        
        result = self.cache_manager.allocate_slots(req_to_num_tokens, unallocated_req_ids)
        
        self.assertEqual(len(result), 0)
        self.assertTrue(self.cache_manager.req_failed_to_allocate["req1"])
        self.cache_manager._release_ahead_touch.assert_called_once_with("req1")

    def test_allocate_slots_skip_failed_requests(self):
        self.cache_manager._free_slots = MagicMock()
        self.cache_manager._release_ahead_touch = MagicMock()
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 0
        req_to_num_tokens = {"req1": 100}
        unallocated_req_ids = {"req1"}
        
        self.cache_manager.req_failed_to_allocate["req1"] = True
        
        result = self.cache_manager.allocate_slots(req_to_num_tokens, unallocated_req_ids)
        
        self.assertEqual(len(result), 0)
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.assert_not_called()

    def test_allocate_slots_with_computed_blocks(self):
        self.cache_manager._free_slots = MagicMock()
        self.cache_manager._release_ahead_touch = MagicMock()
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 0
        req_to_num_tokens = {"req1": 100}
        unallocated_req_ids = {"req1"}
        
        computed_blocks = self.create_mock_blocks(2, start_id=10)
        self.cache_manager.req_to_computed_blocks["req1"] = computed_blocks
        
        self.cache_manager.single_type_manager.get_num_blocks_to_allocate.return_value = 2
        self.mock_block_pool.get_num_free_blocks.return_value = 5
        self.cache_manager.single_type_manager.allocate_new_blocks.return_value = self.create_mock_blocks(2, start_id=1)
        
        result = self.cache_manager.allocate_slots(req_to_num_tokens, unallocated_req_ids)
        
        self.assertEqual(result["req1"], [10, 11, 1, 2])
        self.cache_manager.single_type_manager.save_new_computed_blocks.assert_called_once_with("req1", computed_blocks)
        self.assertNotIn("req1", self.cache_manager.req_to_computed_blocks)

    def test_record_request_cache_and_free_slots(self):
        self.cache_manager.req_to_free = {}
        test_request_id = "req_123"
        test_request = MagicMock(spec=Request)
        test_request.request_id = "req_123"
        
        self.cache_manager.record_request_cache_and_free_slots(test_request)
        
        self.assertIn(test_request_id, self.cache_manager.req_to_free)
        self.assertEqual(self.cache_manager.req_to_free[test_request_id], test_request)
    
    def test_cache_and_free_slots_normal(self):
        test_request_id = "req_123"
        test_num_tokens = 100
        test_request = MagicMock(spec=Request)
        self.cache_manager._free_slots = MagicMock()

        self.cache_manager.req_to_free[test_request_id] = test_request
        self.cache_manager.req_failed_to_allocate[test_request_id] = False
        self.cache_manager.req_to_num_tokens[test_request_id] = test_num_tokens
        
        self.cache_manager.cache_and_free_slots(test_request_id)
        
        self.cache_manager.single_type_manager.cache_blocks.assert_called_once_with(
            test_request, test_num_tokens
        )    
        self.cache_manager._free_slots.assert_called_once_with(test_request_id)
        self.assertNotIn(test_request_id, self.cache_manager.req_to_free)

    def test_cache_and_free_slots_fail(self):
        test_request_id = "req_123"
        test_num_tokens = 100
        test_request = MagicMock(spec=Request)
        self.cache_manager._free_slots = MagicMock()

        self.cache_manager.req_to_free[test_request_id] = test_request
        self.cache_manager.req_failed_to_allocate[test_request_id] = True
        self.cache_manager.req_to_num_tokens[test_request_id] = test_num_tokens
        
        self.cache_manager.cache_and_free_slots(test_request_id)
        
        self.cache_manager.single_type_manager.cache_blocks.assert_not_called()
        
        self.cache_manager._free_slots.assert_called_once_with(test_request_id)
        self.assertNotIn(test_request_id, self.cache_manager.req_to_free)

    def test_free_slots(self):
        self.cache_manager.req_to_block_hashes = {}
        self.cache_manager.req_to_computed_blocks = {}
        self.cache_manager.req_failed_to_allocate = {}
        self.cache_manager.req_to_num_tokens = {}
        
        self.cache_manager._release_ahead_touch = MagicMock()
        self.cache_manager.single_type_manager.free = MagicMock() 
        
        self.test_request_id = "req_123"
        self.cache_manager.req_to_block_hashes[self.test_request_id] = "hash_123"
        self.cache_manager.req_to_computed_blocks[self.test_request_id] = [MagicMock()]
        self.cache_manager.req_failed_to_allocate[self.test_request_id] = False
        self.cache_manager.req_to_num_tokens[self.test_request_id] = 100
        
        self.cache_manager._free_slots(self.test_request_id)
        
        self.cache_manager._release_ahead_touch.assert_called_once_with(self.test_request_id)
        self.cache_manager.single_type_manager.free.assert_called_once_with(self.test_request_id)
        
        self.assertNotIn(self.test_request_id, self.cache_manager.req_to_block_hashes)
        self.assertNotIn(self.test_request_id, self.cache_manager.req_to_computed_blocks)
        self.assertNotIn(self.test_request_id, self.cache_manager.req_failed_to_allocate)
        self.assertNotIn(self.test_request_id, self.cache_manager.req_to_num_tokens)

if __name__ == '__main__':
    unittest.main()
    
