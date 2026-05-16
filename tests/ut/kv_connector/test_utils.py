import unittest
import math

from vllm_ascend.distributed.kv_transfer.utils.utils import parallel_info, get_cp_group, context_parallel_parameters_check, get_tp_rank_head_mapping, get_head_group_mapping, get_local_remote_block_port_mappings, get_transfer_mappings

from unittest.mock import MagicMock

class TestUtils(unittest.TestCase):
    ###测试get_cp_group函数
    def test_get_cp_group_step_zero(self):
        #tp < heads
        result = get_cp_group(tp = 4, heads = 8, dcp = 2)
        #assert
        self.assertEqual(result,[[0, 1]])

    def test_get_cp_group_step_greater(self):
        #tp > heads
        result = get_cp_group(tp = 8, heads = 4, dcp = 2)
        #验证返回的是包含set的list
        self.assertEqual(result, [{0, 1, 2, 3}])

    #测试context_parallel_parameters_check函数
    def test_context_parallel_parameters_check_success(self):
        #arrange: 构造真实的parallel_info 实例
        p_info = parallel_info(
            tp_size = 4,
            pcp_size = 2,
            dcp_size = 2,
            use_mla = False,
            pd_head_ratio= 1
        )
        d_info = parallel_info(
            tp_size = 8,
            pcp_size = 1,
            dcp_size = 2,
            use_mla = False,
            pd_head_ratio= 1
        )

        try:
            context_parallel_parameters_check(
                remote_pcp_size = 1,
                remote_dcp_size = 4,
                p_parallel_info = p_info,
                d_parallel_info = d_info,
                total_num_kv_heads = 8
            )
        except AssertionError:
            self.fail("context_parallel_parameters_check 抛出了意外的AssertionError")
        
    def test_contest_parallel_parameters_check_failure(self):
        #arrange:故意构造一个无法整除的pcp-dcp比例
        p_info = parallel_info(
            tp_size= 4,
            pcp_size= 1,
            dcp_size= 1,
            use_mla=True,
            pd_head_ratio=1
        )
        d_info = parallel_info(
            tp_size= 4,
            pcp_size= 1,
            dcp_size= 1,
            use_mla=True,
            pd_head_ratio=1
        )

        #act and assert: 1 * 1 % 2 * 2 != 0 必然触发assert
        with self.assertRaises(AssertionError):
            context_parallel_parameters_check(
                remote_pcp_size = 2,
                remote_dcp_size = 2,
                p_parallel_info = p_info,
                d_parallel_info = d_info,
                total_num_kv_heads = 8
            )

    def test_get_tp_rank_head_mapping_less_than_heads(self):
        result = get_tp_rank_head_mapping(num_key_value_heads = 8, tp_size = 4)
        expected = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7]
        }
        self.assertDictEqual(result, expected)

    def test_get_tp_rank_head_mapping_raise_error(self):
        #故意传入不能整除的数字
        with self.assertRaises(ValueError):
            get_tp_rank_head_mapping(num_key_value_heads = 7, tp_size = 3)

    #测试get_head_group_mapping函数
    def test_get_head_group_mapping_success(self):
        result = get_head_group_mapping(
            num_key_value_heads = 8,
            tp_size = 4,
            num_groups= 2,
            select_cp_group = [0, 1]
        )
        expected = {
            0:[0, 1, 2, 3],
            1:[4, 5, 6, 7]
        }
        self.assertDictEqual(result, expected)

    def test_get_local_remote_block_port_mappings(self):
        #arrange: 构造真实的小规模数据，避免循环爆炸
        #假定：tp2，pcp1，dcp1
        p_info = parallel_info(
            tp_size= 2,
            pcp_size= 1,
            dcp_size= 1,
            use_mla=False,
            pd_head_ratio=1
        )
        d_info = parallel_info(
            tp_size= 2,
            pcp_size= 1,
            dcp_size= 1,
            use_mla=False,
            pd_head_ratio=1
        )

        req_meta = MagicMock()
        req_meta.remote_cache_tokens = 64

        #act: 执行这个mock函数，获取结果
        p_mapping, d_mapping, pd_mapping, trans_count = get_local_remote_block_port_mappings(
            to_trans_idx = 2,
            p_parallel_info = p_info,
            d_parallel_info= d_info,
            d_hosts = ["192.168.1.1"],
            d_port = 8000,
            select_p_cp_group = [0, 1],
            selected_d_cp_group = [0, 1],
            prompt_len = 128,
            block_size = 64,
            req_meta = req_meta,
            total_num_kv_heads = 4,
            req_id = "ut_test_01"
        )

        #assert: 验证输出结构和内容
        self.assertEqual(len(p_mapping), 1)
        self.assertEqual(len(p_mapping[0]), 2)
        #2.验证p和d的映射关系
        self.assertDictEqual(pd_mapping, {0: [0], 1: [1]})
        #3.验证传输计数器映射
        self.assertIn(("192.168.1.1", 8000), trans_count)
        self.assertIn(("192.168.1.1", 8001), trans_count)

    #测试最后的组装器get_transfer_mappings函数
    def test_get_transfer_mappings(self):
        #arrange:将上一个函数的“预期输出”作为这个函数的“输入”
        #构造这个中间状态字典
        p_rank_mapping = [[[[0, 1]]]]   #pcp = 1,head_group = 1, dcp = 1
        d_block_mapping = {
            0: {0:{"host": "127.0.0.1", "port": 8000, "block_idx": 0}},
            1: {0:{"host": "127.0.0.1", "port": 8000, "block_idx": 1}}
        }
        pd_mapping = {0: [0]}
        trans_count = {("127.0.0.1", 8000): 2}

        #伪造请求数据
        req_meta = MagicMock()
        req_meta.local_block_ids = [100, 101]   #本地显存中的block编号
        req_meta.remote_block_ids = [200, 201]  #远程显存中的block编号

        p_info = parallel_info(
            tp_size = 2,
            pcp_size = 1,
            dcp_size = 1,
            use_mla = False,
            pd_head_ratio= 1
        )

        #act
        result = get_transfer_mappings(
            p_rank_block_mapping = p_rank_mapping,
            d_block_rank_mapping = d_block_mapping,
            pd_head_mapping = pd_mapping,
            d_trans_count_mapping = trans_count,
            req_meta = req_meta,
            p_parallel_info = p_info,
            req_id = "ut_test_02",
            transed_idx = 0,
            to_trans_idx = 2,
            tp_rank = 0,
            pcp_rank = 0,
            dcp_rank = 0
        )

        #assert
        #最总结果 以(host, port)为键，包含本地和远程的映射
        expected_key = ("127.0.0.1", 8000)
        self.assertIn(expected_key, result)
        self.assertEqual(result[expected_key]["local_block_ids"], [100, 101])
        self.assertEqual(result[expected_key]["remote_block_ids"], [200, 201])
        self.assertEqual(result[expected_key]["trans_count"], 2)