import unittest

from vllm_ascend.distributed.kv_transfer.utils.utils import get_cp_group, get_tp_rank_head_mapping, parallel_info

class TestParallelTopologyUtils(unittest.TestCase):

    def test_get_cp_group_step_zero(self):
        tp = 4
        heads = 8
        dcp = 2

        result = get_cp_group(tp, heads, dcp)

        self.assertEqual(result, [[0, 1]])

    def test_get_cp_group_step_greater_than_zero(self):
        tp = 8
        heads = 4
        dcp = 2

        result = get_cp_group(tp, heads, dcp)

        self.assertEqual(result, [{0, 1, 2, 3}])

    # --- 用例 3：头数大于卡数，且能整除 (正常分支 1) ---
    def test_get_tp_rank_head_mapping_tp_less_than_heads(self):
        # 1. Arrange
        num_heads = 8
        tp_size = 4
        
        # 2. Act
        result = get_tp_rank_head_mapping(num_heads, tp_size)
        
        # 3. Assert
        # 预期：8 个头分给 4 张卡，每张卡 2 个头。
        expected_mapping = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7]
        }
        self.assertDictEqual(result, expected_mapping)

    # --- 用例 4：卡数大于头数，且能整除 (正常分支 2) ---
    def test_get_tp_rank_head_mapping_tp_greater_than_heads(self):
        # 1. Arrange
        num_heads = 2
        tp_size = 4
        
        # 2. Act
        result = get_tp_rank_head_mapping(num_heads, tp_size)
        
        # 3. Assert
        # 预期：2 个头分给 4 张卡，每张卡处理 0.5 个头（也就是多卡共用一个头）
        expected_mapping = {
            0: [0], 1: [0], # 卡 0 和卡 1 负责头 0
            2: [1], 3: [1]  # 卡 2 和卡 3 负责头 1
        }
        self.assertDictEqual(result, expected_mapping)

    # --- 用例 5：不能整除时抛出异常 (异常分支测试) ---
    def test_get_tp_rank_head_mapping_raises_error_if_not_divisible(self):
        # 1. Arrange
        num_heads = 7 # 7 个头
        tp_size = 4   # 分给 4 张卡，不能整除
        
        # 2 & 3. Act & Assert (测试异常通常写在一起)
        # 用 assertRaises 捕获异常，如果没抛出异常，测试就会失败
        with self.assertRaises(ValueError) as context:
            get_tp_rank_head_mapping(num_heads, tp_size)
        
        # (可选) 进一步验证异常的信息是不是我们预期的
        self.assertIn("cannot be evenly divided", str(context.exception))