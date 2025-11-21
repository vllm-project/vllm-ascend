# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import unittest
from dataclasses import is_dataclass


#TODO: 添加更多的测试用例   
class TestModelAdditionalConfig(unittest.TestCase):
    
    def test_basic_load_json_config(self):
        """测试基本配置加载功能"""
        os.environ['MODEL_EXTRA_CFG_PATH'] = 'test_config.json'
        from model_config import init_model_extra_config
        config = init_model_extra_config()
        
        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))
        
        # 验证配置值
        self.assertEqual(config.parall_config.dp_size, 18)
        
        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, False)
        self.assertEqual(config.operator_opt_config.prefill_moe_all_to_all, False)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, True)
        self.assertEqual(config.operator_opt_config.best_ep, True)
        self.assertEqual(config.operator_opt_config.merge_qkv, True)
        self.assertEqual(config.operator_opt_config.two_stage_comm, True)
        self.assertEqual(config.operator_opt_config.gmm_nz, True)
        self.assertEqual(config.operator_opt_config.decode_moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.use_omni_placement, True)
        self.assertEqual(config.operator_opt_config.omni_placement_config_path, ".")
        self.assertEqual(config.operator_opt_config.control_accept_rate, -1)
        self.assertEqual(config.operator_opt_config.decode_gear_list, [17])
        
    def test_default_config_when_no_json(self):
         # 准备测试数据
        from model_config import envs, init_model_extra_config
        envs.MODEL_EXTRA_CFG_PATH = ""
        config = init_model_extra_config()

        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))

        self.assertEqual(config.parall_config.dp_size, 1)

        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, True)
        self.assertEqual(config.operator_opt_config.prefill_moe_all_to_all, True)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, False)
        self.assertEqual(config.operator_opt_config.best_ep, False)
        self.assertEqual(config.operator_opt_config.merge_qkv, False)
        self.assertEqual(config.operator_opt_config.two_stage_comm, False)
        self.assertEqual(config.operator_opt_config.gmm_nz, False)
        self.assertEqual(config.operator_opt_config.decode_moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.use_omni_placement, False)
        self.assertEqual(config.operator_opt_config.omni_placement_config_path, None)
        self.assertEqual(config.operator_opt_config.control_accept_rate, -1)
        self.assertEqual(config.operator_opt_config.decode_gear_list, [16])

if __name__ == '__main__':
    unittest.main()
