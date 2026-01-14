import os
import sys
import unittest
from unittest.mock import patch

# isort: off
import pytest
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                         FusedMoEParallelConfig
                                                         )

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils, init_eplb_config
# isort: on


class TestAscendConfig(unittest.TestCase):

    def setUp(self):
        vllm_config = VllmConfig()
        ascend_config = init_ascend_config(vllm_config)
        ascend_config.dynamic_eplb = True
        ascend_config.init_redundancy_expert = 2
        moe_parallel_config = FusedMoEParallelConfig(2, 0, 1, 2, 1, 1, 1, 1,
                                                     True, "hccl")
        moe_config = FusedMoEConfig(8, 8, 8192, 5, moe_parallel_config,
                                    torch.float16)
        moe_config.supports_eplb = True
        self.ascend_config = ascend_config
        self.moe_config = moe_config
        self.mock_npu = patch("torch.Tensor.npu",
                              new=lambda self: self).start()
        self.rank = 1

    def test_init_eplb_config_with_eplb(self):
        expert_map, log2phy, redundant_experts = init_eplb_config(
            self.ascend_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([4, -1, -1, -1, 0, 1, 2, 3])
        gt_log2phy = torch.tensor([9, 1, 2, 3, 5, 6, 7, 8])
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_with_eplb_withmap(self):
        _TEST_DIR = os.path.dirname(__file__)
        self.ascend_config.expert_map_path = _TEST_DIR + "/expert_map.json"
        expert_map, log2phy, redundant_experts = init_eplb_config(
            self.ascend_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3])
        gt_log2phy = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8])
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_without_eplb(self):
        self.ascend_config.dynamic_eplb = False
        self.ascend_config.expert_map_path = None
        expert_map, log2phy, redundant_experts = init_eplb_config(
            self.ascend_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        print(expert_map, log2phy, redundant_experts)
        self.assertTrue(torch.equal(expert_map[self.rank], gt_expert_map))
        self.assertEqual(redundant_experts, 0)


class TestEPLBParamUtils:

    def test_check_iterations_valid(self):
        EPLBParamUtils.check_iterations(1)
        EPLBParamUtils.check_iterations(100)

    def test_check_iterations_type_error(self):
        with pytest.raises(TypeError, match="is not int"):
            EPLBParamUtils.check_iterations("abc")
        with pytest.raises(TypeError, match="is not int"):
            EPLBParamUtils.check_iterations(1.5)
        with pytest.raises(TypeError, match="is not int"):
            EPLBParamUtils.check_iterations(None)

    def test_check_iterations_value_error_less_than_or_equal_zero(self):
        with pytest.raises(ValueError,
                           match="can not less than or equal to 0"):
            EPLBParamUtils.check_iterations(0)
        with pytest.raises(ValueError,
                           match="can not less than or equal to 0"):
            EPLBParamUtils.check_iterations(-1)

    def test_check_iterations_value_error_large_than_sys_maxsize(self):
        large_value = sys.maxsize + 1
        with pytest.raises(ValueError,
                           match=f"can not large than {sys.maxsize}"):
            EPLBParamUtils.check_iterations(large_value)

    def test_check_dynamic_eplb_none(self):
        EPLBParamUtils.check_dynamic_eplb(None)

    def test_check_dynamic_eplb_valid_bool(self):
        EPLBParamUtils.check_dynamic_eplb(False)

    def test_check_dynamic_eplb_type_error(self):
        with pytest.raises(TypeError, match="The dynamic_eplb is not bool."):
            EPLBParamUtils.check_dynamic_eplb("true")
        with pytest.raises(TypeError, match="The dynamic_eplb is not bool."):
            EPLBParamUtils.check_dynamic_eplb(1)

    def test_check_dynamic_eplb_value_error_env_not_set(self, monkeypatch):
        monkeypatch.delenv("DYNAMIC_EPLB", raising=False)
        with pytest.raises(
                ValueError,
                match=
                'Can not enable dynamic_eplb when DYNAMIC_EPLB is not set to "true" or "1".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "false")
        with pytest.raises(
                ValueError,
                match=
                'Can not enable dynamic_eplb when DYNAMIC_EPLB is not set to "true" or "1".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "any_other_value")
        with pytest.raises(
                ValueError,
                match=
                'Can not enable dynamic_eplb when DYNAMIC_EPLB is not set to "true" or "1".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

    def test_check_dynamic_eplb_valid_with_env_set(self, monkeypatch):
        monkeypatch.setenv("DYNAMIC_EPLB", "true")
        EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "True")
        EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "1")
        EPLBParamUtils.check_dynamic_eplb(True)

    def test_check_expert_map_path_none(self):
        EPLBParamUtils.check_expert_map_path(None)

    def test_check_expert_map_path_type_error_not_string(self):
        with pytest.raises(TypeError, match="The expert_map is not str."):
            EPLBParamUtils.check_expert_map_path(123)
        with pytest.raises(TypeError, match="The expert_map is not str."):
            EPLBParamUtils.check_expert_map_path(True)

    def test_check_expert_map_path_value_error_empty_string(self):
        with pytest.raises(ValueError, match="The expert_map is not empty."):
            EPLBParamUtils.check_expert_map_path("")
        with pytest.raises(ValueError, match="The expert_map is not empty."):
            EPLBParamUtils.check_expert_map_path("   ")

    def test_check_expert_map_path_type_error_incorrect_extension(self):
        with pytest.raises(TypeError, match="The expert_map is not json."):
            EPLBParamUtils.check_expert_map_path("path/to/map.txt")
        with pytest.raises(TypeError, match="The expert_map is not json."):
            EPLBParamUtils.check_expert_map_path("path/to/map.JSON_")

    @patch('os.path.exists', return_value=False)
    def test_check_expert_map_path_value_error_not_exist(self, mock_exists):
        with pytest.raises(ValueError, match="The expert_map is not exist."):
            EPLBParamUtils.check_expert_map_path("non_existent_map.json")
        mock_exists.assert_called_once_with("non_existent_map.json")

    def test_check_expert_map_record_path_none(self):
        EPLBParamUtils.check_expert_map_record_path(None)

    def test_check_expert_map_record_path_type_error_not_string(self):
        with pytest.raises(TypeError,
                           match="The expert_map_record_path is not str."):
            EPLBParamUtils.check_expert_map_record_path(123)
        with pytest.raises(TypeError,
                           match="The expert_map_record_path is not str."):
            EPLBParamUtils.check_expert_map_record_path(False)

    def test_check_expert_map_record_path_value_error_empty_string(self):
        with pytest.raises(ValueError,
                           match="The expert_map_record_path is empty."):
            EPLBParamUtils.check_expert_map_record_path("")
        with pytest.raises(ValueError,
                           match="The expert_map_record_path is empty."):
            EPLBParamUtils.check_expert_map_record_path("   ")

    def test_check_expert_map_record_path_type_error_incorrect_extension(self):
        with pytest.raises(TypeError,
                           match="The expert_map_record_path is not json."):
            EPLBParamUtils.check_expert_map_record_path("path/to/record.txt")
        with pytest.raises(TypeError,
                           match="The expert_map_record_path is not json."):
            EPLBParamUtils.check_expert_map_record_path("path/to/record.XML")

    def test_check_expert_map_record_path_value_error_env_not_set(
            self, monkeypatch):
        monkeypatch.delenv("EXPERT_MAP_RECORD", raising=False)
        with pytest.raises(
                ValueError,
                match=
                'Can not enable expert_map_record_path when not export EXPERT_MAP_RECORD="true".'
        ):
            EPLBParamUtils.check_expert_map_record_path("path/to/record.json")

        monkeypatch.setenv("EXPERT_MAP_RECORD", "false")
        with pytest.raises(
                ValueError,
                match=
                'Can not enable expert_map_record_path when not export EXPERT_MAP_RECORD="true".'
        ):
            EPLBParamUtils.check_expert_map_record_path("path/to/record.json")
