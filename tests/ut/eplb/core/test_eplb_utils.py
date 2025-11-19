import random
import sys
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.eplb.core import eplb_utils
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils


def test_determine_default_expert_map_single_world():
    count, expert_map = eplb_utils.determine_default_expert_map(
        global_expert_num=4,
        world_size=1,
        rank_id=0,
        global_redundant_expert_num=0)
    assert count == 4
    assert torch.equal(expert_map, torch.arange(4, dtype=torch.int32))


def test_determine_default_expert_map_multiple_worlds_no_redundant():
    count, expert_map = eplb_utils.determine_default_expert_map(
        global_expert_num=8,
        world_size=2,
        rank_id=0,
        global_redundant_expert_num=0)

    assert count == 4
    assert torch.all(expert_map[:4] >= 0)
    assert torch.all(expert_map[4:] == -1)


def test_determine_default_expert_map_multiple_worlds_with_redundant():
    count, expert_map = eplb_utils.determine_default_expert_map(
        global_expert_num=5,
        world_size=2,
        rank_id=0,
        global_redundant_expert_num=1)

    assert count == 2
    assert torch.all(expert_map[0:2] >= 0)


def test_generate_log2phy_map_single_rank_holding():

    expert_map = torch.tensor([[0, -1], [-1, 0]], dtype=torch.int32)
    log2phy_map = eplb_utils.generate_log2phy_map(expert_map)

    assert torch.all(log2phy_map[:, 0] == log2phy_map[0, 0])
    assert torch.all(log2phy_map[:, 1] == log2phy_map[1, 1])


def test_generate_log2phy_map_multiple_rank_holding(monkeypatch):

    expert_map = torch.tensor([[0], [0]], dtype=torch.int32)

    monkeypatch.setattr(random, "choice", lambda x: x[0])

    log2phy_map = eplb_utils.generate_log2phy_map(expert_map)

    assert log2phy_map.shape == (2, 1)
    assert (log2phy_map >= 0).all()


def test_determine_default_log2phy_map_world_size_1():
    log2phy = eplb_utils.determine_default_log2phy_map(global_expert_num=3,
                                                       world_size=1,
                                                       rank_id=0)
    assert log2phy.shape == (3, )
    assert (log2phy >= 0).all()


def test_determine_default_log2phy_map_world_size_multiple():
    log2phy = eplb_utils.determine_default_log2phy_map(global_expert_num=6,
                                                       world_size=2,
                                                       rank_id=1)
    assert log2phy.shape == (6, )
    assert (log2phy >= 0).all()


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
                'Can not enable dynamic_eplb when not export DYNAMIC_EPLB="true".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "false")
        with pytest.raises(
                ValueError,
                match=
                'Can not enable dynamic_eplb when not export DYNAMIC_EPLB="true".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

        monkeypatch.setenv("DYNAMIC_EPLB", "any_other_value")
        with pytest.raises(
                ValueError,
                match=
                'Can not enable dynamic_eplb when not export DYNAMIC_EPLB="true".'
        ):
            EPLBParamUtils.check_dynamic_eplb(True)

    def test_check_dynamic_eplb_valid_with_env_set(self, monkeypatch):
        monkeypatch.setenv("DYNAMIC_EPLB", "true")
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
