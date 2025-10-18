import random

import torch

from vllm_ascend.eplb.core import eplb_utils


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
    log2phy = eplb_utils.determine_default_log2phy_map(
        global_expert_num=3,
        world_size=1,
        rank_id=0,
        global_redundant_expert_num=0)
    assert log2phy.shape == (3, )
    assert (log2phy >= 0).all()


def test_determine_default_log2phy_map_world_size_multiple():
    log2phy = eplb_utils.determine_default_log2phy_map(
        global_expert_num=6,
        world_size=2,
        rank_id=1,
        global_redundant_expert_num=1)
    assert log2phy.shape == (6, )
    assert (log2phy >= 0).all()
