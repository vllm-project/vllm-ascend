import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker


def build_worker():
    worker = EplbWorker.__new__(EplbWorker)
    worker.rank_id = 0
    return worker


def test_compose_update_info_uses_precomputed_expert_sources():
    worker = build_worker()
    current_expert_maps = torch.tensor(
        [
            [
                [0, 1, -1, -1],
                [-1, 0, 1, -1],
                [-1, -1, -1, 0],
            ]
        ]
    )
    updated_expert_maps = torch.tensor(
        [
            [
                [0, -1, -1, 1],
                [-1, 0, 1, -1],
                [-1, -1, 2, -1],
            ]
        ]
    )

    update_info = list(worker.compose_expert_update_info_greedy(updated_expert_maps, current_expert_maps))

    send_info, recv_info, new_expert_map, layer_id = update_info[0]
    assert layer_id == 0
    assert torch.equal(new_expert_map, updated_expert_maps[0])
    assert send_info == {1: [(2, 2)], 2: [(0, 3)]}
    assert recv_info == {0: [(2, 3)], 2: [(1, 2)]}


def test_compose_update_info_skips_unchanged_layer():
    worker = build_worker()
    current_expert_maps = torch.tensor([[[0, 1], [1, 0]]])

    update_info = list(worker.compose_expert_update_info_greedy(current_expert_maps, current_expert_maps))

    send_info, recv_info, new_expert_map, layer_id = update_info[0]
    assert layer_id == 0
    assert send_info == {}
    assert recv_info == {}
    assert torch.equal(new_expert_map, current_expert_maps[0])


def test_compose_update_info_copies_replica_from_first_holder():
    worker = build_worker()
    current_expert_maps = torch.tensor([[[0, -1], [-1, 0], [-1, -1]]])
    updated_expert_maps = torch.tensor([[[0, -1], [-1, 0], [1, -1]]])

    update_info = list(worker.compose_expert_update_info_greedy(updated_expert_maps, current_expert_maps))

    send_info, recv_info, _new_expert_map, _layer_id = update_info[0]
    assert send_info == {0: [(2, 0)]}
    assert recv_info == {2: [(0, 0)]}
