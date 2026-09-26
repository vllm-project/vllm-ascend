import torch

from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.eplb.core.eplb_worker import EplbWorker


def _make_worker(shared_dict, rank_id=0, tp_size=None):
    worker = EplbWorker.__new__(EplbWorker)
    worker.shared_dict = shared_dict
    worker.rank_id = rank_id
    worker.tp_size = tp_size
    return worker


def _update_info(new_expert_map, layer_id=0):
    return [({0: []}, {0: []}, new_expert_map, layer_id)]


def _physical_expert_map():
    # Full-length physical maps (entry p = local slot of physical expert p,
    # -1 when not owned): rank 0 owns physical 0..4, rank 1 owns 5..9.
    return torch.tensor(
        [
            [0, 1, 2, 3, 4, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 0, 1, 2, 3, 4],
        ],
        dtype=torch.int32,
    )


def test_pack_update_info_uses_shared_dict_phys_to_logical():
    # shared_dict carries the CPU phys_to_logical bridge (see
    # test_warm_up_eplb_stores_cpu_phys_to_logical); pack_update_info must
    # consume it so the packed log2phy stays logical-length even though the
    # map rows are full-length physical maps.
    new_expert_map = _physical_expert_map()
    # Physical 8/9 are the redundant copies of logical 0/5.
    phys_to_logical = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 5], dtype=torch.int32)
    worker = _make_worker({"phys_to_logical": phys_to_logical}, rank_id=0)

    packed = worker.pack_update_info(iter(_update_info(new_expert_map)))

    assert len(packed) == 1
    send, recv, maps, log2phy, layer_ids = packed[0]
    assert send == [] and recv == []
    # The worker ships the rank's full physical row, not just the owned slots.
    assert maps == [0, 1, 2, 3, 4, -1, -1, -1, -1, -1]
    assert layer_ids == 0
    # Logical length (8), not physical length (10).
    assert log2phy == [0, 1, 2, 3, 4, 5, 6, 7]


def test_pack_update_info_falls_back_to_legacy_without_phys_to_logical():
    # No phys_to_logical key in shared_dict (non-EPLB / static EPLB): the
    # legacy branch reads the map rows directly as logical IDs.
    new_expert_map = _physical_expert_map()
    worker = _make_worker({}, rank_id=0)

    packed = worker.pack_update_info(iter(_update_info(new_expert_map)))

    log2phy = packed[0][3]
    expected = generate_log2phy_map(new_expert_map, 0)
    assert log2phy == expected.numpy().tolist()
    assert len(log2phy) == 10
