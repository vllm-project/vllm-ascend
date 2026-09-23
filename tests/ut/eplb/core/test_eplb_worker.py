# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_ascend.eplb.core import eplb_worker


def test_do_update_logs_comparable_rebalance_metrics(monkeypatch):
    worker = object.__new__(eplb_worker.EplbWorker)
    worker.rank_id = 0
    worker.multi_stage = False
    worker.num_local_experts = 1
    worker.old_expert_maps = torch.tensor(
        [
            [[0, -1], [-1, 0]],
            [[0, -1], [-1, 0]],
        ]
    )
    new_expert_maps = worker.old_expert_maps.clone()
    new_expert_maps[0] = torch.tensor([[-1, 0], [0, -1]])
    old_placement = torch.zeros((2, 2, 1), dtype=torch.long)
    new_placement = old_placement.clone()

    worker.fetch_and_sum_load_info = MagicMock(return_value=np.ones((2, 2)))
    worker.global2local = MagicMock(return_value=old_placement)
    worker.calculate_rebalance_experts = MagicMock(return_value=(True, [0], new_placement))
    worker._calculate_hotness = MagicMock(return_value=np.ones((2, 2)))
    worker._compute_imbalance = MagicMock(side_effect=[(1.5, 2.0, [1.0, 2.0]), (1.25, 1.5, [1.0, 1.5])])
    worker.check_expert_placement = MagicMock()
    worker.local2global = MagicMock(return_value=new_expert_maps)
    worker.update_expert_map = MagicMock()
    worker.compose_expert_update_info_greedy = MagicMock(return_value=iter(()))
    worker.pack_update_info = MagicMock(return_value=[])
    log_info = MagicMock()
    monkeypatch.setattr(eplb_worker.logger, "info", log_info)

    assert worker.do_update() == []

    log_info.assert_called_once_with(
        "[eplb/worker] Expert hotness imbalance, current: mean=%.3f p95=%.3f max=%.3f, "
        "updated: mean=%.3f p95=%.3f max=%.3f, changed_layers=%d rank_transfers=%d",
        1.5,
        1.95,
        2.0,
        1.25,
        1.475,
        1.5,
        1,
        2,
    )
