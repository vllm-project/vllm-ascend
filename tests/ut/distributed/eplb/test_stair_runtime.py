from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb import stair_worker


def test_rank_zero_plans_for_existing_async_worker(monkeypatch):
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    monkeypatch.setattr(
        stair_worker,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=group, cpu_group=SimpleNamespace(size=lambda: 1)),
    )
    communicator = SimpleNamespace()
    model_state = SimpleNamespace(
        eplb_stats=SimpleNamespace(
            global_expert_load_window=torch.tensor([[[1, 1, 1, 20]]]),
            num_nodes=1,
        ),
        _stair_accepted_scores=np.array([np.nan]),
        communicator=communicator,
    )
    state = SimpleNamespace(
        _stair_config=StairConfig(
            hysteresis_enabled=False,
            p95_regression_tolerance=1.0,
            lpt_max_backtracks=64,
        )
    )
    old = torch.tensor([[0, 1, 2, 3, 0, 1]])

    new = stair_worker.run_stair_planner(model_state, state, old, None)

    assert new.shape == old.shape
    assert communicator._stair_source_rank.shape == (1, 2, 3)
    assert communicator._stair_source_slot.shape == (1, 2, 3)
    assert np.isfinite(model_state._stair_candidate_scores[0])
