# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb import policy as eplb_policy
from vllm_ascend.distributed.eplb.policy import AscendV2EplbPolicy


def test_compute_hotness_imbalance_matches_v1_metric() -> None:
    logical_load = torch.tensor(
        [
            [100, 20],
            [40, 40],
        ]
    )
    mapping = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ]
    )

    mean, maximum, per_layer = AscendV2EplbPolicy._compute_hotness_imbalance(
        logical_load,
        mapping,
        num_ranks=2,
    )

    assert per_layer == pytest.approx([5 / 3, 1.0])
    assert mean == pytest.approx(4 / 3)
    assert maximum == pytest.approx(5 / 3)


def test_log_hotness_imbalance_reports_actual_selected_mapping(
    monkeypatch,
) -> None:
    policy = AscendV2EplbPolicy.__new__(AscendV2EplbPolicy)
    policy.ep_rank = 0
    log_info = MagicMock()
    monkeypatch.setattr(eplb_policy.logger, "info", log_info)

    policy._log_hotness_imbalance(
        logical_load=torch.tensor([[100, 20]]),
        old_mapping=torch.tensor([[0, 0, 1, 1]]),
        new_mapping=torch.tensor([[0, 1, 0, 1]]),
        num_ranks=2,
    )

    assert policy.latest_expert_hotness == {
        "current_mean": pytest.approx(5 / 3),
        "current_max": pytest.approx(5 / 3),
        "update_mean": pytest.approx(1.0),
        "update_max": pytest.approx(1.0),
        "current_imbalance_list": pytest.approx([5 / 3]),
        "update_imbalance_list": pytest.approx([1.0]),
    }
    log_info.assert_called_once_with(
        "[eplb/worker] Expert hotness imbalance, current: mean=%.3f max=%.3f, updated: mean=%.3f max=%.3f",
        pytest.approx(5 / 3),
        pytest.approx(5 / 3),
        pytest.approx(1.0),
        pytest.approx(1.0),
    )


def test_nonzero_ep_rank_does_not_log_hotness(monkeypatch) -> None:
    policy = AscendV2EplbPolicy.__new__(AscendV2EplbPolicy)
    policy.ep_rank = 1
    log_info = MagicMock()
    monkeypatch.setattr(eplb_policy.logger, "info", log_info)

    policy._log_hotness_imbalance(
        logical_load=torch.tensor([[100, 20]]),
        old_mapping=torch.tensor([[0, 0, 1, 1]]),
        new_mapping=torch.tensor([[0, 1, 0, 1]]),
        num_ranks=2,
    )

    log_info.assert_not_called()
    assert not hasattr(policy, "latest_expert_hotness")
