# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize("enabled,probabilistic", [(False, False), (True, True), (True, False)])
def test_sampling_dispatch(monkeypatch, enabled, probabilistic):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.draft_logits = object() if probabilistic else None
    spec._step_cols = [torch.tensor(0)]
    spec._maybe_predict_acceptance = MagicMock()
    spec.model = SimpleNamespace(map_draft_to_target=MagicMock(side_effect=lambda x: x + 10))
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(enable_dspark_split_argmax=enabled))
    fast = MagicMock(return_value=torch.tensor([3]))
    original = MagicMock(return_value=torch.tensor([9]))
    monkeypatch.setattr(module, "greedy_argmax", fast)
    monkeypatch.setattr(DSparkSpeculator, "_sample_logits", original)
    logits = torch.zeros(1, 4096)
    mapping = torch.tensor([0])
    position = torch.tensor([42])
    result = spec._sample_logits(logits, mapping, position, 0)
    if enabled and not probabilistic:
        assert result.tolist() == [13]
        fast.assert_called_once_with(logits)
        spec._maybe_predict_acceptance.assert_called_once_with(logits, mapping, spec._step_cols[0])
        original.assert_not_called()
    else:
        assert result.tolist() == [9]
        original.assert_called_once_with(logits, mapping, position, 0)
        fast.assert_not_called()
        spec._maybe_predict_acceptance.assert_not_called()
