#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for HCCL weight transfer draft-model sync helpers."""

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine


def test_set_draft_model_and_start_also_initializes_draft():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine._draft_model = None
    engine._draft_model_config = None
    draft = MagicMock()
    draft_cfg = MagicMock()

    engine.set_draft_model(draft, draft_cfg)

    with patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as mock_initialize:
        engine.start_weight_update()

    assert mock_initialize.call_count == 2
    mock_initialize.assert_any_call(engine.model)
    mock_initialize.assert_any_call(draft)


def test_finish_weight_update_finalizes_draft_with_draft_config():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock(name="target_cfg")
    draft = MagicMock()
    draft_cfg = MagicMock(name="draft_cfg")
    engine.set_draft_model(draft, draft_cfg)

    with patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as mock_finalize:
        engine.finish_weight_update()

    assert mock_finalize.call_count == 2
    mock_finalize.assert_any_call(engine.model, engine.model_config)
    mock_finalize.assert_any_call(draft, draft_cfg)


def test_load_weights_with_draft_loads_both_models():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    draft = MagicMock()
    engine.set_draft_model(draft, None)
    weights = [("a.weight", torch.ones(2))]

    engine._load_weights_with_draft(weights)

    engine.model.load_weights.assert_called_once()
    draft.load_weights.assert_called_once()
    assert engine.model.load_weights.call_args.args[0] == weights
    assert draft.load_weights.call_args.args[0] == weights


def test_load_weights_with_draft_skips_when_no_draft():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.set_draft_model(None, None)
    weights = [("a.weight", torch.ones(2))]

    engine._load_weights_with_draft(weights)

    engine.model.load_weights.assert_called_once_with(weights)
