# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.weight_transfer.lifecycle import (
    DirectWeightUpdateLifecyclePolicy,
    LayerwiseReloadLifecyclePolicy,
    get_weight_update_lifecycle_policy,
)


def test_get_weight_update_lifecycle_policy_selects_checkpoint_policy():
    assert isinstance(get_weight_update_lifecycle_policy(True), LayerwiseReloadLifecyclePolicy)


def test_get_weight_update_lifecycle_policy_selects_direct_policy():
    assert isinstance(get_weight_update_lifecycle_policy(False), DirectWeightUpdateLifecyclePolicy)


def test_layerwise_reload_policy_delegates_start_and_finish():
    policy = LayerwiseReloadLifecyclePolicy()
    model = MagicMock()
    model_config = MagicMock()

    with (
        patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as init_reload,
        patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as finalize_reload,
    ):
        policy.start(model)
        policy.finish(model, model_config)

    init_reload.assert_called_once_with(model)
    finalize_reload.assert_called_once_with(model, model_config)


def test_layerwise_reload_policy_uses_model_load_weights():
    policy = LayerwiseReloadLifecyclePolicy()
    model = MagicMock()

    assert policy.make_load_weights(model) is model.load_weights


def test_direct_policy_copies_weights_to_parameters():
    policy = DirectWeightUpdateLifecyclePolicy()
    param = torch.zeros(2)
    weight = torch.tensor([1.0, 2.0])
    model = MagicMock()
    model.get_parameter.return_value = param

    load_weights = policy.make_load_weights(model)
    load_weights([("model.weight", weight)])

    model.get_parameter.assert_called_once_with("model.weight")
    assert torch.equal(param, weight)
