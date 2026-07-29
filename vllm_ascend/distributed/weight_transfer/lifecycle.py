# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle policies for weight update flows."""

from collections.abc import Callable

import torch


class WeightUpdateLifecyclePolicy:
    """Base policy for weight update start, loading, and finish behavior."""

    def start(self, model: torch.nn.Module) -> None:
        pass

    def finish(self, model: torch.nn.Module, model_config) -> None:
        pass

    def make_load_weights(self, model: torch.nn.Module) -> Callable[[list[tuple[str, torch.Tensor]]], None]:
        raise NotImplementedError


class LayerwiseReloadLifecyclePolicy(WeightUpdateLifecyclePolicy):
    """Policy for checkpoint-format updates using vLLM layerwise reload."""

    def start(self, model: torch.nn.Module) -> None:
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        initialize_layerwise_reload(model)

    def finish(self, model: torch.nn.Module, model_config) -> None:
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(model, model_config)

    def make_load_weights(self, model: torch.nn.Module) -> Callable[[list[tuple[str, torch.Tensor]]], None]:
        return model.load_weights


class DirectWeightUpdateLifecyclePolicy(WeightUpdateLifecyclePolicy):
    """Policy for direct in-place parameter updates."""

    def make_load_weights(self, model: torch.nn.Module) -> Callable[[list[tuple[str, torch.Tensor]]], None]:
        def load_weights_direct(weights: list[tuple[str, torch.Tensor]]) -> None:
            with torch.no_grad():
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        return load_weights_direct


def get_weight_update_lifecycle_policy(is_checkpoint_format: bool) -> WeightUpdateLifecyclePolicy:
    if is_checkpoint_format:
        return LayerwiseReloadLifecyclePolicy()
    return DirectWeightUpdateLifecyclePolicy()
