# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Iterable

import torch
from vllm.distributed.utils import is_weak_contiguous
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class EplbExpertTensorList(list[torch.Tensor]):
    """Per-expert tensors exposed through the upstream EPLB weight contract."""

    @property
    def shape(self) -> torch.Size:
        return torch.Size((len(self), *self[0].shape))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.empty_like:
            source = args[0]
            return cls(torch.empty_like(tensor, **(kwargs or {})) for tensor in source)
        return NotImplemented


class AscendRoutedExperts(RoutedExperts):
    """RoutedExperts with layout-aware Ascend EPLB weight views."""

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        try:
            get_weight_views = self.quant_method.get_eplb_weight_views
        except AttributeError as exc:
            raise NotImplementedError(
                f"{self.quant_method.__class__.__name__} must implement get_eplb_weight_views() for Ascend EPLB."
            ) from exc
        weights = list(get_weight_views(self))
        if not weights:
            raise NotImplementedError(f"EPLB weight views are not defined for {self.quant_method.__class__.__name__}.")
        flattened_weights = []
        for weight in weights:
            if isinstance(weight, (list, tuple)):
                if len(weight) != self.local_num_experts:
                    raise ValueError(
                        "Every EPLB expert tensor list must contain "
                        f"local_num_experts ({self.local_num_experts}) tensors, got {len(weight)}."
                    )
                if not all(is_weak_contiguous(expert_weight) for expert_weight in weight):
                    raise ValueError("Every tensor in an Ascend EPLB expert tensor list must be weakly contiguous.")
                flattened_weights.append(EplbExpertTensorList(weight))
                continue
            if weight.shape[0] != self.local_num_experts:
                raise ValueError(
                    "The first dimension of every EPLB weight view must equal "
                    f"local_num_experts ({self.local_num_experts}), got {tuple(weight.shape)}."
                )
            if not is_weak_contiguous(weight):
                raise ValueError("Every Ascend EPLB weight view must be weakly contiguous.")
            try:
                flattened_weights.append(weight.view(self.local_num_experts, -1))
            except RuntimeError as exc:
                raise ValueError("Every Ascend EPLB expert row must be flattenable without a copy.") from exc
        return flattened_weights
