# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

_INITIAL_GROUP_CAPACITY = 4096


class SharedSwiGLUCache:
    """Executor-local constants, prepared eagerly and never mutated by replay.

    Weight reloads replace weight_scale_fp32 and invalidate this cache. Explicit
    invalidate() is required for in-place writes to inference tensors without
    version counters. Calls belong to the shared-expert execution stream; this
    object is not shared between concurrent executors.
    """

    def __init__(self) -> None:
        self._source: torch.Tensor | None = None
        self._source_pointer = 0
        self._source_version: int | None = None
        self._weights: torch.Tensor | None = None
        self._groups: torch.Tensor | None = None
        self._retired_groups: list[torch.Tensor] = []

    def invalidate(self) -> None:
        self._source = None

    def get_parameters(self, weight_scale: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        same_weight = self._source is weight_scale and self._source_pointer == weight_scale.data_ptr()
        if same_weight and self._source_version is not None:
            same_weight = self._source_version == weight_scale._version
        needs_groups = self._groups is None or self._groups.numel() <= max(num_tokens, 8)
        if not same_weight or needs_groups:
            # A cold capture uses the stateless native path rather than retain
            # cache tensors allocated from a graph-private memory pool.
            if torch.npu.is_current_stream_capturing():  # type: ignore[attr-defined]
                return None
            if not same_weight:
                halves = weight_scale.detach().reshape(-1).chunk(2)
                self._weights = torch.stack(halves, dim=-1).reshape(1, -1)
                self._source = weight_scale
                self._source_pointer = weight_scale.data_ptr()
                try:
                    self._source_version = weight_scale._version
                except RuntimeError:
                    self._source_version = None
            if needs_groups:
                capacity = max(_INITIAL_GROUP_CAPACITY, 1 << max(num_tokens, 8).bit_length())
                if self._groups is not None:
                    self._retired_groups.append(self._groups)
                self._groups = torch.arange(capacity + 1, dtype=torch.int64, device=weight_scale.device)
        assert self._weights is not None and self._groups is not None
        return self._weights, self._groups


def shared_dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    clamp_limit: float,
    *,
    cache: SharedSwiGLUCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official clipped A3 shared SwiGLU with one native dispatch.

    The binding interleaves [gate | up] and repeats rows for small groups.
    No new custom OPP or computation kernel is required.
    """
    num_tokens, width = x.shape
    if num_tokens == 0:
        return x.new_empty((0, width // 2), dtype=torch.int8), x.new_empty((0,), dtype=torch.float32)

    parameters = None if cache is None else cache.get_parameters(weight_scale, num_tokens)
    if parameters is None:
        return torch.ops._C_ascend.official_shared_dequant_swiglu_quant(x, weight_scale, activation_scale, clamp_limit)
    weights, groups = parameters
    return torch.ops._C_ascend.official_shared_dequant_swiglu_quant(
        x, weights, activation_scale, clamp_limit, groups, True
    )
