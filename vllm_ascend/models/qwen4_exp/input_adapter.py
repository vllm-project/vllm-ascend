# SPDX-License-Identifier: Apache-2.0
"""Fixed-address Model Runner V1 inputs for Qwen4Exp PLE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Qwen4ExpInputContext:
    """Runtime state required to prepare Qwen4Exp auxiliary inputs."""

    num_reqs: int
    num_reqs_padded: int
    is_dummy: bool
    query_start_loc: torch.Tensor
    num_computed_tokens: np.ndarray
    token_ids: np.ndarray


def pad_ple_graph_inputs(
    context_buffer: torch.Tensor,
    query_start_loc_buffer: torch.Tensor,
    runtime_context: torch.Tensor,
    runtime_query_start_loc: torch.Tensor,
    num_reqs: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refresh fixed-address PLE inputs for ACL graph replay."""
    max_num_reqs = context_buffer.shape[0]
    if not 0 < num_reqs <= max_num_reqs:
        raise ValueError(f"PLE num_reqs must be in [1, {max_num_reqs}], got {num_reqs}")
    if runtime_context.shape != context_buffer[:num_reqs].shape:
        raise ValueError(
            "PLE runtime context shape does not match the active graph rows: "
            f"expected={tuple(context_buffer[:num_reqs].shape)}, "
            f"actual={tuple(runtime_context.shape)}"
        )
    if runtime_query_start_loc.shape != (num_reqs + 1,):
        raise ValueError(
            "PLE runtime query_start_loc must have one boundary per active "
            f"request: expected={(num_reqs + 1,)}, "
            f"actual={tuple(runtime_query_start_loc.shape)}"
        )

    context_buffer.fill_(eos_token_id)
    context_buffer[:num_reqs].copy_(runtime_context, non_blocking=False)
    query_start_loc_buffer[: num_reqs + 1].copy_(runtime_query_start_loc, non_blocking=False)
    query_start_loc_buffer[num_reqs + 1 :].fill_(runtime_query_start_loc[-1])
    return context_buffer, query_start_loc_buffer


class Qwen4ExpPLEInputAdapter:
    """Own PLE history inputs and their graph-stable backing allocations."""

    def __init__(
        self,
        text_config: Any,
        *,
        max_num_reqs: int,
        device: torch.device,
    ) -> None:
        self.context_len = int(text_config.ngram_size) - 1
        if self.context_len <= 0:
            raise ValueError("Qwen4Exp PLE requires ngram_size >= 2")
        self.eos_token_id = int(text_config.eos_token_id)
        self.max_num_reqs = max_num_reqs
        self.device = device
        self._context_buffer: torch.Tensor | None = None
        self._query_start_loc_buffer: torch.Tensor | None = None

    def _get_context_buffer(self) -> torch.Tensor:
        expected_shape = (self.max_num_reqs, self.context_len)
        if self._context_buffer is None:
            self._context_buffer = torch.empty(
                expected_shape,
                dtype=torch.int32,
                device=self.device,
            )
        elif (
            self._context_buffer.shape != expected_shape
            or self._context_buffer.dtype != torch.int32
            or self._context_buffer.device != self.device
        ):
            raise RuntimeError(
                "Qwen4Exp PLE persistent ngram_context buffer changed "
                f"unexpectedly: shape={tuple(self._context_buffer.shape)}, "
                f"dtype={self._context_buffer.dtype}, "
                f"device={self._context_buffer.device}"
            )
        return self._context_buffer

    def _get_query_start_loc_buffer(self) -> torch.Tensor:
        expected_shape = (self.max_num_reqs + 1,)
        if self._query_start_loc_buffer is None:
            self._query_start_loc_buffer = torch.empty(
                expected_shape,
                dtype=torch.int32,
                device=self.device,
            )
        elif (
            self._query_start_loc_buffer.shape != expected_shape
            or self._query_start_loc_buffer.dtype != torch.int32
            or self._query_start_loc_buffer.device != self.device
        ):
            raise RuntimeError(
                "Qwen4Exp PLE persistent query_start_loc buffer changed "
                f"unexpectedly: shape={tuple(self._query_start_loc_buffer.shape)}, "
                f"dtype={self._query_start_loc_buffer.dtype}, "
                f"device={self._query_start_loc_buffer.device}"
            )
        return self._query_start_loc_buffer

    def _build_runtime_context(
        self,
        context: Qwen4ExpInputContext,
    ) -> torch.Tensor:
        runtime_context = np.full(
            (context.num_reqs_padded, self.context_len),
            self.eos_token_id,
            dtype=np.int32,
        )
        if context.is_dummy:
            return torch.from_numpy(runtime_context)

        for req_idx in range(context.num_reqs):
            context_end = int(context.num_computed_tokens[req_idx])
            context_start = max(0, context_end - self.context_len)
            tokens = context.token_ids[req_idx, context_start:context_end]
            if len(tokens) > 0:
                runtime_context[req_idx, -len(tokens) :] = tokens
        return torch.from_numpy(runtime_context)

    def prepare(
        self,
        model_kwargs: dict[str, Any],
        context: Qwen4ExpInputContext,
    ) -> None:
        """Populate graph-stable PLE inputs for one target or dummy forward."""
        if not 0 < context.num_reqs_padded <= self.max_num_reqs:
            raise ValueError(
                f"Qwen4Exp PLE num_reqs_padded must be in [1, {self.max_num_reqs}], got {context.num_reqs_padded}"
            )

        runtime_query_start_loc = model_kwargs.get("query_start_loc", context.query_start_loc)[
            : context.num_reqs_padded + 1
        ]
        runtime_context = model_kwargs.get("ngram_context")
        if runtime_context is None:
            runtime_context = self._build_runtime_context(context)
        elif (
            runtime_context.shape != (context.num_reqs_padded, self.context_len) or runtime_context.dtype != torch.int32
        ):
            raise ValueError(
                "Qwen4Exp PLE ngram_context must have shape "
                f"({context.num_reqs_padded}, {self.context_len}) and dtype "
                f"torch.int32, got shape={tuple(runtime_context.shape)}, "
                f"dtype={runtime_context.dtype}"
            )

        static_context, static_query_start_loc = pad_ple_graph_inputs(
            self._get_context_buffer(),
            self._get_query_start_loc_buffer(),
            runtime_context.to(device=self.device, dtype=torch.int32),
            runtime_query_start_loc.to(device=self.device, dtype=torch.int32),
            context.num_reqs_padded,
            self.eos_token_id,
        )
        model_kwargs["ngram_context"] = static_context
        model_kwargs["query_start_loc"] = static_query_start_loc


def build_qwen4_exp_input_adapter(
    model_config: Any,
    *,
    max_num_reqs: int,
    device: torch.device,
) -> Qwen4ExpPLEInputAdapter | None:
    """Build the Qwen4Exp PLE adapter when the model enables PLE."""
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None or not getattr(text_config, "ple_layer_ids", None):
        return None

    return Qwen4ExpPLEInputAdapter(
        text_config,
        max_num_reqs=max_num_reqs,
        device=device,
    )


__all__ = [
    "Qwen4ExpInputContext",
    "Qwen4ExpPLEInputAdapter",
    "build_qwen4_exp_input_adapter",
    "pad_ple_graph_inputs",
]
