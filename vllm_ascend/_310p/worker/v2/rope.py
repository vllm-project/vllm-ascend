# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

from vllm_ascend._310p.worker.v2.states import Ascend310PStagedWriteTensor


class Ascend310PRopeState:
    """Triton-free MRoPE state for 310P (Qwen3-VL / Qwen3.5)."""

    def __init__(
        self,
        num_dims: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ) -> None:
        self.num_dims = num_dims
        self.max_model_len = max_model_len
        self.device = device
        self.prefill_positions = Ascend310PStagedWriteTensor(
            (max_num_reqs * num_dims, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.prefill_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.positions_cpu = torch.zeros((num_dims, max_num_tokens + 1), dtype=torch.int64, device="cpu")
        self.positions = torch.zeros((num_dims, max_num_tokens + 1), dtype=torch.int64, device=device)

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        mrope_model = cast(SupportsMRoPE, model)
        # Qwen3-VL / Qwen3.5 return ``(Tensor[num_dims, seq], delta)``.
        prefill_positions, delta = mrope_model.get_mrope_input_positions(prefill_token_ids, mm_features)
        self.prefill_delta.np[req_idx] = delta

        for dim in range(self.num_dims):
            self.prefill_positions.stage_write(
                self.num_dims * req_idx + dim,
                0,
                prefill_positions[dim].tolist(),
            )

    def apply_staged_writes(self) -> None:
        self.prefill_positions.apply_write()
        self.prefill_delta.copy_to_uva()

    def prepare_positions_cpu(
        self,
        idx_mapping_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        prefill_lens_np: np.ndarray,
        num_computed_tokens_np: np.ndarray,
        num_tokens_after_padding: int,
    ) -> None:
        self.positions_cpu[:, :num_tokens_after_padding].zero_()
        for batch_idx, req_idx in enumerate(idx_mapping_np):
            query_start = int(query_start_loc_np[batch_idx])
            query_end = int(query_start_loc_np[batch_idx + 1])
            num_computed = int(num_computed_tokens_np[req_idx])
            query_len = query_end - query_start
            if num_computed < int(prefill_lens_np[req_idx]):
                row_start = self.num_dims * int(req_idx)
                positions = self.prefill_positions.cpu[
                    row_start : row_start + self.num_dims,
                    num_computed : num_computed + query_len,
                ]
                self.positions_cpu[:, query_start:query_end].copy_(positions)
            else:
                delta = int(self.prefill_delta.np[req_idx])
                decode_positions = torch.arange(
                    num_computed + delta,
                    num_computed + delta + query_len,
                    dtype=torch.int64,
                )
                self.positions_cpu[:, query_start:query_end] = decode_positions

        self.positions[:, :num_tokens_after_padding].copy_(
            self.positions_cpu[:, :num_tokens_after_padding], non_blocking=True
        )

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]


def get_310p_rope_state(
    model_config: ModelConfig,
    model: nn.Module,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    device: torch.device,
) -> Ascend310PRopeState | None:
    # 310P Qwen3-VL / Qwen3.5 use MRoPE only; XD-RoPE is out of scope.
    if model_config.uses_mrope:
        assert isinstance(model, SupportsMRoPE)
        return Ascend310PRopeState(3, max_num_reqs, max_num_tokens, max_model_len, device)
    return None
