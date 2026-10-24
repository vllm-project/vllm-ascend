# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


@dataclass
class Ascend310PInputBatch(AscendInputBatch):
    """310P host metadata retained between preparation and sampling."""

    block_tables_np: tuple[np.ndarray, ...] | None = None
    input_ids_cpu: torch.Tensor | None = None
    logits_indices_np: np.ndarray | None = None
    # CPU mirrors of expanded mapping / local pos (avoid NPU→CPU sync in sampler).
    expanded_idx_mapping_np: np.ndarray | None = None
    expanded_local_pos_np: np.ndarray | None = None
