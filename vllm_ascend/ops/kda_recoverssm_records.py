# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords

from vllm_ascend.ops.triton.kda.recoverssm import allocate_kda_verify_caches


@dataclass(frozen=True)
class AscendKDARecords(KDARecoverSSMRecords):
    decay_cache: torch.Tensor | None
    normalized_key_cache: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        max_num_reqs: int,
        num_heads: int,
        spec_query_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "AscendKDARecords":
        records = KDARecoverSSMRecords.allocate(max_num_reqs, num_heads, spec_query_len, head_dim, dtype, device)
        decay, normalized_key = allocate_kda_verify_caches(records.key_gate)
        return cls(records.correction, records.key_gate, decay, normalized_key)
