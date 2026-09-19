# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

from vllm.config import CUDAGraphMode

from vllm_ascend.utils import embedding_tp_enable, lmhead_tp_enable, mlp_tp_enable, oproj_tp_enable


def pad_eager_batch_for_finegrained_tp(batch_desc, num_tokens_across_dp):
    if (
        batch_desc.cg_mode == CUDAGraphMode.NONE
        and num_tokens_across_dp is not None
        and (oproj_tp_enable() or embedding_tp_enable() or mlp_tp_enable() or lmhead_tp_enable())
    ):
        # Like MRV1, pad using metadata already synchronized by the CPU DP
        # group. numpy() shares that CPU storage; no device transfer, device
        # synchronization, or additional collective is needed here.
        assert num_tokens_across_dp.device.type == "cpu"
        token_counts = num_tokens_across_dp.numpy()
        num_tokens_padded = int(token_counts.max())
        token_counts.fill(num_tokens_padded)
        batch_desc = replace(batch_desc, num_tokens=num_tokens_padded)
    return batch_desc, num_tokens_across_dp
