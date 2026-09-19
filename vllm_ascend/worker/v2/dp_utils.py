# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp as upstream_dispatch_cg_and_sync_dp

from vllm_ascend.utils import embedding_tp_enable, lmhead_tp_enable, mlp_tp_enable, oproj_tp_enable


def dispatch_cg_and_sync_dp(*args, **kwargs):
    batch_desc, num_tokens_across_dp = upstream_dispatch_cg_and_sync_dp(*args, **kwargs)
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
