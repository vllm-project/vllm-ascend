# SPDX-License-Identifier: Apache-2.0
"""The CatCCOS fused path retains MC2's TP split and output restoration."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMC2


def test_tp4_nonmultiple_token_count_restores_original_order():
    global_tokens = 7
    tp_size = 4
    hidden_states = torch.arange(global_tokens * 8, dtype=torch.float32).view(global_tokens, 8)
    router_logits = torch.zeros(global_tokens, 2)
    context = SimpleNamespace(
        mc2_mask=torch.tensor([True] * global_tokens + [False]),
        padded_num_tokens=8,
    )
    moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=object()))

    with (
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config") as get_config,
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank", return_value=2),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context),
        patch("torch.distributed.all_gather") as all_gather,
    ):
        get_config.return_value.multistream_overlap_gate = False
        method = PrepareAndFinalizeWithMC2(moe_config)
        prepared = method.prepare(hidden_states, router_logits)
        torch.testing.assert_close(prepared.hidden_states, hidden_states[4:6])
        assert prepared.mc2_mask.tolist() == [True, True]

        padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, 1))

        def gather(shards, _local, _group):
            for rank, shard in enumerate(shards):
                shard.copy_(padded[rank * 2 : (rank + 1) * 2])

        all_gather.side_effect = gather
        restored = method.finalize(
            prepared.hidden_states,
            reduce_results=False,
            padded_hidden_states_shape=prepared.padded_hidden_states_shape,
        )

    all_gather.assert_called_once()
    torch.testing.assert_close(restored, hidden_states)
