# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.fused_moe.token_dispatcher import remap_global_expert_ids_310


def test_remap_global_expert_ids_310_masks_non_local_routes() -> None:
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[4:8] = torch.arange(4, dtype=torch.int32)
    topk_ids = torch.tensor([[0, 4, 7], [5, 2, 6]], dtype=torch.int32)

    local_ids, local_mask = remap_global_expert_ids_310(topk_ids, expert_map)

    torch.testing.assert_close(
        local_ids,
        torch.tensor([[0, 0, 3], [1, 0, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        local_mask,
        torch.tensor([[False, True, True], [True, False, True]]),
    )
