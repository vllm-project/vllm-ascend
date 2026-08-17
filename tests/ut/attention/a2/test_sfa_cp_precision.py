# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl


def test_sfa_dcp_torch_merge_handles_invalid_lse() -> None:
    output = torch.tensor(
        [
            [[[1.0]], [[3.0]]],
            [[[5.0]], [[7.0]]],
        ]
    )
    lse = torch.tensor(
        [
            [[0.0], [float("-inf")]],
            [[0.0], [0.0]],
        ]
    )

    merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=2)

    torch.testing.assert_close(merged, torch.tensor([[[3.0], [7.0]]]))

    dsa_merged = AscendSFADCPImpl._merge_dcp_outputs_with_torch(output, lse, token_dim=1)
    torch.testing.assert_close(dsa_merged, torch.tensor([[[3.0]], [[7.0]]]))
