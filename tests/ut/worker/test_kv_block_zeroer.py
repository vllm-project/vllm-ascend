# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.worker.utils import AscendKVBlockZeroer


def test_physical_block_tensors_are_grouped_by_page_size():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    small = torch.empty(4 * 32, dtype=torch.int8)
    large = torch.empty(4 * 64, dtype=torch.int8)

    zeroer.init_meta(
        attn_groups_iter=[],
        kernel_block_sizes=[],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={},
        physical_block_tensors=[small, small, large],
        num_blocks=4,
    )

    assert [(page_size_el, n_segs) for _, page_size_el, _, n_segs in zeroer._metas] == [
        (8, 1),
        (16, 1),
    ]
