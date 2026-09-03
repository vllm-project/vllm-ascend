from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker.npu_input_batch import NPUInputBatch


@pytest.mark.parametrize(
    ("block_sizes", "kernel_block_sizes", "max_num_blocks"),
    [
        ([384, 384], [128, 384], [8, 4]),
        ([128, 384], [128, 384], [16, 4]),
        ([384, 384], [0, 384], [8, 4]),
    ],
)
def test_input_batch_forwards_flat_kernel_sizes_per_group(
    block_sizes,
    kernel_block_sizes,
    max_num_blocks,
):
    with patch("vllm_ascend.worker.npu_input_batch.MultiGroupBlockTable") as block_table_cls:
        NPUInputBatch(
            max_num_reqs=2,
            max_model_len=1024,
            max_num_batched_tokens=64,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=128,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks_per_req=max_num_blocks,
            num_speculative_tokens=3,
            cp_kv_cache_interleave_size=2,
        )

    block_table_cls.assert_called_once_with(
        max_num_reqs=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=block_sizes,
        max_num_blocks=max_num_blocks,
        num_speculative_tokens=3,
        kernel_sizes=kernel_block_sizes,
        cp_kv_cache_interleave_size=2,
        kv_cache_groups=None,
    )
