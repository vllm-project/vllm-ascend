# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


@pytest.mark.parametrize("shape", [(0, 1, 8), (2, 1, 0), (0, 0)])
def test_empty_remap_preserves_input_without_launching_kernels(shape):
    indices = torch.empty(shape, dtype=torch.int32)

    result = remap_sparse_indices_triton(indices, dcp_size=2, dcp_rank=0, interleave_size=2)

    assert result is indices
