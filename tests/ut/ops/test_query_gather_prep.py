# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_query_prep_rejects_incompatible_fragments(invalid):
    query = torch.empty(1, 12, 512, dtype=torch.bfloat16)
    rope = torch.empty(
        2 if invalid == "shape" else 1,
        12,
        64,
        dtype=torch.float32 if invalid == "dtype" else query.dtype,
        device="meta" if invalid == "device" else "cpu",
    )
    with pytest.raises(RuntimeError, match="matching|share a dtype"):
        prep_query_head_major(query, rope)
