from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@pytest.mark.parametrize(
    ("configured_mode", "expected_num_reqs_padded", "expected_query_start_loc"),
    [
        (CUDAGraphMode.FULL_DECODE_ONLY, 4, [0, 1, 2, 3, 4]),
        (CUDAGraphMode.FULL_AND_PIECEWISE, 4, [0, 1, 2, 3, 4]),
        (CUDAGraphMode.FULL, 3, [0, 1, 2, 4, 0]),
    ],
)
def test_fullgraph_runtime_preserves_configured_batch_routing(
    configured_mode: CUDAGraphMode,
    expected_num_reqs_padded: int,
    expected_query_start_loc: list[int],
) -> None:
    runner = cast(
        NPUModelRunner,
        SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=configured_mode),
            decode_query_len=1,
        ),
    )
    query_start_loc = np.array([0, 1, 2, 0, 0], dtype=np.int32)

    result, num_reqs_padded = NPUModelRunner._pad_query_start_loc_for_fia(
        runner,
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_reqs=2,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == expected_num_reqs_padded
    np.testing.assert_array_equal(result, expected_query_start_loc)
