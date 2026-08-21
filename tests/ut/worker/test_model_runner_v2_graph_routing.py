from types import SimpleNamespace

import numpy as np
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_runner(configured_mode: CUDAGraphMode) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = 1
    runner.compilation_config = SimpleNamespace(cudagraph_mode=configured_mode)
    return runner


def test_full_decode_only_runtime_retains_graph_descriptor_batch_size():
    runner = _make_runner(CUDAGraphMode.FULL_DECODE_ONLY)
    query_start_loc = np.zeros(6, dtype=np.int32)
    query_start_loc[1] = 1

    query_start_loc, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_reqs=1,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == 4
    np.testing.assert_array_equal(query_start_loc[:5], [0, 1, 2, 3, 4])


def test_full_mode_runtime_keeps_unpadded_request_count():
    runner = _make_runner(CUDAGraphMode.FULL)
    query_start_loc = np.zeros(6, dtype=np.int32)
    query_start_loc[1] = 1

    _, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=1,
        num_reqs_padded=4,
        num_reqs=1,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == 1
