# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
    _make_spec_pp_comm_batch,
)


@dataclass
class _InputBatch:
    num_computed_tokens_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    prefill_len_np: np.ndarray
    max_seq_len_np: np.ndarray


def test_spec_pp_comm_does_not_skip_rank_local_finishing_request():
    input_batch = _InputBatch(
        num_computed_tokens_np=np.array([56, 0], dtype=np.int32),
        num_scheduled_tokens=np.array([8, 25], dtype=np.int32),
        prefill_len_np=np.array([25, 50], dtype=np.int32),
        max_seq_len_np=np.array([57, 100], dtype=np.int32),
    )

    assert compute_need_sampled_mask(input_batch) is None

    comm_batch = _make_spec_pp_comm_batch(input_batch)

    np.testing.assert_array_equal(
        compute_need_sampled_mask(comm_batch),
        np.array([True, False]),
    )
    np.testing.assert_array_equal(input_batch.max_seq_len_np, [57, 100])
