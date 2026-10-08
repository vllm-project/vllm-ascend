# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm_ascend.worker.v2.prompt_tail import prompt_tail_request_mask


def test_kv_recv_prompt_tail_is_decode_shaped():
    computed = np.array([127, 64], dtype=np.int32)
    prefill_len = np.array([128, 64], dtype=np.int32)
    # Second request has already finished its prompt, so it is not prefilling.
    is_prefilling = computed < prefill_len
    mask = prompt_tail_request_mask(
        is_prefilling,
        computed,
        np.array([1, 1], dtype=np.int32),
        prefill_len,
        decode_query_len=1,
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_mid_prefill_and_first_token_stay_prefill():
    is_prefilling = np.array([True, True, True])
    mask = prompt_tail_request_mask(
        is_prefilling,
        np.array([64, 0, 127], dtype=np.int32),
        np.array([8, 1, 2], dtype=np.int32),
        np.array([128, 128, 128], dtype=np.int32),
        decode_query_len=1,
    )
    np.testing.assert_array_equal(mask, [False, False, False])


def test_multi_token_decode_query_finishes_prompt():
    mask = prompt_tail_request_mask(
        np.array([True, False]),
        np.array([126, 64], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        np.array([128, 64], dtype=np.int32),
        decode_query_len=2,
    )
    np.testing.assert_array_equal(mask, [True, False])
