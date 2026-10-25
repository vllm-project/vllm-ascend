# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for UNO's draft LoRA token mappings."""

import pytest

from vllm_ascend.worker.v2.spec_decode.uno.lora import (
    base_prompt_mapping,
    draft_lora_mapping,
)

UNO_ID = 1_000_003


def test_seed_row_is_base_and_noise_rows_use_adapter():
    mapping = draft_lora_mapping(num_input_tokens=8, batch_size=2, k=4, uno_lora_id=UNO_ID)

    assert mapping == (0, UNO_ID, UNO_ID, UNO_ID, 0, UNO_ID, UNO_ID, UNO_ID)


def test_padding_rows_stay_on_base():
    # Extra rows beyond batch_size * k are graph padding and must not route
    # through UNO's adapter.
    mapping = draft_lora_mapping(num_input_tokens=10, batch_size=2, k=4, uno_lora_id=UNO_ID)

    assert mapping[8:] == (0, 0)
    assert len(mapping) == 10


def test_single_row_per_request_has_no_adapter_rows():
    mapping = draft_lora_mapping(num_input_tokens=3, batch_size=3, k=1, uno_lora_id=UNO_ID)

    assert mapping == (0, 0, 0)


def test_lora_id_is_coerced_to_int():
    mapping = draft_lora_mapping(num_input_tokens=2, batch_size=1, k=2, uno_lora_id=UNO_ID)

    assert all(isinstance(v, int) for v in mapping)


def test_returns_tuple_for_lora_mapping():
    assert isinstance(draft_lora_mapping(4, 1, 4, UNO_ID), tuple)


@pytest.mark.parametrize(
    "num_input_tokens,batch_size,k",
    [
        (-1, 1, 1),
        (4, 0, 1),
        (4, 1, 0),
        # buffer shorter than the real query rows
        (3, 2, 2),
    ],
)
def test_invalid_dimensions_raise(num_input_tokens, batch_size, k):
    with pytest.raises(ValueError):
        draft_lora_mapping(num_input_tokens, batch_size, k, UNO_ID)


def test_base_prompt_mapping_is_all_base():
    assert base_prompt_mapping(5) == (0,) * 5
    assert base_prompt_mapping(0) == ()


def test_base_prompt_mapping_rejects_negative():
    with pytest.raises(ValueError):
        base_prompt_mapping(-1)
