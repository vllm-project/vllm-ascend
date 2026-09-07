# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import patch

import torch

from vllm_ascend.sample.penalties import apply_all_penalties


@patch("vllm_ascend.sample.penalties.is_pin_memory_available", return_value=False)
@patch("vllm_ascend.sample.penalties.apply_penalties_triton", side_effect=lambda logits, *args, **kwargs: logits)
def test_apply_all_penalties(mock_triton, _mock_pin):
    logits = torch.zeros(2, 8)
    prompt_token_ids = torch.zeros(2, 2, dtype=torch.int64)
    zeros = torch.zeros(2)
    out = apply_all_penalties(logits, prompt_token_ids, zeros, zeros, zeros, [[1, -1], [2]])
    assert out is logits
    output_tokens = mock_triton.call_args[0][2]
    assert output_tokens[0, 1].item() == 8
