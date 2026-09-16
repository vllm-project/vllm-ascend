#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch

from .transform import ActualInputSemanticOracle, InvalidTransformError


def make_unbounded_swa_case(q_length, ori_length):
    return {
        "params": {},
        "metadata_input": {"batch_size": 1},
        "op_input": {
            "q": torch.zeros((q_length, 1, 1)),
            "q_descale": None,
            "cu_seqlens_q": torch.tensor([0, q_length], dtype=torch.int32),
            "seqused_q": torch.tensor([q_length], dtype=torch.int32),
            "seqused_ori_kv": torch.tensor([ori_length], dtype=torch.int32),
            "seqused_cmp_kv": None,
            "cmp_kv": None,
            "sinks": torch.zeros((1,)),
            "ori_mask_mode": 4,
            "cmp_mask_mode": 0,
            "ori_win_left": -1,
            "ori_win_right": -1,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
        },
    }


def test_unbounded_mode4_rejects_token_split_that_shortens_kv():
    baseline = make_unbounded_swa_case(q_length=2, ori_length=4096)
    derived = make_unbounded_swa_case(q_length=1, ori_length=4095)

    assert ActualInputSemanticOracle._windows(baseline, 0, 0)["ori"] == (0, 4096)
    assert ActualInputSemanticOracle._windows(derived, 0, 0)["ori"] == (0, 4095)
    with pytest.raises(InvalidTransformError, match="different mask window"):
        ActualInputSemanticOracle.validate_mapped_tokens(
            baseline, derived, [(0, 0, 0, 0)]
        )
