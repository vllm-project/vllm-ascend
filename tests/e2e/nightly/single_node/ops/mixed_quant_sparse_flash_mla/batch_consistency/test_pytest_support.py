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

from batch_consistency.pytest_support import format_consistency_summary


def test_format_consistency_summary_lists_all_outcomes():
    report = {
        "baseline_precision": {
            "pass": False,
            "output": {"fulfill_percent": 74.32098388671875},
        },
        "relations": [
            {
                "case": "mode1_reorder_B2",
                "pass": False,
                "batch_consistency": {"pass": True},
                "precision": {
                    "pass": False,
                    "output": {"fulfill_percent": 74.32098388671875},
                },
            },
            {
                "case": "mode2_split_g0_B1",
                "pass": True,
                "batch_consistency": {"pass": True},
                "precision": {
                    "pass": True,
                    "output": {"fulfill_percent": 100.0},
                },
            },
        ],
        "skipped_modes": {
            "token-split": "NOT_APPLICABLE: mask window changed",
        },
        "pass": False,
    }

    summary = format_consistency_summary(report)

    assert "baseline" in summary and "FAILED (74.32%)" in summary
    assert "mode1_reorder_B2" in summary and "PASS" in summary
    assert "mode2_split_g0_B1" in summary and "PASS (100.00%)" in summary
    assert "token-split" in summary and "SKIPPED" in summary
    assert "NOT_APPLICABLE" not in summary
    assert "OVERALL: FAILED" in summary
