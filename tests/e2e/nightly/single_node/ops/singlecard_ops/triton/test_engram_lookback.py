# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.e2e.pull_request.one_card.test_engram_lookback import check_lookback_window


def test_lookback_with_128k_history():
    check_lookback_window(capacity=8, depth=3, width=128 * 1024, row_padding=0)
