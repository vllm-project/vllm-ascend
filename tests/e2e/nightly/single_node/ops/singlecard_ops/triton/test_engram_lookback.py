# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.e2e.pull_request.one_card.test_engram_lookback import check_lookback_window


def test_lookback_with_128k_history():
    width = 128 * 1024
    check_lookback_window(
        depth=3,
        width=width,
        row_padding=0,
        first_slots=[2, 3, 4, 5, 6, 7, 0, 1],
        initial_counts=[0, 1, 2, 3, 4, width, width, width],
    )
