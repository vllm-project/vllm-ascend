# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


def test_unsupported_block_verification_uses_warning_once() -> None:
    dummy = torch.empty(0)
    warning = (
        "Block verification is not supported on NPU Model Runner V2 yet; "
        "falling back to standard token-by-token verification."
    )

    with (
        patch("vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils.logger.warning_once") as mock_warning_once,
        pytest.raises(NotImplementedError, match="Synthetic rejection sampling is not supported"),
    ):
        rejection_sample(
            dummy,
            None,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            num_speculative_steps=1,
            synthetic_conditional_rates=dummy,
            use_block_verification=True,
        )

    mock_warning_once.assert_called_once_with(warning)
