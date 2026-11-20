# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops.gdn import _gdn_output_norm


def test_other_norm_implementation_retains_original_call_contract():
    x = torch.randn(8, 48, 128)
    z = torch.randn(8, 128, 128)[:, 80:]
    norm = Mock(side_effect=lambda x, z: x + z)
    with patch("vllm_ascend.ops.gdn.gdn_gated_norm", side_effect=AssertionError("not a GDN fast-path norm")):
        result = _gdn_output_norm(norm, x, z)
    norm.assert_called_once()
    args = norm.call_args.args
    assert args[0].shape == args[1].shape == (8 * 48, 128)
    torch.testing.assert_close(result, x + z)
