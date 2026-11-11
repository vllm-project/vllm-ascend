# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import torch
from torch import nn

from vllm_ascend.worker.v1.spec_decode.utils import compact_mtp_topk_indices


def test_compaction_uses_predictor_hook():
    model = nn.Module()
    model.compact_topk_indices = Mock()
    copy_sfa_attention = Mock()
    indices = torch.tensor([1, 5], dtype=torch.int32)
    compact_mtp_topk_indices(model, indices, lim_topk_compactors=[copy_sfa_attention])
    model.compact_topk_indices.assert_called_once_with(indices)
    copy_sfa_attention.compact_lim_topk_metadata.assert_called_once_with(indices)
