#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.metadata_builder import AscendAttentionMetadataBuilder310
from vllm_ascend.attention.attention_v1 import AscendAttentionState


class TestAscendAttentionMetadataBuilder310Causal(TestBase):
    def test_build_non_causal_uses_zero_compressed_mask(self):
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32, device="cpu")

        from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310

        builder.attn_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), 4096)

        common = MagicMock()
        common.num_reqs = 2
        common.causal = False
        common.query_start_loc = torch.tensor([0, 1, 3])
        common.query_start_loc_cpu = torch.tensor([0, 1, 3])
        common.seq_lens = torch.tensor([4, 6])
        common.attn_state = AscendAttentionState.ChunkedPrefill

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.ChunkedPrefill

        with patch.object(
            AscendAttentionMetadataBuilder310.__bases__[0], "build", return_value=base_metadata
        ), patch(
            "vllm_ascend._310p.attention.metadata_builder.is_compressed_mask_supported",
            return_value=True,
        ), patch(
            "vllm_ascend._310p.attention.metadata_builder.AttentionMaskBuilder310.get_compressed_non_causal_splitfuse_mask",
            return_value=torch.zeros(2048, 2048),
        ) as mock_non_causal:
            result = builder.build(0, common)
            mock_non_causal.assert_called_once_with(builder.device)
            self.assertIs(result.attn_mask, mock_non_causal.return_value)

    def test_build_attaches_host_seq_lens_cpu_for_prefill(self):
        # PrefillNoCache (non-splitfuse) returns early, but the host seq_lens must
        # still be attached so ATB flash attention gets host data even when the base
        # builder left seq_lens on device (parallel-drafting path).
        builder = object.__new__(AscendAttentionMetadataBuilder310)
        builder.device = torch.device("cpu")
        builder._query_lens_cpu_buffer = torch.zeros(8, dtype=torch.int32, device="cpu")

        from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310

        builder.attn_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), 4096)

        common = MagicMock()
        common.num_reqs = 2
        common._seq_lens_cpu = torch.tensor([4, 6, 99], dtype=torch.int32)
        common.attn_state = AscendAttentionState.PrefillNoCache

        base_metadata = MagicMock()
        base_metadata.attn_state = AscendAttentionState.PrefillNoCache

        with patch.object(AscendAttentionMetadataBuilder310.__bases__[0], "build", return_value=base_metadata):
            result = builder.build(0, common)

        torch.testing.assert_close(result.seq_lens_cpu, torch.tensor([4, 6], dtype=torch.int32))
