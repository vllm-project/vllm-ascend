#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, writing
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class TestAscendSpecDecodeBaseProposer310(TestBase):
    def test_run_merged_draft_sets_rope_flag_before_call(self):
        flag_states = []

        def mock_original(
            self,
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill=None,
        ):
            flag_states.append(AscendRotaryEmbedding310._is_drafting_update_enabled)
            return torch.zeros(num_tokens, dtype=torch.long)

        with patch.object(AscendSpecDecodeBaseProposer, "_run_merged_draft", mock_original):
            with patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft", mock_original):
                proposer = object.__new__(AscendSpecDecodeBaseProposer310)
                result = proposer._run_merged_draft(
                    num_input_tokens=4,
                    batch_size=2,
                    token_indices_to_sample=torch.tensor([0, 1]),
                    target_positions=torch.tensor([0, 1, 2, 3]),
                    inputs_embeds=torch.zeros(4, 128),
                    multi_steps_attn_metadata=None,
                    num_tokens=4,
                )

        self.assertEqual(len(flag_states), 1)
        self.assertTrue(flag_states[0])
        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_run_merged_draft_restores_rope_flag_after_exception(self):
        def mock_original(*args, **kwargs):
            raise RuntimeError("Test exception")

        with patch.object(AscendSpecDecodeBaseProposer, "_run_merged_draft", mock_original):
            with patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310._original_run_merged_draft", mock_original):
                proposer = object.__new__(AscendSpecDecodeBaseProposer310)
                with self.assertRaises(RuntimeError):
                    proposer._run_merged_draft(
                        num_input_tokens=4,
                        batch_size=2,
                        token_indices_to_sample=torch.tensor([0, 1]),
                        target_positions=torch.tensor([0, 1, 2, 3]),
                        inputs_embeds=torch.zeros(4, 128),
                        multi_steps_attn_metadata=None,
                        num_tokens=4,
                    )

        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)
