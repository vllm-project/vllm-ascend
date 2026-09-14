# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Lean UTs for 310P MRv2 MTP (rejection offset, capture-safe draft step, RoPE flag).

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.worker.v2.spec_decode.aclgraph import AutoRegressiveAclGraphManager310
from vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator import AscendMTPSpeculator310
from vllm_ascend._310p.worker.v2.spec_utils import (
    greedy_rejection_sample_cpu,
    set_draft_step_host,
    update_draft_inputs_cpu,
)


class TestMRv2Mtp310(TestBase):
    def test_greedy_rejection_uses_logit_idx_plus_one(self):
        # draft_sampled[logit_idx+1] must match argmax(logits[logit_idx]) to accept.
        target_logits = torch.tensor(
            [
                [0.1, 0.9, 0.0],  # predicts token 1
                [0.0, 0.2, 0.8],  # bonus → token 2
            ],
            dtype=torch.float32,
        )
        draft_sampled = torch.tensor([7, 1], dtype=torch.int32)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32)

        sampled, num_sampled = greedy_rejection_sample_cpu(
            target_logits, draft_sampled, cu_num_logits, num_speculative_steps=1
        )

        self.assertEqual(num_sampled.tolist(), [2])
        self.assertEqual(sampled[0, :2].tolist(), [1, 2])

    def test_update_draft_inputs_uses_host_step_under_capture(self):
        num_reqs = 2
        draft_tokens = torch.tensor([11, 22], dtype=torch.int32)
        current_draft_step = torch.tensor([99], dtype=torch.int64)  # must not .item() under capture
        hidden_states = torch.randn(num_reqs, 4)
        output_draft_tokens = torch.full((num_reqs, 2), -1, dtype=torch.int32)
        next_input_hidden_states = torch.zeros(num_reqs, 4)
        input_buffers = SimpleNamespace(
            input_ids=torch.zeros(num_reqs, dtype=torch.int32),
            positions=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        )
        set_draft_step_host(0)

        with patch("torch.npu.is_current_stream_capturing", return_value=True):
            update_draft_inputs_cpu(
                draft_tokens=draft_tokens,
                current_draft_step=current_draft_step,
                hidden_states=hidden_states,
                output_draft_tokens=output_draft_tokens,
                next_input_hidden_states=next_input_hidden_states,
                input_buffers=input_buffers,
                num_reqs=num_reqs,
                max_model_len=128,
                num_speculative_steps=2,
                advance_draft_positions=True,
            )

        self.assertEqual(output_draft_tokens[:, 0].tolist(), [11, 22])
        self.assertEqual(input_buffers.positions.tolist(), [4, 6])

    def test_run_model_sets_rope_flag(self):
        flag_states: list[bool] = []

        def mock_parent_run(self, *args, **kwargs):
            del self, args, kwargs
            flag_states.append(AscendRotaryEmbedding310._is_drafting_update_enabled)
            return torch.zeros(1), torch.zeros(1)

        speculator = object.__new__(AscendMTPSpeculator310)
        with patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.AscendAutoRegressiveSpeculator._run_model",
            mock_parent_run,
        ):
            AscendMTPSpeculator310._run_model(
                speculator,
                num_tokens=1,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

        self.assertEqual(flag_states, [True])
        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_decode_capture_routes_to_per_step(self):
        manager = object.__new__(AutoRegressiveAclGraphManager310)
        manager.is_draft_model_prefill = False
        called = {"per_step": False}

        def fake_per_step(*args, **kwargs):
            del args, kwargs
            called["per_step"] = True

        with patch.object(AutoRegressiveAclGraphManager310, "_capture_decode_per_step", fake_per_step):
            AutoRegressiveAclGraphManager310.capture(
                manager,
                forward_fn=lambda: None,
                model_state=object(),
                input_buffers=object(),
                block_tables=object(),
                attn_groups=[],
                kv_cache_config=object(),
            )

        self.assertTrue(called["per_step"])
