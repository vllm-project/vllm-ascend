# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Lean UTs for 310P MRv2 MTP (rejection offset, capture-safe draft step, RoPE flag).

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.worker.v2.spec_decode.aclgraph import AutoRegressiveAclGraphManager310
from vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator import AscendMTPSpeculator310
from vllm_ascend._310p.worker.v2.spec_utils import (
    greedy_rejection_sample_cpu,
    probabilistic_rejection_sample_cpu,
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

        sampled, num_sampled, _, _ = greedy_rejection_sample_cpu(
            target_logits, draft_sampled, cu_num_logits, num_speculative_steps=1
        )

        self.assertEqual(num_sampled.tolist(), [2])
        self.assertEqual(sampled[0, :2].tolist(), [1, 2])

    def test_probabilistic_rejection_accepts_when_u_below_p_draft(self):
        """MRV1 IS_NGRAM: accept draft iff u < p(draft); forced u=0 always accepts."""
        # After softmax, token1 dominates first row → p(draft=1) high.
        target_logits = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        )
        draft_sampled = torch.tensor([9, 1], dtype=torch.int32)  # draft at +1 is 1
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32)
        temperature_np = np.array([0.8], dtype=np.float32)
        idx_mapping_np = np.array([0], dtype=np.int32)

        with patch(
            "vllm_ascend._310p.worker.v2.spec_utils._draw_uniform_cpu",
            side_effect=[0.0, 0.99],  # accept draft; bonus near end of CDF → token 2
        ):
            sampled, num_sampled, _, _ = probabilistic_rejection_sample_cpu(
                target_logits,
                draft_sampled,
                cu_num_logits,
                num_speculative_steps=1,
                temperature_np=temperature_np,
                idx_mapping_np=idx_mapping_np,
                source_generators={},
            )

        self.assertEqual(int(num_sampled[0].item()), 2)
        self.assertEqual(int(sampled[0, 0].item()), 1)  # accepted draft
        self.assertEqual(int(sampled[0, 1].item()), 2)  # bonus from last logit row

    def test_probabilistic_rejection_recovers_when_u_rejects_draft(self):
        """u >= p(draft) → recovered token from residual (draft mass zeroed)."""
        target_logits = torch.tensor(
            [
                [0.0, 5.0, 4.0],  # draft=1; residual keeps token 2
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        draft_sampled = torch.tensor([0, 1], dtype=torch.int32)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int32)
        temperature_np = np.array([1.0], dtype=np.float32)
        idx_mapping_np = np.array([0], dtype=np.int32)

        with patch(
            "vllm_ascend._310p.worker.v2.spec_utils._draw_uniform_cpu",
            return_value=0.999,  # always reject / pick high CDF
        ):
            sampled, num_sampled, _, _ = probabilistic_rejection_sample_cpu(
                target_logits,
                draft_sampled,
                cu_num_logits,
                num_speculative_steps=1,
                temperature_np=temperature_np,
                idx_mapping_np=idx_mapping_np,
                source_generators={},
            )

        self.assertEqual(int(num_sampled[0].item()), 1)
        # Recovered must not be the rejected draft token 1.
        self.assertNotEqual(int(sampled[0, 0].item()), 1)

    def test_prepare_decode_inputs_advances_sample_src_positions(self):
        """Upstream K>1 path passes sample_src_positions; CPU fallback must accept it."""
        from vllm_ascend._310p.worker.v2.spec_utils import prepare_decode_inputs_cpu

        num_reqs = 2
        draft_tokens = torch.tensor([7, 8], dtype=torch.int32)
        target_seq_lens = torch.tensor([10, 12], dtype=torch.int32)
        num_rejected = torch.tensor([0, 1], dtype=torch.int32)
        sample_src_positions = torch.tensor([5, 6], dtype=torch.int64)
        input_buffers = SimpleNamespace(
            input_ids=torch.zeros(num_reqs, dtype=torch.int32),
            positions=torch.tensor([4, 5], dtype=torch.int64),
            query_start_loc=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.zeros(num_reqs, dtype=torch.int32),
        )

        prepare_decode_inputs_cpu(
            draft_tokens,
            target_seq_lens,
            num_rejected,
            input_buffers,
            sample_src_positions,
            max_model_len=128,
            max_num_reqs=num_reqs,
            advance_draft_positions=True,
        )

        self.assertEqual(input_buffers.input_ids.tolist(), [7, 8])
        self.assertEqual(sample_src_positions.tolist(), [6, 7])
        self.assertEqual(input_buffers.positions.tolist(), [5, 6])
        self.assertEqual(input_buffers.seq_lens.tolist(), [11, 12])

    def test_update_draft_inputs_uses_host_step_under_capture(self):
        num_reqs = 2
        draft_tokens = torch.tensor([11, 22], dtype=torch.int32)
        current_draft_step = torch.tensor([99], dtype=torch.int64)  # must not .item() under capture
        hidden_states = torch.randn(num_reqs, 4)
        output_draft_tokens = torch.full((num_reqs, 2), -1, dtype=torch.int32)
        next_input_hidden_states = torch.zeros(num_reqs, 4)
        sample_src_positions = torch.tensor([9, 10], dtype=torch.int64)
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
                sample_src_positions=sample_src_positions,
                num_reqs=num_reqs,
                max_model_len=128,
                num_speculative_steps=2,
                advance_draft_positions=True,
            )

        self.assertEqual(output_draft_tokens[:, 0].tolist(), [11, 22])
        self.assertEqual(input_buffers.positions.tolist(), [4, 6])
        self.assertEqual(sample_src_positions.tolist(), [10, 11])

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

    def test_draft_config_preserves_profiling_chunk(self):
        """The 310P draft keeps the target CPP setting while using PP=1."""

        def fake_replace(config, **changes):
            values = vars(config).copy()
            values.update(changes)
            return SimpleNamespace(**values)

        target_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
            additional_config={"scheduler_config": {"profiling_chunk_config": {"enabled": True}}},
            model_config=SimpleNamespace(model="/target"),
            quant_config=None,
        )
        speculator = object.__new__(AscendMTPSpeculator310)
        speculator.vllm_config = target_config
        speculator.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_overrides=None, model="/draft")
        )

        with (
            patch(
                "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator.replace",
                side_effect=fake_replace,
            ),
            patch(
                "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.replace",
                side_effect=fake_replace,
            ),
        ):
            draft_config = speculator._create_draft_vllm_config()

        self.assertTrue(draft_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"])
        self.assertTrue(target_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"])
        self.assertEqual(draft_config.parallel_config.pipeline_parallel_size, 1)

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
