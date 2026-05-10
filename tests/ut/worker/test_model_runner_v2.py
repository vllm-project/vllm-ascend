import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType, get_mrv2_in_profile_run
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class TestNPUModelRunnerV2(unittest.TestCase):
    @staticmethod
    def _make_runner(max_num_tokens: int = 16):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.max_num_tokens = max_num_tokens
        runner.vllm_config = MagicMock()
        return runner

    @staticmethod
    def _make_runner_for_execute_model(
        *,
        dp_size: int = 1,
        dp_rank: int = 0,
        is_encoder_decoder: bool = False,
    ):
        runner = TestNPUModelRunnerV2._make_runner()
        runner.finish_requests = MagicMock()
        runner.free_states = MagicMock()
        runner.add_requests = MagicMock()
        runner.update_requests = MagicMock()
        runner.block_tables = MagicMock()
        runner.kv_connector = MagicMock()
        runner.cudagraph_manager = MagicMock()
        runner.dp_size = dp_size
        runner.dp_rank = dp_rank
        runner.is_encoder_decoder = is_encoder_decoder
        return runner

    def test_profile_run_marks_only_mc2_warmup_dummy_run(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=8),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(8, True), (16, True)])
        self.assertFalse(get_mrv2_in_profile_run())

    def test_profile_run_keeps_normal_dummy_run_outside_profile_override(self):
        runner = self._make_runner(max_num_tokens=16)
        observed_runs: list[tuple[int, bool]] = []

        def fake_base_dummy_run(self, num_tokens, *args, **kwargs):
            observed_runs.append((num_tokens, get_mrv2_in_profile_run()))
            return None, None

        def fake_base_profile_run(self):
            self._dummy_run(self.max_num_tokens, skip_attn=True)

        with (
            patch.object(GPUModelRunner, "_dummy_run", new=fake_base_dummy_run),
            patch.object(GPUModelRunner, "profile_run", new=fake_base_profile_run),
            patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=32),
            patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        ):
            runner.profile_run()

        self.assertEqual(observed_runs, [(16, True)])

    def test_execute_model_returns_no_forward_when_dp_sync_yields_zero_tokens(self):
        runner = self._make_runner_for_execute_model(
            dp_size=2,
            dp_rank=0,
            is_encoder_decoder=True,
        )

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 3
        scheduler_output.num_scheduled_tokens = {"req-1": 3}
        scheduler_output.scheduled_encoder_inputs = {"req-1": object()}

        zero_tok_batch_desc = MagicMock()
        zero_tok_batch_desc.num_tokens = 0

        with (
            patch("vllm_ascend.worker.v2.model_runner.get_uniform_token_count", return_value=3),
            patch(
                "vllm_ascend.worker.v2.model_runner.dispatch_cg_and_sync_dp",
                return_value=(zero_tok_batch_desc, 6),
            ) as mock_dispatch,
        ):
            output = runner.execute_model(scheduler_output=scheduler_output, is_profile=True)

        runner.kv_connector.no_forward.assert_called_once_with(scheduler_output)
        self.assertIs(output, runner.kv_connector.no_forward.return_value)
        mock_dispatch.assert_called_once_with(
            runner.cudagraph_manager,
            1,
            3,
            3,
            2,
            0,
            need_eager=True,
        )

    def test_execute_model_runs_piecewise_path_and_persists_execute_state(self):
        runner = self._make_runner_for_execute_model(dp_size=2, dp_rank=1)
        runner.model_state = MagicMock()
        runner.model_state.prepare_inputs.return_value = {}
        runner.attn_groups = MagicMock()
        runner.kv_cache_config = MagicMock()
        runner.req_states = MagicMock()
        runner.lora_config = None
        runner.supports_mm_inputs = False
        runner.is_first_pp_rank = True
        runner.is_last_pp_rank = True
        runner.use_aux_hidden_state_outputs = False

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 4
        scheduler_output.num_scheduled_tokens = {"req-1": 1, "req-2": 3}
        scheduler_output.scheduled_encoder_inputs = {}

        batch_desc = MagicMock()
        batch_desc.num_tokens = 8
        batch_desc.num_reqs = 2
        batch_desc.cg_mode = CUDAGraphMode.PIECEWISE

        input_batch = MagicMock()
        input_batch.num_tokens_after_padding = 8
        runner.prepare_inputs = MagicMock(return_value=input_batch)
        runner.prepare_attn = MagicMock(return_value=(MagicMock(), MagicMock()))

        hidden_states = torch.zeros(8, 4)
        runner.model = MagicMock(return_value=hidden_states)

        num_tokens_across_dp = torch.tensor([4, 4])
        captured_state: list[dict] = []

        def record_execute_state(**kwargs):
            captured_state.append(kwargs)
            return MagicMock()

        with (
            patch("vllm_ascend.worker.v2.model_runner.get_uniform_token_count", return_value=4),
            patch(
                "vllm_ascend.worker.v2.model_runner.dispatch_cg_and_sync_dp",
                return_value=(batch_desc, num_tokens_across_dp),
            ) as mock_dispatch,
            patch(
                "vllm_ascend.worker.v2.model_runner.build_slot_mappings_by_layer",
                return_value="slot-by-layer",
            ),
            patch("vllm_ascend.worker.v2.model_runner.set_ascend_forward_context") as mock_ctx,
            patch(
                "vllm_ascend.worker.v2.model_runner.ExecuteModelState",
                side_effect=record_execute_state,
            ),
        ):
            output = runner.execute_model(scheduler_output=scheduler_output)

        self.assertIsNone(output)
        runner.kv_connector.pre_forward.assert_called_once_with(scheduler_output)
        runner.kv_connector.post_forward.assert_called_once_with(scheduler_output)
        runner.model.assert_called_once()
        mock_dispatch.assert_called_once_with(
            runner.cudagraph_manager,
            2,
            4,
            4,
            2,
            1,
            need_eager=False,
        )

        self.assertEqual(len(captured_state), 1)
        state_kwargs = captured_state[0]
        self.assertIs(state_kwargs["input_batch"], input_batch)
        self.assertEqual(state_kwargs["slot_mappings_by_layer"], "slot-by-layer")
        self.assertIs(state_kwargs["hidden_states"], hidden_states)

        mock_ctx.assert_called_once()
        ctx_kwargs = mock_ctx.call_args.kwargs
        self.assertEqual(ctx_kwargs["num_tokens"], 8)
        self.assertEqual(ctx_kwargs["aclgraph_runtime_mode"], CUDAGraphMode.PIECEWISE)
        self.assertIs(ctx_kwargs["num_tokens_across_dp"], num_tokens_across_dp)
        self.assertEqual(ctx_kwargs["slot_mapping"], "slot-by-layer")
        self.assertFalse(ctx_kwargs["skip_compiled"])
