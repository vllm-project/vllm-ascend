from unittest.mock import MagicMock, patch

from vllm.config import ProfilerConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import NPUProfilerConfig


class TestTorchNPUProfilerWrapper(TestBase):
    def test_init_defers_underlying_profiler_until_start(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler) as mock_create:
            wrapper = TorchNPUProfilerWrapper(profiler_config)

            mock_create.assert_not_called()
            self.assertIsNone(wrapper.profiler)
            wrapper.set_output_name("dp0_pp0_tp0_dcp0_ep0_rank0")
            wrapper.start()

        mock_create.assert_called_once()
        self.assertTrue(mock_create.call_args.args[1].startswith("dp0_pp0_tp0_dcp0_ep0_rank0_"))
        self.assertIs(wrapper.profiler, mock_profiler)

    def test_start_stop_delegate_to_underlying_profiler(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")
            wrapper.start()
            wrapper.stop()

        mock_profiler.start.assert_called_once()
        mock_profiler.stop.assert_called_once()

    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler._ExperimentalConfig", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.profile", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.tensorboard_trace_handler", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ExportType", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ProfilerLevel", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.AiCMetrics", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ProfilerActivity", create=True)
    def test_create_profiler_enabled(
        self,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_profile,
        mock_experimental_config,
        mock_get_ascend_config,
    ):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_get_ascend_config.side_effect = RuntimeError("Ascend config is not initialized")

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            torch_profiler_with_stack=True,
            torch_profiler_with_memory=True,
            torch_profiler_record_shapes=True,
        )

        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.PipeUtilization = "PipeUtilization"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"

        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_profiler_instance = MagicMock()
        mock_profile.return_value = mock_profiler_instance

        result = TorchNPUProfilerWrapper._create_profiler(
            profiler_config,
            "warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
        )

        mock_experimental_config.assert_called_once()
        config_kwargs = mock_experimental_config.call_args.kwargs
        expected_config = {
            "export_type": "Text",
            "profiler_level": "Level1",
            "mstx": False,
            "aic_metrics": "PipeUtilization",
            "l2_cache": False,
            "op_attr": False,
            "data_simplification": True,
            "record_op_args": False,
            "gc_detect_threshold": None,
        }
        for key, expected_value in expected_config.items():
            self.assertEqual(config_kwargs[key], expected_value)

        mock_trace_handler.assert_called_once_with(
            "/path/to/traces",
            worker_name="warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
        )

        mock_profile.assert_called_once()
        profile_kwargs = mock_profile.call_args.kwargs
        self.assertEqual(profile_kwargs["activities"], ["CPU", "NPU"])
        self.assertTrue(profile_kwargs["profile_memory"])
        self.assertTrue(profile_kwargs["record_shapes"])
        self.assertFalse(profile_kwargs["with_stack"])
        self.assertEqual(profile_kwargs["with_modules"], True)
        self.assertEqual(profile_kwargs["on_trace_ready"], mock_trace_handler_instance)
        self.assertEqual(result, mock_profiler_instance)

    def test_create_profiler_disabled(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(profiler=None, torch_profiler_dir="")

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn("Unrecognized profiler: None", str(cm.exception))

    def test_create_profiler_empty_dir(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = MagicMock()
        profiler_config.profiler = "torch"
        profiler_config.torch_profiler_dir = ""

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn("torch_profiler_dir cannot be empty", str(cm.exception))

    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_raises_when_msmonitor_enabled(self, mock_get_ascend_config):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_get_ascend_config.return_value = MagicMock(msmonitor_use_daemon=True)
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn(
            "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time.",
            str(cm.exception),
        )

    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler._ExperimentalConfig", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.profile", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.tensorboard_trace_handler", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ExportType", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ProfilerLevel", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.AiCMetrics", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.ProfilerActivity", create=True)
    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_with_msmonitor_disabled(
        self,
        mock_get_ascend_config,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_profile,
        mock_experimental_config,
    ):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_ascend_config.npu_profiler_config = NPUProfilerConfig()
        mock_get_ascend_config.return_value = mock_ascend_config

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.PipeUtilization = "PipeUtilization"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"
        mock_profile.return_value = MagicMock()

        TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        mock_profile.assert_called_once()

    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    def test_create_profiler_config_enables_msmonitor(self, mock_get_ascend_config):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = True
        mock_get_ascend_config.return_value = mock_ascend_config
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with self.assertRaises(RuntimeError) as cm:
            TorchNPUProfilerWrapper._create_profiler(profiler_config, "test_trace")

        self.assertIn(
            "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time.",
            str(cm.exception),
        )

    def test_profiler_step_returns_true(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=MagicMock()):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")

        self.assertTrue(wrapper._profiler_step())

    def test_step_calls_underlying_start_after_delay_iterations(self):
        """Work matches vLLM WorkerProfiler: first N worker steps defer _start."""
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            delay_iterations=3,
            max_iterations=0,
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")
            wrapper.start()
            mock_profiler.start.assert_not_called()

            wrapper.step()
            wrapper.step()
            mock_profiler.start.assert_not_called()

            wrapper.step()
            mock_profiler.start.assert_called_once()

    def test_step_stops_underlying_profiler_after_max_iterations(self):
        """Work matches vLLM WorkerProfiler: stop when recorded steps exceed max_iterations."""
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            delay_iterations=0,
            max_iterations=1,
        )
        mock_profiler = MagicMock()

        with patch.object(TorchNPUProfilerWrapper, "_create_profiler", return_value=mock_profiler):
            wrapper = TorchNPUProfilerWrapper(profiler_config, "trace_name")
            wrapper.start()
            mock_profiler.start.assert_called_once()
            mock_profiler.stop.assert_not_called()

            wrapper.step()
            mock_profiler.stop.assert_not_called()

            wrapper.step()
            mock_profiler.stop.assert_called_once()

    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.profile")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.tensorboard_trace_handler")
    def test_sessions_apply_overrides_and_restore_startup_defaults(self, mock_handler, mock_profile, mock_config):
        """Session overrides reach the backend and never leak into later captures."""
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        mock_config.return_value.msmonitor_use_daemon = False
        mock_config.return_value.npu_profiler_config = NPUProfilerConfig()
        config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            torch_profiler_with_stack=True,
            torch_profiler_with_memory=True,
        )
        overrides = {"profile_memory": False, "record_shapes": True, "with_modules": False, "with_stack": True}
        original_overrides = dict(overrides)
        profilers = [MagicMock() for _ in range(4)]
        mock_profile.side_effect = profilers
        wrapper = TorchNPUProfilerWrapper(config, "first_rank0")
        for index, session_overrides in enumerate((overrides, dict.fromkeys(overrides, False), None, {})):
            wrapper.set_output_name(f"session{index}_rank0")
            wrapper.set_profiler_kwargs(session_overrides)
            wrapper.start()
            kwargs = mock_profile.call_args.kwargs
            expected = session_overrides or {
                "record_shapes": False,
                "profile_memory": True,
                "with_stack": False,
                "with_modules": True,
            }
            for name, value in expected.items():
                self.assertIs(kwargs[name], value)
            self.assertIs(wrapper.profiler, profilers[index])
            self.assertTrue(mock_handler.call_args.kwargs["worker_name"].startswith(f"session{index}_rank0_"))
            wrapper.stop()
            profilers[index].stop.assert_called_once()
        self.assertEqual(overrides, original_overrides)
        self.assertTrue(config.torch_profiler_with_memory)
        self.assertTrue(config.torch_profiler_with_stack)

    def test_invalid_overrides_preserve_pending_session(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        overrides = {"profile_memory": True}
        wrapper.set_profiler_kwargs(overrides)
        overrides["profile_memory"] = False
        for invalid in (
            {"activities": []},
            {"activities": ["CPU", "CPU"]},
            {"activities": ["CUDA"]},
            {"activities": "NPU"},
            {"activities": [[]]},
            {"output_dir": "relative/path"},
            {"output_dir": "s3://bucket/path"},
            {"output_dir": "/invalid\x00path"},
            {"output_dir": None},
            {"record_shapes": 1},
            {"with_stack": "false"},
            {"with_modules": None},
            {"with_flops": 0},
            {"experimental_config": None},
            {"experimental_config": {"profiler_level": "Level3"}},
            {"experimental_config": {"profiler_level": "Level_none"}},
            {"experimental_config": {"mstx": "true"}},
            {"experimental_config": {"msprof_tx": True}},
            {"experimental_config": {"aic_metrics": "Memory"}},
            {"experimental_config": {"export_type": ["Db"]}},
            {"experimental_config": {"l2_cache": True}},
            {"experimental_config": {"record_op_args": True}},
            {"experimental_config": {"op_attr": True}},
            {"on_trace_ready": "callback"},
            [],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wrapper.set_profiler_kwargs(invalid)
        with patch.object(wrapper, "_create_profiler", return_value=MagicMock()) as mock_create:
            wrapper.start()
            self.assertIs(mock_create.call_args.args[2]["profile_memory"], True)
            wrapper.stop()

    @patch("vllm_ascend.profiler.torch_npu_profiler.get_ascend_config")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler._ExperimentalConfig")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.profile")
    @patch("vllm_ascend.profiler.torch_npu_profiler.torch_npu.profiler.tensorboard_trace_handler")
    def test_extended_sessions_merge_with_startup_snapshot(self, mock_handler, mock_profile, mock_exp, mock_config):
        """Partial NPU overrides inherit startup fields, never the preceding session."""
        import torch_npu

        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/startup/traces",
            torch_profiler_activities=["CPU"],
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
            torch_profiler_with_flops=True,
            torch_profiler_with_stack=False,
        )
        npu_config = NPUProfilerConfig(with_stack=True, experimental_config={"profiler_level": "Level2", "mstx": True})
        mock_config.return_value.msmonitor_use_daemon = False
        mock_config.return_value.npu_profiler_config = npu_config
        wrapper = TorchNPUProfilerWrapper(config, "rank0")
        config.torch_profiler_with_memory = False
        config.torch_profiler_dir = "/changed/traces"
        config.torch_profiler_activities.append("PrivateUse1")
        npu_config.experimental_config.mstx = False

        wrapper.set_profiler_kwargs(
            {
                "output_dir": "/session/traces",
                "activities": ["NPU"],
                "with_flops": False,
                "experimental_config": {"profiler_level": "Level1"},
            }
        )
        wrapper.start()
        self.assertEqual(mock_profile.call_args.kwargs["activities"], [torch_npu.profiler.ProfilerActivity.NPU])
        self.assertFalse(mock_profile.call_args.kwargs["with_flops"])
        self.assertTrue(mock_profile.call_args.kwargs["record_shapes"])
        self.assertTrue(mock_profile.call_args.kwargs["with_stack"])
        self.assertFalse(mock_profile.call_args.kwargs["with_modules"])
        self.assertEqual(mock_handler.call_args.args, ("/session/traces",))
        self.assertEqual(mock_exp.call_args.kwargs["profiler_level"], torch_npu.profiler.ProfilerLevel.Level1)
        self.assertTrue(mock_exp.call_args.kwargs["mstx"])
        wrapper.stop()

        wrapper.set_profiler_kwargs({"experimental_config": {"mstx": False}})
        wrapper.start()
        self.assertFalse(mock_exp.call_args.kwargs["mstx"])
        self.assertEqual(mock_exp.call_args.kwargs["profiler_level"], torch_npu.profiler.ProfilerLevel.Level2)
        wrapper.stop()

        for overrides in ({"experimental_config": {}}, {}, None):
            wrapper.set_profiler_kwargs(overrides)
            wrapper.start()
            kwargs = mock_profile.call_args.kwargs
            self.assertEqual(kwargs["activities"], [torch_npu.profiler.ProfilerActivity.CPU])
            self.assertTrue(kwargs["profile_memory"])
            self.assertTrue(kwargs["with_flops"])
            self.assertEqual(mock_handler.call_args.args, ("/startup/traces",))
            self.assertEqual(mock_exp.call_args.kwargs["profiler_level"], torch_npu.profiler.ProfilerLevel.Level2)
            self.assertTrue(mock_exp.call_args.kwargs["mstx"])
            wrapper.stop()

    def test_nested_session_inputs_are_copied(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        overrides = {"activities": ["NPU"], "experimental_config": {"mstx": True}}
        wrapper.set_profiler_kwargs(overrides)
        overrides["activities"].append("CPU")
        overrides["experimental_config"]["mstx"] = False
        with patch.object(wrapper, "_create_profiler", return_value=MagicMock()) as mock_create:
            wrapper.start()
            self.assertEqual(mock_create.call_args.args[2]["activities"], ["NPU"])
            self.assertTrue(mock_create.call_args.args[2]["experimental_config"]["mstx"])
            wrapper.stop()

    def test_startup_privateuse1_activity_maps_to_npu(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(
            profiler="torch", torch_profiler_dir="/path/to/traces", torch_profiler_activities=["PrivateUse1"]
        )
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        with patch.object(wrapper, "_create_profiler", return_value=MagicMock()) as mock_create:
            wrapper.start()
            self.assertEqual(mock_create.call_args.args[2]["activities"], ["NPU"])
            wrapper.stop()

    def test_delayed_session_ignores_duplicate_start_and_configuration_changes(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "first")
        wrapper.set_profiler_kwargs({"record_shapes": True})
        with patch.object(wrapper, "_create_profiler", return_value=MagicMock()) as mock_create:
            wrapper.start(delay_iterations=2)
            wrapper.set_output_name("ignored")
            wrapper.set_profiler_kwargs({"record_shapes": False})
            wrapper.start()
            wrapper.step()
            mock_create.assert_not_called()
            wrapper.step()
            self.assertIs(mock_create.call_args.args[2]["record_shapes"], True)
            self.assertTrue(mock_create.call_args.args[1].startswith("first_"))
            wrapper.stop()
            wrapper.stop()
            mock_create.return_value.stop.assert_called_once()

    def test_cancelled_delay_does_not_create_profiler(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        with patch.object(wrapper, "_create_profiler", return_value=MagicMock()) as mock_create:
            wrapper.stop()
            wrapper.start(delay_iterations=2)
            wrapper.step()
            wrapper.stop()
            mock_create.assert_not_called()
            wrapper.start()
            mock_create.assert_called_once()
            wrapper.stop()

    def test_auto_stop_allows_new_session_with_unique_output(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "same_prefix_rank0")
        with patch.object(wrapper, "_create_profiler", side_effect=[MagicMock(), MagicMock()]) as mock_create:
            for _ in range(2):
                wrapper.start(max_iterations=1)
                wrapper.step()
                wrapper.step()
                self.assertFalse(wrapper.is_running)
            self.assertNotEqual(mock_create.call_args_list[0].args[1], mock_create.call_args_list[1].args[1])

    def test_failed_backend_start_raises_and_can_be_retried(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        failed, succeeding = MagicMock(), MagicMock()
        failed.start.side_effect = RuntimeError("start failed")
        with patch.object(wrapper, "_create_profiler", side_effect=[failed, succeeding]):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                wrapper.start()
            self.assertFalse(wrapper.is_running)
            self.assertFalse(wrapper._active)
            self.assertIsNone(wrapper.profiler)
            failed.stop.assert_called_once()
            wrapper.start()
            self.assertIs(wrapper.profiler, succeeding)
            self.assertTrue(wrapper.is_running)
            wrapper.stop()

    def test_failed_backend_creation_raises_and_can_be_retried(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        succeeding = MagicMock()
        with patch.object(wrapper, "_create_profiler", side_effect=[RuntimeError("create failed"), succeeding]):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                wrapper.start()
            self.assertFalse(wrapper.is_running)
            self.assertFalse(wrapper._active)
            self.assertIsNone(wrapper.profiler)
            wrapper.start()
            self.assertIs(wrapper.profiler, succeeding)
            wrapper.stop()

    def test_failed_start_cleanup_preserves_original_error(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        native = MagicMock()
        native.start.side_effect = RuntimeError("start failed")
        native.stop.side_effect = RuntimeError("cleanup failed")
        with (
            patch.object(wrapper, "_create_profiler", return_value=native),
            self.assertRaisesRegex(RuntimeError, "start failed"),
        ):
            wrapper.start()
        self.assertFalse(wrapper._active)
        self.assertFalse(wrapper.is_running)
        self.assertIsNone(wrapper.profiler)

    def test_delayed_start_failure_resets_state_without_interrupting_step(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        for failure_stage in ("create", "start"):
            with self.subTest(failure_stage=failure_stage):
                wrapper = TorchNPUProfilerWrapper(config, "trace")
                failed, succeeding = MagicMock(), MagicMock()
                failed.start.side_effect = RuntimeError("start failed")
                first = RuntimeError("create failed") if failure_stage == "create" else failed
                with (
                    patch.object(wrapper, "_create_profiler", side_effect=[first, succeeding]),
                    patch("vllm_ascend.profiler.torch_npu_profiler.logger") as mock_logger,
                ):
                    wrapper.start(delay_iterations=1, max_iterations=1)
                    wrapper.step()
                    mock_logger.exception.assert_called_once()
                    self.assertFalse(wrapper._active)
                    self.assertFalse(wrapper.is_running)
                    self.assertIsNone(wrapper.profiler)
                    self.assertEqual(wrapper._active_iteration_count, 0)
                    self.assertEqual(wrapper._profiling_for_iters, 0)
                    wrapper.start(delay_iterations=0)
                    self.assertTrue(wrapper.is_running)
                    self.assertIs(wrapper.profiler, succeeding)
                    wrapper.stop()

    def test_stop_without_native_profiler_is_safe(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        wrapper._stop()
        wrapper._stop()
        self.assertIsNone(wrapper.profiler)

    def test_failed_backend_stop_allows_new_profiler(self):
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        wrapper = TorchNPUProfilerWrapper(config, "trace")
        failed, succeeding = MagicMock(), MagicMock()
        failed.stop.side_effect = RuntimeError("stop failed")
        with patch.object(wrapper, "_create_profiler", side_effect=[failed, succeeding]):
            wrapper.start()
            wrapper.stop()
            self.assertFalse(wrapper.is_running)
            self.assertIsNone(wrapper.profiler)
            wrapper.start()
            self.assertIs(wrapper.profiler, succeeding)
            wrapper.stop()
