"""Unit tests for the inductor-track warmup tail hooks (stage5 F2).

``NPUWorker._inductor_track_warmup_tail`` mirrors the upstream gpu_worker.py
compile_or_warm_up_model tail (trigger_inductor_lazy_init + jit monitor) on
the active inductor compile-backend track only:

  1. track on (backend == "inductor", mode == VLLM_COMPILE): both upstream
     hooks are invoked;
  2. track off (mode == NONE, or a non-inductor backend): neither hook runs
     — off-track behavior stays bit-identical (总纲 §〇 红线 1);
  3. a failing jit monitor (triton-ascend 3.2.2 cannot import triton.knobs,
     stage5 M0 P0-4) degrades to a warning instead of crashing warmup.

Config-level only: no NPU device is initialized (worker built via
object.__new__, attributes set directly).
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import CompilationConfig, CompilationMode, VllmConfig

from tests.ut.base import TestBase


def _make_vllm_config(track: bool, mode: CompilationMode) -> VllmConfig:
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                backend="inductor" if track else "",
            ),
        )
    vllm_config.compilation_config.mode = mode
    vllm_config.model_config = SimpleNamespace(enforce_eager=False)
    return vllm_config


def _make_worker(track: bool, mode: CompilationMode) -> "object":
    from vllm_ascend.worker.worker import NPUWorker

    worker = object.__new__(NPUWorker)
    worker.vllm_config = _make_vllm_config(track, mode)
    worker.device = "npu:0"
    return worker


class TestInductorTrackWarmupTail(TestBase):
    def test_track_on_triggers_lazy_init_and_jit_monitor(self):
        worker = _make_worker(
            track=True, mode=CompilationMode.VLLM_COMPILE
        )
        with patch(
            "vllm.compilation.compiler_interface.trigger_inductor_lazy_init"
        ) as mock_lazy, patch(
            "vllm.utils.jit_monitor.activate"
        ) as mock_monitor:
            worker._inductor_track_warmup_tail()
        mock_lazy.assert_called_once_with("npu:0")
        mock_monitor.assert_called_once()

    def test_track_off_mode_none_runs_nothing(self):
        worker = _make_worker(track=True, mode=CompilationMode.NONE)
        with patch(
            "vllm.compilation.compiler_interface.trigger_inductor_lazy_init"
        ) as mock_lazy, patch(
            "vllm.utils.jit_monitor.activate"
        ) as mock_monitor:
            worker._inductor_track_warmup_tail()
        mock_lazy.assert_not_called()
        mock_monitor.assert_not_called()

    def test_track_off_legacy_backend_runs_nothing(self):
        worker = _make_worker(
            track=False, mode=CompilationMode.VLLM_COMPILE
        )
        with patch(
            "vllm.compilation.compiler_interface.trigger_inductor_lazy_init"
        ) as mock_lazy, patch(
            "vllm.utils.jit_monitor.activate"
        ) as mock_monitor:
            worker._inductor_track_warmup_tail()
        mock_lazy.assert_not_called()
        mock_monitor.assert_not_called()

    def test_failing_jit_monitor_degrades_to_warning(self):
        worker = _make_worker(
            track=True, mode=CompilationMode.VLLM_COMPILE
        )
        with patch(
            "vllm.compilation.compiler_interface.trigger_inductor_lazy_init"
        ) as mock_lazy, patch(
            "vllm.utils.jit_monitor.activate",
            side_effect=ImportError(
                "cannot import name 'getenv' from 'triton._C.libtriton'"
            ),
        ):
            # Must not raise: warmup survives a broken jit monitor.
            worker._inductor_track_warmup_tail()
        mock_lazy.assert_called_once()
