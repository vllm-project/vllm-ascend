"""Exercise AISBench lifecycle handling with real, CPU-only child processes."""

import contextlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from tools import aisbench


@contextlib.contextmanager
def deadline(seconds=3):
    def expired(signum, frame):
        raise TimeoutError("AISBench lifecycle test timed out")

    old_handler = signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay:
            remaining = max(old_delay - (time.monotonic() - started), 0.000001)
            signal.setitimer(signal.ITIMER_REAL, remaining, old_interval)


@unittest.skipUnless(hasattr(os, "killpg"), "AISBench process groups require POSIX")
class TestAisbenchProcess(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="aisbench-process-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, old_cwd)
        environment = patch.dict(os.environ, {"PATH": f"{self.root}{os.pathsep}{os.environ.get('PATH', '')}"})
        environment.start()
        self.addCleanup(environment.stop)
        real_sleep = time.sleep
        sleep = patch.object(aisbench.time, "sleep", lambda seconds: real_sleep(min(seconds, 0.01)))
        sleep.start()
        self.addCleanup(sleep.stop)
        self.runner = aisbench.AisbenchRunner.__new__(aisbench.AisbenchRunner)
        self.runner.__dict__.update(
            task_type="accuracy",
            dataset_conf="test/dataset",
            request_conf="chat",
            num_prompts=None,
            profile_marker=None,
            profile_targets=(),
            proc=None,
            exp_folder=None,
            result_line=None,
        )
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        # Also clean the detached child when proving failures on the old code.
        for path in self.root.glob("*.pid"):
            with contextlib.suppress(ProcessLookupError, ValueError):
                pid = int(path.read_text())
                if Path("/proc").exists() and Path(f"/proc/{pid}/cwd").resolve() != self.root:
                    continue
                os.kill(pid, signal.SIGKILL)
        if self.runner.proc is not None:
            self.runner.__exit__(None, None, None)

    def executable(self, body):
        path = self.root / "ais_bench"
        path.write_text(
            f"#!{sys.executable}\n"
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "Path('benchmark.pid').write_text(str(os.getpid()))\n"
            f"{body}\n",
            encoding="utf-8",
        )
        path.chmod(0o700)

    def wait_for_file(self, name):
        path = self.root / name
        with deadline():
            while not path.exists() or not path.read_text():
                time.sleep(0.01)
        return path

    def run_constructor(self):
        config = dict(
            case_type="accuracy",
            dataset_path_local=str(self.root),
            model_path=str(self.root),
            request_conf="chat",
            dataset_conf="test/dataset",
            max_out_len=1,
            batch_size=1,
        )
        launch = aisbench.AisbenchRunner._run_aisbench_task

        def record_launch(instance):
            self.runner = instance
            instance.profile_marker = None
            instance.profile_targets = ()
            launch(instance)
            self.launched_proc = instance.proc
            self.wait_for_file("benchmark.pid")

        with (
            patch.object(aisbench.AisbenchRunner, "_init_dataset_conf"),
            patch.object(aisbench.AisbenchRunner, "_init_request_conf"),
            patch.object(aisbench.AisbenchRunner, "_prepare_profiling"),
            patch.object(aisbench.AisbenchRunner, "_start_profiling", return_value=None),
            patch.object(aisbench.AisbenchRunner, "_run_aisbench_task", record_launch),
        ):
            return aisbench.AisbenchRunner("model", 8000, config, verify=False)

    def test_missing_executable_fails_at_launch(self):
        with patch.dict(os.environ, {"PATH": str(self.root)}), deadline(), self.assertRaises(FileNotFoundError):
            self.runner._run_aisbench_task()

    def test_nonzero_exit_is_reported_without_error_keyword(self):
        for folder in ("", "print('Current exp folder: /tmp/experiment', flush=True)\n"):
            with self.subTest(folder=bool(folder)):
                self.executable(folder + "print('configuration failed', flush=True)\nraise SystemExit(7)")
                self.runner._run_aisbench_task()
                with deadline(), self.assertRaisesRegex(RuntimeError, "exited with code 7"):
                    self.runner._wait_for_task()
                self.runner.__exit__(None, None, None)

    def test_zero_exit_without_expected_output_fails(self):
        for folder in ("", "print('Current exp folder: /tmp/experiment', flush=True)\n"):
            with self.subTest(folder=bool(folder)):
                self.executable(folder + "print('no result marker', flush=True)")
                self.runner._run_aisbench_task()
                with deadline(), self.assertRaisesRegex(RuntimeError, "exited without expected output"):
                    self.runner._wait_for_task()
                self.runner.__exit__(None, None, None)

    def test_success_waits_for_process_exit_and_preserves_output(self):
        self.executable(
            "print('Current exp folder: /tmp/experiment', flush=True)\n"
            "print('write csv to /tmp/results.csv', flush=True)\n"
            "time.sleep(0.1)\n"
            "Path('finished').write_text('done')"
        )
        self.runner._run_aisbench_task()
        with deadline():
            self.runner._wait_for_task()
        self.assertEqual(self.runner.exp_folder, "/tmp/experiment")
        self.assertIn("write csv to /tmp/results.csv", self.runner.result_line)
        self.assertTrue((self.root / "finished").exists())
        self.assertEqual(self.runner.proc.returncode, 0)

    def test_success_marker_does_not_hide_late_failure(self):
        self.executable(
            "print('Current exp folder: /tmp/experiment', flush=True)\n"
            "print('write csv to /tmp/results.csv', flush=True)\n"
            "time.sleep(0.1)\n"
            "raise SystemExit(9)"
        )
        self.runner._run_aisbench_task()
        with deadline(), self.assertRaisesRegex(RuntimeError, "exited with code 9"):
            self.runner._wait_for_task()

    def test_context_exit_stops_and_reaps_the_actual_process(self):
        self.executable("time.sleep(30)")
        self.runner._run_aisbench_task()
        pid = int(self.wait_for_file("benchmark.pid").read_text())
        proc = self.runner.proc
        self.assertEqual(proc.pid, pid)
        with deadline():
            self.runner.__exit__(None, None, None)
        self.assertIsNotNone(proc.returncode)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.runner.__exit__(None, None, None)

    def test_arguments_are_not_interpreted_by_a_shell(self):
        self.executable("import json\nPath('argv.json').write_text(json.dumps(sys.argv[1:]))")
        self.runner.request_conf = "chat config; touch injected"
        self.runner._run_aisbench_task()
        self.runner.proc.wait(timeout=3)
        arguments = json.loads((self.root / "argv.json").read_text())
        self.assertEqual(arguments[arguments.index("--models") + 1], "chat config; touch injected_custom")
        self.assertFalse((self.root / "injected_custom").exists())

    def test_constructor_failure_stops_process_before_context_entry(self):
        self.executable("print('ERROR benchmark failed', flush=True)\ntime.sleep(30)")
        with deadline(), self.assertRaisesRegex(RuntimeError, "Some errors happened"):
            self.run_constructor()
        self.assertIsNone(self.runner.proc)
        self.assertIsNotNone(self.launched_proc.returncode)

    def test_keyboard_interrupt_stops_process_before_context_entry(self):
        self.executable("time.sleep(30)")
        with (
            patch.object(aisbench.AisbenchRunner, "_wait_for_task", side_effect=KeyboardInterrupt),
            deadline(),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.run_constructor()
        self.assertIsNone(self.runner.proc)
        self.assertIsNotNone(self.launched_proc.returncode)

    def test_cleanup_failure_preserves_original_error_and_finishes_profiling(self):
        self.executable("time.sleep(30)")
        with (
            patch.object(aisbench.AisbenchRunner, "_wait_for_task", side_effect=ValueError("original failure")),
            patch.object(
                aisbench.AisbenchRunner, "_stop_aisbench_task", side_effect=OSError("cleanup failure"), create=True
            ),
            patch.object(aisbench.AisbenchRunner, "_finish_profiling") as finish,
            self.assertRaisesRegex(ValueError, "original failure"),
        ):
            self.run_constructor()
        finish.assert_called_once_with(None)

    def test_profiling_failure_occurs_after_process_cleanup(self):
        self.executable(
            "print('Current exp folder: /tmp/experiment', flush=True)\n"
            "print('write csv to /tmp/results.csv', flush=True)"
        )
        with (
            patch.object(aisbench.AisbenchRunner, "_finish_profiling", side_effect=ValueError("profile failure")),
            deadline(),
            self.assertRaisesRegex(ValueError, "profile failure"),
        ):
            self.run_constructor()
        self.assertIsNone(self.runner.proc)
        self.assertEqual(self.launched_proc.returncode, 0)

    def test_context_exit_stops_worker_processes(self):
        worker = (
            "import os, time; from pathlib import Path; Path('worker.pid').write_text(str(os.getpid())); time.sleep(30)"
        )
        self.executable(
            "import signal, subprocess\n"
            f"worker = subprocess.Popen([sys.executable, '-c', {worker!r}])\n"
            "def stop(signum, frame):\n"
            "    worker.wait(timeout=2)\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "Path('ready').write_text('ready')\n"
            "time.sleep(30)"
        )
        self.runner._run_aisbench_task()
        self.wait_for_file("ready")
        worker_pid = int(self.wait_for_file("worker.pid").read_text())
        with deadline():
            self.runner.__exit__(None, None, None)
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)

    def test_sigterm_resistant_process_is_killed_and_reaped(self):
        self.executable(
            "import signal\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path('ready').write_text('ready')\n"
            "time.sleep(30)"
        )
        self.runner._run_aisbench_task()
        self.wait_for_file("ready")
        proc = self.runner.proc
        original_wait = proc.wait

        def short_grace_period(timeout=None):
            return original_wait(timeout=min(timeout, 0.05) if timeout is not None else None)

        with patch.object(proc, "wait", short_grace_period), deadline():
            self.runner.__exit__(None, None, None)
        self.assertEqual(proc.returncode, -signal.SIGKILL)

    def test_sigkill_timeout_preserves_handle_for_later_reaping(self):
        # Simulate a child that remains alive after both termination signals.
        proc = Mock(pid=123)
        self.runner.proc = proc
        self.addCleanup(setattr, self.runner, "proc", None)
        proc.wait.side_effect = aisbench.subprocess.TimeoutExpired("ais_bench", 8)
        with patch.object(aisbench.os, "killpg") as killpg, self.assertRaises(aisbench.subprocess.TimeoutExpired):
            self.runner._stop_aisbench_task()
        self.assertEqual(proc.wait.call_args_list, [call(timeout=8), call(timeout=8)])
        self.assertEqual(killpg.call_args_list, [call(proc.pid, signal.SIGTERM), call(proc.pid, signal.SIGKILL)])
        self.assertIs(self.runner.proc, proc)

        # A later cleanup can still reap the same child once it becomes ready.
        proc.wait.side_effect = None
        proc.wait.return_value = -signal.SIGKILL
        with patch.object(aisbench.os, "killpg"):
            self.runner._stop_aisbench_task()
        self.assertIsNone(self.runner.proc)


if __name__ == "__main__":
    unittest.main()
