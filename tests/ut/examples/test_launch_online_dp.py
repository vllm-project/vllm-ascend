"""Run with: python -m unittest discover -s tests/ut/examples -p test_launch_online_dp.py -v."""

import contextlib
import importlib.util
import io
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "external_online_dp"
spec = importlib.util.spec_from_file_location("launch_online_dp", EXAMPLE_DIR / "launch_online_dp.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dp launcher ")
        self.addCleanup(temp.cleanup)
        self.template = Path(temp.name) / "run template.sh"
        self.template.write_text("exit 0\n", encoding="utf-8")
        self.base = [
            "--dp-size",
            "4",
            "--dp-size-local",
            "2",
            "--dp-rank-start",
            "2",
            "--tp-size",
            "2",
            "--dp-address",
            "127.0.0.1",
            "--template",
            str(self.template),
        ]

    def capture(self, *extra, statuses=(0, 0)):
        workers = [mock.Mock(wait=mock.Mock(return_value=status)) for status in statuses]
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base, *extra]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=workers) as popen,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = launcher.main()
        return result, popen.call_args_list, workers

    def test_fixed_interface_and_second_node_device_allocation(self):
        result, calls, _ = self.capture()
        self.assertEqual(result, 0)
        self.assertEqual(calls[0].args[0][4:], ["0,1", "9000", "4", "2", "127.0.0.1", "12345", "2", "1", "1"])
        self.assertEqual(calls[1].args[0][4:], ["2,3", "9001", "4", "3", "127.0.0.1", "12345", "2", "1", "1"])
        for call in calls:
            self.assertTrue(call.kwargs["start_new_session"])

    def test_pcp_and_pp_individually_and_together(self):
        for sizes, tail, devices in [
            ([], ["2", "1", "1"], "2,3"),
            (["--pcp-size", "2"], ["2", "2", "1"], "4,5,6,7"),
            (["--pp-size", "2"], ["2", "1", "2"], "4,5,6,7"),
            (["--pcp-size", "2", "--pp-size", "3"], ["2", "2", "3"], "12,13,14,15,16,17,18,19,20,21,22,23"),
        ]:
            with self.subTest(sizes=sizes):
                _, calls, _ = self.capture(*sizes)
                command = calls[1].args[0]
                self.assertEqual(command[10:], tail)
                self.assertEqual(command[4], devices)

    def test_invalid_topology_and_missing_template(self):
        for extra in [
            ["--dp-size", "0"],
            ["--tp-size", "0"],
            ["--dp-size-local", "0"],
            ["--pcp-size", "-1"],
            ["--pp-size", "0"],
            ["--dp-rank-start", "-1"],
            ["--dp-rank-start", "3"],
            ["--template", str(self.template.parent / "missing.sh")],
        ]:
            with (
                self.subTest(extra=extra),
                mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base, *extra]),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    launcher.parse_args()
                self.assertEqual(caught.exception.code, 2)

    def test_script_alias_and_dp1_default(self):
        with mock.patch.object(
            launcher.sys,
            "argv",
            [
                "launch_online_dp.py",
                *self.base,
                "--script",
                str(self.template),
                "--dp-size",
                "1",
                "--dp-size-local",
                "-1",
                "--dp-rank-start",
                "0",
            ],
        ):
            args = launcher.parse_args()
        self.assertEqual(args.template, str(self.template))
        self.assertEqual(args.dp_size_local, 1)

    def test_waits_for_all_ranks_and_reports_failure(self):
        result, _, workers = self.capture(statuses=(7, 0))
        self.assertEqual(result, 1)
        for worker in workers:
            worker.wait.assert_called_once()

    def test_spawn_failure_cleans_started_groups(self):
        worker = mock.Mock(pid=101)
        error = OSError("cannot spawn second rank")
        previous_handlers = [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=[worker, error]),
            mock.patch.object(launcher.os, "killpg", side_effect=[None, ProcessLookupError()], create=True) as killpg,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(OSError) as caught,
        ):
            launcher.main()
        self.assertIs(caught.exception, error)
        self.assertEqual(killpg.call_args_list, [mock.call(101, signal.SIGTERM), mock.call(101, 0)])
        worker.wait.assert_called_once()
        self.assertEqual([signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)], previous_handlers)

    def test_interrupt_during_wait_cleans_all_groups(self):
        def interrupted_wait():
            if stop_signal == signal.SIGTERM:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            raise KeyboardInterrupt

        def group_signal(pid, sig):
            if sig == 0:
                raise ProcessLookupError
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)

        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(stop_signal=stop_signal):
                workers = [mock.Mock(pid=101), mock.Mock(pid=102)]

                def wait(worker=workers[0]):
                    worker.wait.side_effect = None
                    interrupted_wait()

                workers[0].wait.side_effect = wait
                previous_handlers = [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
                with (
                    mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
                    mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
                    mock.patch.object(launcher.os, "killpg", side_effect=group_signal, create=True) as killpg,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises((KeyboardInterrupt, SystemExit)) as caught,
                ):
                    launcher.main()
                if stop_signal == signal.SIGTERM:
                    self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
                self.assertEqual(
                    killpg.call_args_list[:2], [mock.call(101, signal.SIGTERM), mock.call(102, signal.SIGTERM)]
                )
                self.assertEqual(workers[0].wait.call_count, 2)
                workers[1].wait.assert_called_once()
                self.assertEqual([signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)], previous_handlers)

    def test_signal_during_spawn_records_worker_before_cleanup(self):
        worker = mock.Mock(pid=101)

        def spawn(*args, **kwargs):
            signal.getsignal(stop_signal)(stop_signal, None)
            return worker

        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(stop_signal=stop_signal):
                worker.reset_mock()
                with (
                    mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
                    mock.patch.object(launcher.subprocess, "Popen", side_effect=spawn) as popen,
                    mock.patch.object(launcher.os, "killpg", side_effect=[None, ProcessLookupError()], create=True),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    launcher.main()
                self.assertEqual(caught.exception.code, 128 + stop_signal)
                popen.assert_called_once()
                worker.wait.assert_called_once()

    def test_shutdown_waits_for_descendants_after_shell_exits(self):
        worker = mock.Mock(pid=101, poll=mock.Mock(return_value=0))
        with (
            mock.patch.object(
                launcher.os, "killpg", side_effect=[None, None, ProcessLookupError()], create=True
            ) as killpg,
            mock.patch.object(launcher.time, "monotonic", side_effect=[0, 0, 1]),
            mock.patch.object(launcher.time, "sleep") as sleep,
        ):
            launcher.shutdown_processes([worker])
        self.assertEqual(killpg.call_args_list, [mock.call(101, signal.SIGTERM), mock.call(101, 0), mock.call(101, 0)])
        sleep.assert_called_once()
        worker.wait.assert_called_once()

    def test_shutdown_kills_remaining_groups_after_shared_timeout(self):
        workers = [mock.Mock(pid=101), mock.Mock(pid=102)]
        with (
            mock.patch.object(launcher.os, "killpg", create=True) as killpg,
            mock.patch.object(launcher.signal, "SIGKILL", 9, create=True),
            mock.patch.object(launcher.time, "monotonic", side_effect=[0, 0, launcher.SHUTDOWN_TIMEOUT + 1]),
            mock.patch.object(launcher.time, "sleep"),
        ):
            launcher.shutdown_processes(workers)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(101, signal.SIGTERM),
                mock.call(102, signal.SIGTERM),
                mock.call(101, 0),
                mock.call(102, 0),
                mock.call(101, 9),
                mock.call(102, 9),
            ],
        )
        for worker in workers:
            worker.wait.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "Requires POSIX process groups")
    def test_real_stop_signals_reach_shell_descendants(self):
        self.bash()
        worker_script = self.template.parent / "worker.py"
        worker_script.write_text(
            "import os, signal, sys\nfrom pathlib import Path\n"
            "def stop(signum, frame):\n"
            "    Path(sys.argv[2]).write_text('cleaned')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "while True:\n    signal.pause()\n",
            encoding="utf-8",
        )
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(stop_signal=stop_signal):
                ready = self.template.parent / f"ready-{stop_signal}"
                cleaned = self.template.parent / f"cleaned-{stop_signal}"
                group = self.template.parent / f"group-{stop_signal}"
                self.template.write_text(
                    f"echo $$ > {shlex.quote(str(group))}\n"
                    f"{shlex.join([sys.executable, str(worker_script), str(ready), str(cleaned)])}\nwait\n",
                    encoding="utf-8",
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(EXAMPLE_DIR / "launch_online_dp.py"),
                        *self.base,
                        "--dp-size",
                        "1",
                        "--dp-size-local",
                        "1",
                        "--dp-rank-start",
                        "0",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue(ready.exists(), "Worker did not become ready")
                    process.send_signal(stop_signal)
                    _, stderr = process.communicate(timeout=launcher.SHUTDOWN_TIMEOUT + 5)
                    self.assertNotEqual(process.returncode, 0, stderr)
                    self.assertEqual(cleaned.read_text(), "cleaned")
                finally:
                    if group.exists():
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(int(group.read_text()), signal.SIGKILL)
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)

    def bash(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("Bash is not installed")
        return bash

    def test_real_shell_receives_all_nine_arguments(self):
        self.template.write_text(
            'printf "%s\\n" "$#"\nfor arg in "$@"; do printf "%s\\n" "$arg"; done\n',
            encoding="utf-8",
        )
        for sizes in [
            [],
            ["--pcp-size", "2"],
            ["--pp-size", "3"],
            ["--pcp-size", "2", "--pp-size", "3"],
        ]:
            with self.subTest(sizes=sizes):
                _, calls, _ = self.capture(*sizes)
                command = calls[1].args[0]
                command[0] = self.bash()
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["9", *command[4:]])

    def test_real_shell_preserves_pipeline_failure(self):
        self.template.write_text("false | cat\n", encoding="utf-8")
        _, calls, _ = self.capture()
        command = calls[0].args[0]
        command[0] = self.bash()
        result = subprocess.run(command, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)

    def test_supplied_template_forwards_parallel_sizes(self):
        # Replace only the vllm executable; execute the actual template with Bash.
        wrapper = 'vllm() { printf "%s\\n" "$ASCEND_RT_VISIBLE_DEVICES" "$@"; }; source "$@"'
        for sizes, expected in [([], ["1", "1"]), (["--pcp-size", "2", "--pp-size", "3"], ["2", "3"])]:
            with self.subTest(sizes=sizes):
                _, calls, _ = self.capture(*sizes)
                command = calls[0].args[0]
                result = subprocess.run(
                    [
                        self.bash(),
                        "-c",
                        wrapper,
                        "bash",
                        (EXAMPLE_DIR / "run_dp_template.sh").as_posix(),
                        *command[4:],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                values = result.stdout.splitlines()
                self.assertEqual(values[:3], [command[4], "serve", "model_path"])
                for flag, value in [
                    ("--tensor-parallel-size", "2"),
                    ("--prefill-context-parallel-size", expected[0]),
                    ("--pipeline-parallel-size", expected[1]),
                ]:
                    self.assertEqual(values[values.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
