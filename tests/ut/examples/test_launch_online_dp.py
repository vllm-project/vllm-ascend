"""Run with: python -m unittest discover -s tests/ut/examples -p test_launch_online_dp.py -v."""

import contextlib
import importlib.util
import io
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "external_online_dp"
spec = importlib.util.spec_from_file_location("launch_online_dp", EXAMPLE_DIR / "launch_online_dp.py")
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
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

    def capture(self, *extra):
        workers = [mock.Mock(poll=mock.Mock(return_value=0)) for _ in range(2)]
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

    def test_success_requires_all_ranks_to_exit(self):
        workers = [mock.Mock(), mock.Mock()]
        workers[0].poll.side_effect = [None, 0, 0]
        workers[1].poll.side_effect = [None, None, 0]
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
            mock.patch.object(launcher.time, "sleep") as sleep,
            mock.patch.object(launcher.os, "killpg", create=True) as killpg,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = launcher.main()
        self.assertEqual(result, 0)
        self.assertEqual(sleep.call_args_list, [mock.call(launcher.POLL_INTERVAL)] * 2)
        killpg.assert_not_called()
        for worker in workers:
            self.assertEqual(worker.poll.call_count, 3)

    def test_second_rank_failure_stops_running_first_rank(self):
        for code in (7, -signal.SIGTERM):
            with self.subTest(code=code):
                workers = [mock.Mock(pid=101), mock.Mock(pid=102)]
                workers[0].poll.return_value = None
                workers[1].poll.return_value = code

                events = mock.Mock()
                events.attach_mock(workers[1].poll, "poll_second")
                events.attach_mock(workers[0].wait, "wait_first")
                with (
                    mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
                    mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
                    mock.patch.object(launcher.time, "sleep") as sleep,
                    mock.patch.object(launcher.os, "killpg", create=True) as killpg,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(RuntimeError, f"DP rank 3 exited with code {code}"),
                ):
                    events.attach_mock(killpg, "signal_group")
                    launcher.main()
                # Detect and signal the failure before waiting on the live rank.
                events.assert_has_calls(
                    [
                        mock.call.poll_second(),
                        mock.call.signal_group(101, signal.SIGTERM),
                        mock.call.signal_group(102, signal.SIGTERM),
                        mock.call.wait_first(),
                    ]
                )
                sleep.assert_not_called()
                for worker in workers:
                    worker.wait.assert_called_once()

    def test_spawn_failure_cleans_started_groups(self):
        worker = mock.Mock(pid=101)
        error = OSError("cannot spawn second rank")
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=[worker, error]),
            mock.patch.object(launcher.os, "killpg", create=True) as killpg,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(OSError) as caught,
        ):
            launcher.main()
        self.assertIs(caught.exception, error)
        killpg.assert_called_once_with(101, signal.SIGTERM)
        worker.wait.assert_called_once()

    def test_interrupt_during_monitoring_cleans_all_groups(self):
        for error in (KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                workers = [mock.Mock(pid=101), mock.Mock(pid=102)]
                workers[0].poll.side_effect = error
                with (
                    mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
                    mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
                    mock.patch.object(
                        launcher.os, "killpg", side_effect=[ProcessLookupError(), None], create=True
                    ) as killpg,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(type(error)) as caught,
                ):
                    launcher.main()
                self.assertIs(caught.exception, error)
                self.assertEqual(
                    killpg.call_args_list, [mock.call(101, signal.SIGTERM), mock.call(102, signal.SIGTERM)]
                )
                workers[0].wait.assert_called_once()
                workers[1].wait.assert_called_once()

    def test_sigterm_handler_cleans_all_groups(self):
        workers = [mock.Mock(pid=101), mock.Mock(pid=102)]
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
            mock.patch.object(launcher.signal, "signal") as register,
            mock.patch.object(launcher.os, "killpg", create=True) as killpg,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            # Invoke the registered handler during monitoring without sending a real signal.
            workers[0].poll.side_effect = lambda: register.call_args.args[1](signal.SIGTERM, None)
            launcher.main()
        register.assert_called_once_with(signal.SIGTERM, launcher.handle_sigterm)
        self.assertIsNone(caught.exception.code)
        self.assertEqual(killpg.call_args_list, [mock.call(101, signal.SIGTERM), mock.call(102, signal.SIGTERM)])
        for worker in workers:
            worker.wait.assert_called_once()

    def test_exited_shell_group_is_still_signalled(self):
        workers = [mock.Mock(pid=101, returncode=0), mock.Mock(pid=102)]
        workers[0].poll.return_value = 0
        workers[1].poll.side_effect = KeyboardInterrupt()
        with (
            mock.patch.object(launcher.sys, "argv", ["launch_online_dp.py", *self.base]),
            mock.patch.object(launcher.subprocess, "Popen", side_effect=workers),
            mock.patch.object(launcher.os, "killpg", create=True) as killpg,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(KeyboardInterrupt),
        ):
            launcher.main()
        self.assertEqual(killpg.call_args_list, [mock.call(101, signal.SIGTERM), mock.call(102, signal.SIGTERM)])
        for worker in workers:
            worker.wait.assert_called_once()

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
