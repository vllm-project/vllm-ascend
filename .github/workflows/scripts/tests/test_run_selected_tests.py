# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Bash runner with a recording pytest/coverage boundary.

Run with python3 -m unittest discover -s .github/workflows/scripts/tests.
These test routing and failure propagation, not connector correctness.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

V1 = "tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py"
PREFILL = "tests/ut/distributed/kv_transfer/kv_p2p/test_remote_prefill_lifecycle.py"
OTHER = "tests/ut/kv_offload/test_mooncake_hybrid_connector.py"
RECORDER = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['ROUTING_CALLS'], 'a') as log:
    log.write(json.dumps({'args': sys.argv[1:], 'coverage': os.environ.get('COVERAGE_FILE')}) + '\\n')
sys.exit(1 if os.environ.get('FAIL_TARGET') in sys.argv[1:] else 0)
"""


class TestSelectedTestsRouting(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / ".github/workflows/scripts/run_selected_tests.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / "run_selected_tests.sh", self.script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("pytest", "python"):
            path = self.bin / name
            path.write_text(RECORDER)
            path.chmod(0o755)
        self.calls = self.root / "calls.jsonl"

    def run_runner(self, targets, *, coverage=False, timing=False, fail=None):
        env = os.environ.copy()
        for key in ("CI", "ENABLE_COVERAGE", "COVERAGE_FILE", "FAIL_TARGET"):
            env.pop(key, None)
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}", RUNNER_TEMP=str(self.root), ROUTING_CALLS=str(self.calls)
        )
        if fail:
            env["FAIL_TARGET"] = fail
        command = ["bash", str(self.script)]
        if coverage:
            command.append("--enable-coverage")
        command += ["cpu", "0", "without-device"]
        if timing:
            command.append("--timing")
        result = subprocess.run(command + targets, cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return result, calls

    def test_v1_only_gets_cpu_contract_mode(self):
        result, calls = self.run_runner([V1])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call["args"] for call in calls], [["-sv", "--color=yes", "--pd-unit", V1]])

    def test_other_connector_keeps_default_mode(self):
        result, calls = self.run_runner([OTHER])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call["args"] for call in calls], [["-sv", "--color=yes", OTHER]])

    def test_mixed_files_are_isolated_without_losing_nodeids(self):
        nodeid = V1 + "::TestUtils::test_ensure_zmq_send_success"
        result, calls = self.run_runner([OTHER, "./" + nodeid])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"], ["-sv", "--color=yes", "--pd-unit", nodeid])
        self.assertEqual(calls[1]["args"], ["-sv", "--color=yes", OTHER])

    def test_full_ut_directory_routes_every_file_once(self):
        for name in (V1, PREFILL, OTHER):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        result, calls = self.run_runner(["tests/ut"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls[0]["args"], ["-sv", "--color=yes", "--pd-unit", V1, PREFILL])
        self.assertEqual(calls[1]["args"], ["-sv", "--color=yes", OTHER])

    def test_v1_directory_does_not_create_an_empty_default_batch(self):
        path = self.root / V1
        path.parent.mkdir(parents=True)
        path.touch()
        result, calls = self.run_runner([str(Path(V1).parent)])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("--pd-unit", calls[0]["args"])

    def test_coverage_files_are_separate_for_each_batch(self):
        result, calls = self.run_runner([V1, OTHER], coverage=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["coverage"], calls[1]["coverage"])
        self.assertEqual(Path(calls[0]["coverage"]).parent.parent.name, "cpu-ut")
        self.assertEqual(Path(calls[1]["coverage"]).parent.parent.name, "cpu-ut")
        self.assertEqual(Path(calls[0]["coverage"]).name, "coverage-mooncake-v1")
        self.assertIn("--pd-unit", calls[0]["args"])
        self.assertNotIn("--pd-unit", calls[1]["args"])

    def test_v1_failure_is_not_hidden_by_a_later_passing_batch(self):
        result, calls = self.run_runner([V1, OTHER], timing=True, fail=V1, coverage=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertTrue((Path(calls[0]["coverage"]).parent / "FAILED").is_file())
        # The assembler consumes the aggregate cpu-ut key. A later passing
        # process must not clear the failed-batch sentinel for that bucket.
        self.assertEqual(Path(calls[0]["coverage"]).parent, Path(calls[1]["coverage"]).parent)
        self.assertTrue((Path(calls[1]["coverage"]).parent / "FAILED").is_file())
        timing = json.loads((self.root / "selected-tests-cpu-0card/test_timing_data.json").read_text())
        self.assertEqual([entry["passed"] for entry in timing], [False, True])

    def test_default_fail_fast_is_preserved(self):
        result, calls = self.run_runner([V1, OTHER], fail=V1)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1)


class TestDefaultUTInitializationOrder(unittest.TestCase):
    @staticmethod
    def watchdog_code():
        path = Path(__file__).resolve().parents[4] / "tests/ut/conftest.py"
        tree = ast.parse(path.read_text())
        hook = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pytest_runtest_protocol"
        )
        hook.decorator_list = []
        return ast.unparse(hook)

    def test_pd_watchdog_cancels_after_teardown_and_does_not_affect_other_ut(self):
        for enabled in (False, True):
            with self.subTest(pd_unit=enabled):
                handler = MagicMock()
                scope = {"faulthandler": handler}
                exec(self.watchdog_code(), scope)
                item = types.SimpleNamespace(config=types.SimpleNamespace(getoption=lambda _, enabled=enabled: enabled))
                protocol = scope["pytest_runtest_protocol"](item, None)
                next(protocol)
                # A test/fixture exception still runs the hook's finally block.
                with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                    protocol.throw(RuntimeError("fixture failed"))
                if enabled:
                    handler.dump_traceback_later.assert_called_once_with(60, exit=True)
                    handler.cancel_dump_traceback_later.assert_called_once_with()
                else:
                    handler.dump_traceback_later.assert_not_called()
                    handler.cancel_dump_traceback_later.assert_not_called()

    def test_pd_watchdog_exits_even_when_executor_teardown_is_blocked(self):
        # Test the actual production hook in a disposable runner process. The
        # PD case remains single-process; this only verifies CI timeout machinery.
        # Shorten the watchdog clock, not the hook's exit/cleanup implementation.
        program = (
            """import faulthandler, threading, types
from concurrent.futures import ThreadPoolExecutor
real_arm = faulthandler.dump_traceback_later
faulthandler.dump_traceback_later = lambda seconds, **kw: real_arm(0.2, **kw)
"""
            + self.watchdog_code()
            + """
item = types.SimpleNamespace(config=types.SimpleNamespace(getoption=lambda _: True))
protocol = pytest_runtest_protocol(item, None)
next(protocol)
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(threading.Event().wait)
    future.result(timeout=0.01)
"""
        )
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Timeout", result.stderr)
        self.assertIn("shutdown", result.stderr)

    def test_model_patches_remain_at_import_time_except_in_explicit_pd_mode(self):
        """Execute the actual startup guard without importing device dependencies.

        A child conftest may import patched APIs before pytest_configure. Moving
        this guard into a hook must fail this regression. Actual dependency
        initialization remains the responsibility of the full UT job.
        """
        path = Path(__file__).resolve().parents[4] / "tests/ut/conftest.py"
        tree = ast.parse(path.read_text())
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "adapt_patch"
                for child in ast.walk(node)
            )
        ]
        self.assertEqual(len(guards), 1, "Default model patches must run while conftest is imported")
        code = compile(ast.Module(body=guards, type_ignores=[]), str(path), "exec")
        for argv, expected in [(["pytest"], [False, True, "customops"]), (["pytest", "--pd-unit"], [])]:
            with self.subTest(argv=argv):
                events = []
                exec(
                    code,
                    {
                        "sys": types.SimpleNamespace(argv=argv),
                        "_npu_available": True,
                        "adapt_patch": lambda worker=False, events=events: events.append(worker),
                        "register_ascend_customop": lambda events=events: events.append("customops"),
                    },
                )
                self.assertEqual(events, expected)


if __name__ == "__main__":
    unittest.main()
