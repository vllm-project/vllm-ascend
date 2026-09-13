import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.nightly_environment import (
    cann_environment,
    check_baseline,
    main,
    new_conflicts,
    validate_pip_check,
    validate_sha,
    web_requirements,
)


class ImageReuseTests(unittest.TestCase):
    def test_dependency_install_flag_cannot_change_server_or_default_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "must-not-exist"
            for extra in (["--runtime-mode", "image-reuse", "--role", "server"], ["--role", "client"]):
                argv = [
                    "nightly_environment",
                    *extra,
                    "--vllm-sha",
                    "a" * 40,
                    "--dep-dir",
                    str(root),
                    "--install-client-dependencies",
                ]
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as failure,
                ):
                    main()
                self.assertEqual(failure.exception.code, 2)
                self.assertFalse(root.exists())

    def test_explicit_client_dependency_install_is_private_and_preserves_image_stack(self):
        self.exercise_client(install=True)

    def test_failed_client_dependency_install_never_reports_ready(self):
        self.exercise_client(install=True, fail_install=True)

    def test_client_only_image_opencv_conflict_is_explicitly_recorded(self):
        self.exercise_client(install=True, client_conflict=True)

    def test_other_new_client_dependency_conflicts_still_prevent_activation(self):
        self.exercise_client(install=True, client_conflict=True, unexpected_conflict=True)

    def test_client_clones_fixed_source_and_missing_cli_dependency_cannot_report_success(self):
        self.exercise_client(install=False, fail_help=True)

    def exercise_client(
        self, *, install, fail_help=False, fail_install=False, client_conflict=False, unexpected_conflict=False
    ):
        from tools import nightly_environment

        source = Path(nightly_environment.__file__).resolve().parents[1]
        requested_vllm = (source / ".github/vllm-main-verified.commit").read_text().strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            shared = Path(directory) / "shared-benchmark"
            shared.mkdir()
            marker = shared / "user-file"
            marker.write_text("shared working tree must remain unchanged")
            calls = []
            packages = {
                "torch": "2.10.0",
                "numpy": "1.26.4",
                "transformers": "5.14.1",
                "opencv-python-headless": "5.0.0.93",
                "Pillow": "12.3.0",
            }
            effective = dict(packages, **{"opencv-python-headless": "4.11.0.86", "Pillow": "11.2.1"})
            known_conflict = (
                "vllm 0.28.0+empty has requirement opencv-python-headless>=4.13.0, "
                "but you have opencv-python-headless 4.11.0.86."
            )

            def check_output(command, **kwargs):
                if command[0] == "bash":
                    return json.dumps({"PATH": "/usr/bin", "PYTHONPATH": str(source)})
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return (nightly_environment.AISBENCH_SHA if "benchmark" in str(command[2]) else "d" * 40) + "\n"
                if "status" in command:
                    return "?? user-file\n" if str(command[2]) == str(shared) else ""
                raise AssertionError(command)

            def run(command, **kwargs):
                command = list(map(str, command))
                calls.append(command)
                if "clone" in command:
                    self.assertEqual(command[-2:], [str(shared), str(root / "benchmark")])
                    (root / "benchmark").mkdir()
                elif "checkout" in command:
                    self.assertEqual(command[-1], nightly_environment.AISBENCH_SHA)
                else:
                    self.assertEqual(Path(kwargs["cwd"]), root / "neutral")
                    if command[-3:] == ["-m", "pip", "check"]:
                        if str(root / "venv") in command[0] and client_conflict:
                            kwargs["stdout"].write(known_conflict + "\n")
                            if unexpected_conflict:
                                kwargs["stdout"].write("example 1.0 requires missing, which is not installed.\n")
                            return subprocess.CompletedProcess(command, 1)
                        kwargs["stdout"].write("No broken requirements found.\n")
                    elif "install" in command:
                        self.assertTrue(install)
                        self.assertIn(str(root / "benchmark") + "[api]", command)
                        self.assertNotIn("torch==2.10.0+cpu", command)
                        self.assertIn(str(root / "venv"), command[0])
                        if fail_install:
                            kwargs["stdout"].write("ResolutionImpossible\n")
                            return subprocess.CompletedProcess(command, 1)
                    elif "venv" in command:
                        self.assertIn("--system-site-packages", command)
                    elif command[-1] == "--help":
                        if fail_help:
                            kwargs["stdout"].write("ModuleNotFoundError: No module named mmengine\n")
                            return subprocess.CompletedProcess(command, 1)
                    elif "-c" in command:
                        if "NIGHTLY_PACKAGES_JSON=" in command[command.index("-c") + 1]:
                            versions = effective if str(root / "venv") in command[0] else packages
                            kwargs["stdout"].write("NIGHTLY_PACKAGES_JSON=" + json.dumps(versions) + "\n")
                        else:
                            actual = {
                                "ais_bench": {
                                    "version": "0.0.0",
                                    "import_path": str(root / "benchmark" / "ais_bench" / "__init__.py"),
                                    "source_root": str(root / "benchmark"),
                                    "git_sha": nightly_environment.AISBENCH_SHA,
                                    "dirty": False,
                                    "git_status": "",
                                }
                            }
                            kwargs["stdout"].write("NIGHTLY_RUNTIME_JSON=" + json.dumps(actual) + "\n")
                    else:
                        raise AssertionError(command)
                return subprocess.CompletedProcess(command, 0)

            argv = [
                "nightly_environment",
                "--runtime-mode",
                "image-reuse",
                "--role",
                "client",
                "--vllm-sha",
                requested_vllm,
                "--dep-dir",
                str(root),
                "--benchmark-source",
                str(shared),
            ]
            if install:
                argv.append("--install-client-dependencies")
            with (
                patch.object(sys, "argv", argv),
                patch.object(subprocess, "check_output", side_effect=check_output),
                patch.object(subprocess, "run", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if fail_help or fail_install or unexpected_conflict:
                    with self.assertRaisesRegex(
                        RuntimeError, "Baseline changed" if unexpected_conflict else "Command failed"
                    ):
                        main()
                else:
                    main()
            self.assertEqual(marker.read_text(), "shared working tree must remain unchanged")
            self.assertEqual(list(shared.iterdir()), [marker])
            report = json.loads((root / "environment-report.json").read_text())
            self.assertFalse(report["runtime_matches_requested"])
            if fail_help or fail_install or unexpected_conflict:
                self.assertFalse(report["cli_verified"])
                self.assertFalse((root / "activate.sh").exists())
            else:
                self.assertTrue(report["cli_verified"])
                self.assertEqual(report["actual"]["ais_bench"]["git_sha"], nightly_environment.AISBENCH_SHA)
                self.assertIn("ais_bench.benchmark.cli.main", (root / "bin" / "ais_bench").read_text())
                self.assertEqual(report["protected_before"], packages)
                self.assertEqual(report["protected_after"], effective)
                self.assertEqual(report["image_after"], packages)
                constraints = (root / "reuse-constraints.txt").read_text()
                self.assertIn("torch==2.10.0", constraints)
                self.assertIn("transformers==5.14.1", constraints)
                self.assertIn("opencv-python-headless==4.11.0.86", constraints)
                self.assertIn("Pillow==11.2.1", constraints)
                if client_conflict:
                    self.assertEqual(report["client_only_conflicts"][0]["message"], known_conflict)
                    self.assertIn("does not run vLLM", report["client_only_conflicts"][0]["reason"])
            if not install:
                self.assertFalse(any("install" in command or "venv" in command for command in calls))

    def test_server_reuses_image_and_reports_actual_runtime_without_installing(self):
        self.exercise_server()

    def test_original_dirty_image_is_reported_without_claiming_requested_runtime(self):
        self.exercise_server(dirty=True)

    def test_image_mode_rejects_import_shadowing_by_requested_pr(self):
        self.exercise_server(shadow=True)

    def exercise_server(self, *, dirty=False, shadow=False):
        from tools import nightly_environment

        source = Path(nightly_environment.__file__).resolve().parents[1]
        requested_vllm = (source / ".github/vllm-main-verified.commit").read_text().strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private"
            calls = []
            actual = {
                name: {
                    "version": "0.1.image",
                    "import_path": str((source if shadow else Path(directory) / "image") / name / "__init__.py"),
                    "source_root": str(Path(directory) / "image"),
                    "git_sha": sha,
                    "dirty": dirty,
                    "git_status": " M image-original-patch.py\n" if dirty else "",
                }
                for name, sha in (("vllm", "2" * 40), ("vllm_ascend", "9" * 40))
            }

            def check_output(command, **kwargs):
                if command[0] == "bash":
                    return json.dumps({"PATH": "/usr/bin", "PYTHONPATH": str(source)})
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return "d" * 40 + "\n"
                if "status" in command:
                    return ""
                raise AssertionError(command)

            def run(command, **kwargs):
                command = list(map(str, command))
                calls.append((command, kwargs))
                self.assertNotIn("install", command)
                self.assertNotIn("clone", command)
                self.assertEqual(Path(kwargs["cwd"]), root / "neutral")
                self.assertNotIn(str(source), kwargs["env"].get("PYTHONPATH", "").split(os.pathsep))
                if command[-3:] == ["-m", "pip", "check"]:
                    kwargs["stdout"].write("No broken requirements found.\n")
                elif command[-1] == "--help":
                    kwargs["stdout"].write("vllm serve options\n")
                elif "-c" in command:
                    kwargs["stdout"].write("NIGHTLY_RUNTIME_JSON=" + json.dumps(actual) + "\n")
                else:
                    raise AssertionError(command)
                return subprocess.CompletedProcess(command, 0)

            args = [
                "nightly_environment",
                "--runtime-mode",
                "image-reuse",
                "--role",
                "server",
                "--vllm-sha",
                requested_vllm,
                "--dep-dir",
                str(root),
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(subprocess, "check_output", side_effect=check_output),
                patch.object(subprocess, "run", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if shadow:
                    with self.assertRaisesRegex(RuntimeError, "shadows requested PR"):
                        main()
                    self.assertFalse((root / "activate.sh").exists())
                    return
                main()
            report = json.loads((root / "environment-report.json").read_text())
            self.assertEqual(report["runtime_mode"], "image-reuse")
            self.assertEqual(report["requested"]["vllm_sha"], requested_vllm)
            self.assertEqual(report["actual"]["vllm"]["git_sha"], "2" * 40)
            self.assertEqual(report["actual"]["vllm"]["dirty"], dirty)
            self.assertFalse(report["runtime_matches_requested"])
            self.assertFalse(report["complete_dependency_solution"])
            self.assertEqual(report["pip_check_baseline"], "No broken requirements found.\n")
            activation_text = (root / "activate.sh").read_text()
            self.assertIn(str(root / "neutral"), activation_text)
            self.assertTrue((root / "bin" / "vllm").is_file())
            self.assertEqual(len(calls), 3)


class BaselineImageTests(unittest.TestCase):
    def test_owner_version_change_does_not_hide_same_existing_conflict(self):
        before = "vllm 0.28.0 has requirement opencv>=4.13, but you have opencv 4.11."
        after = "vllm 0.1.dev1 has requirement opencv>=4.13, but you have opencv 4.11."
        self.assertEqual(new_conflicts(before, after), [])

    def test_new_or_worsened_conflict_is_rejected(self):
        before = "vllm 0.28 has requirement opencv>=4.13, but you have opencv 4.11."
        after = "vllm 0.1 has requirement opencv>=4.13, but you have opencv 4.10."
        self.assertEqual(new_conflicts(before, after), [after])

    def test_success_message_is_not_a_conflict(self):
        self.assertEqual(new_conflicts("", "No broken requirements found."), [])

    def test_pip_failure_is_not_an_existing_dependency_conflict(self):
        for code, text in [(2, "broken pip"), (1, "Traceback: pip module missing"), (0, "unrecognized output")]:
            with self.assertRaises(RuntimeError):
                validate_pip_check(code, text)
        validate_pip_check(0, "No broken requirements found.")
        validate_pip_check(1, "vllm 0.28 has requirement opencv>=4.13, but you have opencv 4.11.")

    def test_protected_stack_must_not_change(self):
        with self.assertRaisesRegex(RuntimeError, "torch"):
            check_baseline({"torch": "2.10.0+cpu"}, {"torch": "2.11"}, "", "")

    def test_only_full_frozen_sha_accepted(self):
        self.assertEqual(validate_sha("a" * 40), "a" * 40)
        for value in ["main", "a" * 39, "../repo", "A" * 40]:
            with self.assertRaises(ValueError):
                validate_sha(value)

    def test_web_requirements_keep_markers_not_inline_comments(self):
        text = (
            "fastapi[standard] >= 0.133, < 0.137 # explanation\n"
            "starlette >= 1.0.1 # CVE\n"
            'setuptools>=77; python_version > "3.11" # note\nnumpy\n'
        )
        self.assertEqual(
            web_requirements(text),
            ["fastapi[standard] >= 0.133, < 0.137", "starlette >= 1.0.1", 'setuptools>=77; python_version > "3.11"'],
        )

    @unittest.skipUnless(shutil.which("bash"), "vendor environment loader requires Bash")
    def test_vendor_script_failure_stops_loading_in_real_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            cann = Path(directory) / "cann.sh"
            atb = Path(directory) / "atb.sh"
            with (
                patch("tools.nightly_environment.CANN_SCRIPTS", (cann.as_posix(),)),
                patch("tools.nightly_environment.ATB_SCRIPT", atb.as_posix()),
            ):
                cann.write_text("return 23\n")
                with self.assertRaises(subprocess.CalledProcessError):
                    cann_environment()
                cann.write_text("export NIGHTLY_TEST_CANN_LOADED=1\n")
                atb.write_text("return 24\n")
                with self.assertRaises(subprocess.CalledProcessError):
                    cann_environment()
                atb.write_text("export NIGHTLY_TEST_ATB_LOADED=1\n")
                environment = cann_environment()
                self.assertEqual(environment["NIGHTLY_TEST_CANN_LOADED"], "1")
                self.assertEqual(environment["NIGHTLY_TEST_ATB_LOADED"], "1")


if __name__ == "__main__":
    unittest.main()
