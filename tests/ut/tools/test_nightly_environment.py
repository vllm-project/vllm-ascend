import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.nightly_environment import (
    cann_environment,
    check_baseline,
    new_conflicts,
    validate_pip_check,
    validate_sha,
    web_requirements,
)


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
