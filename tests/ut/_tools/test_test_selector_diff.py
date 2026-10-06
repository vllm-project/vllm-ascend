# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise oversized PR downloads without GitHub or accelerator dependencies."""

import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


class TestPRDiffDownload(unittest.TestCase):
    selector: ModuleType

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/test_selector.py"
        spec = importlib.util.spec_from_file_location("test_selector_diff", path)
        assert spec is not None and spec.loader is not None
        cls.selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.selector)

    def test_api_success_does_not_fetch_git(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"complete diff"
        with (
            patch.object(self.selector.urllib.request, "urlopen", return_value=response),
            patch.object(self.selector, "_git_pr_diff") as fallback,
        ):
            self.assertEqual(
                self.selector._download_pr_diff(None, None, {"base": {"sha": "base"}}),
                (b"complete diff", "base"),
            )
        fallback.assert_not_called()

    def test_only_explicit_size_limit_uses_git(self):
        pr = {"base": {"sha": "base", "repo": {"clone_url": "repo"}}, "head": {"sha": "head"}}
        for status, code in ((406, "too_large"), (406, "other"), (403, "too_large"), (500, "other")):
            with self.subTest(status=status, code=code):
                error = urllib.error.HTTPError(
                    "https://api.github.com/test",
                    status,
                    "failure",
                    Message(),
                    io.BytesIO(json.dumps({"errors": [{"code": code}]}).encode()),
                )
                with (
                    patch.object(self.selector.urllib.request, "urlopen", side_effect=error),
                    patch.object(self.selector, "_git_pr_diff", return_value=(b"diff", "ancestor")) as fallback,
                ):
                    if status == 406 and code == "too_large":
                        self.assertEqual(self.selector._download_pr_diff(None, None, pr), (b"diff", "ancestor"))
                        fallback.assert_called_once_with("repo", "base", "head")
                    else:
                        with self.assertRaises(urllib.error.HTTPError):
                            self.selector._download_pr_diff(None, None, pr)
                        fallback.assert_not_called()

    def test_shallow_checkout_preserves_large_diff_renames_and_deletions(self):
        def git(directory, *args):
            return (
                subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.PIPE).decode().strip()
            )

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / "source"
            source.mkdir()
            git(source, "init", "-b", "main")
            git(source, "config", "user.name", "Test")
            git(source, "config", "user.email", "test@example.com")
            (source / "old.py").write_text("original = 1\n", encoding="utf-8")
            (source / "deleted.py").write_text("deleted = 1\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "base")
            ancestor = git(source, "rev-parse", "HEAD")
            git(source, "checkout", "-b", "feature")
            git(source, "mv", "old.py", "renamed.py")
            git(source, "rm", "deleted.py")
            (source / "large.py").write_text("value = 1\n" * 21000, encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "large change")
            head = git(source, "rev-parse", "HEAD")
            git(source, "checkout", "main")
            (source / "unrelated.py").write_text("unrelated = 1\n", encoding="utf-8")
            git(source, "add", ".")
            git(source, "commit", "-m", "advance main")
            base = git(source, "rev-parse", "HEAD")
            checkout = root / "checkout"
            git(root, "clone", "--depth=1", source.as_uri(), str(checkout))
            previous = os.getcwd()
            try:
                os.chdir(checkout)
                diff, merge_base = self.selector._git_pr_diff(source.as_uri(), base, head)
            finally:
                os.chdir(previous)
            self.assertEqual(merge_base, ancestor)
            self.assertEqual(diff.count(b"+value = 1"), 21000)
            self.assertIn(b"rename from old.py", diff)
            self.assertIn(b"rename to renamed.py", diff)
            self.assertIn(b"deleted file mode", diff)
            self.assertNotIn(b"unrelated.py", diff)


if __name__ == "__main__":
    unittest.main()
