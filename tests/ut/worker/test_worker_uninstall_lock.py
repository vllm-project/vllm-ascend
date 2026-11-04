# SPDX-License-Identifier: Apache-2.0
"""CPU/POSIX regression for static-kernel uninstaller lock ownership.

Run directly: python3 tests/ut/worker/test_worker_uninstall_lock.py
The real flock kernel primitive checks exclusion; os.dup models inherited
descriptors, and a temporary harmless shell script checks real child inheritance.
No NPU runtime or CANN uninstall script is executed.
"""

import ast
import fcntl
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"


def load_uninstall():
    tree = ast.parse(SOURCE.read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
    method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "uninstall_static_kernel")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["uninstall_static_kernel"]


class WorkerUninstallLockTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.kernel_dir = self.root / "opp/static_kernel"
        (self.kernel_dir / "ai_core").mkdir(parents=True)
        (self.kernel_dir / "ai_core/uninstall.sh").touch()
        self.lock_path = self.kernel_dir / "uninstall.lock"
        self.uninstall = load_uninstall()
        self.env = patch.dict(os.environ, {"ASCEND_HOME_PATH": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.hardware = patch.dict(sys.modules, {"torch_npu": MagicMock(), "acl": MagicMock()})
        self.hardware.start()
        self.addCleanup(self.hardware.stop)

    def test_failed_contender_cannot_unlink_active_lock(self):
        with self.lock_path.open("w") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner_inode = os.fstat(owner.fileno()).st_ino
            with patch("subprocess.Popen") as popen:
                self.uninstall(None)
            popen.assert_not_called()
            self.assertTrue(self.lock_path.exists(), "losing worker removed another worker's lock")
            self.assertEqual(self.lock_path.stat().st_ino, owner_inode)
            with self.lock_path.open("w") as contender, self.assertRaises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_background_child_keeps_lock_until_exit(self):
        inherited_fds = []

        def start_child(*args, **kwargs):
            for fd in kwargs.get("pass_fds", ()):
                inherited_fds.append(os.dup(fd))
            return MagicMock()

        try:
            with patch("subprocess.Popen", side_effect=start_child) as popen:
                self.uninstall(None)
                self.uninstall(None)
                self.assertEqual(popen.call_count, 1, "two workers launched concurrent uninstallers")
                self.assertEqual(len(inherited_fds), 1)
                self.assertTrue(self.lock_path.exists())
                with self.lock_path.open("w") as contender, self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.close(inherited_fds.pop())
            with self.lock_path.open("w") as next_owner:
                fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            for fd in inherited_fds:
                os.close(fd)

    def test_missing_script_does_not_create_a_lock_or_spawn(self):
        (self.kernel_dir / "ai_core/uninstall.sh").unlink()
        with patch("subprocess.Popen") as popen:
            self.uninstall(None)
        popen.assert_not_called()
        self.assertFalse(self.lock_path.exists())

    def test_real_child_holds_lock_until_harmless_script_exits(self):
        ready = self.root / "child-ready"
        release = self.root / "allow-child-exit"
        script = self.kernel_dir / "ai_core/uninstall.sh"
        script.write_text(
            f"printf ready > {shlex.quote(str(ready))}\n"
            f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done\n"
        )
        children = []
        real_popen = subprocess.Popen

        def start_child(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        try:
            with patch("subprocess.Popen", side_effect=start_child):
                self.uninstall(None)
            self.assertEqual(len(children), 1)
            child = children[0]
            deadline = time.monotonic() + 5
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "harmless child did not signal readiness")
            self.assertIsNone(child.poll())
            with self.lock_path.open("w") as contender, self.assertRaises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release.touch()
            self.assertEqual(child.wait(timeout=5), 0)
            with self.lock_path.open("w") as next_owner:
                fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.touch()
            for child in children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)

    def test_failed_spawn_releases_the_acquired_lock(self):
        with patch("subprocess.Popen", side_effect=OSError("spawn failed")):
            self.uninstall(None)
        with self.lock_path.open("w") as next_owner:
            fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
