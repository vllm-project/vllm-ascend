# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "csrc" / "build_aclnn.sh"


class TestBuildAclnn(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": str(self.root / "gitconfig"),
            "VLLM_BATCH_INVARIANT": "0",
        }

    def run_build(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), str(self.root), *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_reject_unknown_soc(self):
        for soc in ("", "unknown", "ascend999", "Ascend910B1", "ascend910B1"):
            with self.subTest(soc=soc):
                result = self.run_build(soc)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Unsupported SOC_VERSION", result.stderr)
                self.assertIn("ascend910b", result.stderr)
                self.assertNotIn("skip build_aclnn", result.stdout)
                self.assertFalse((self.root / "gitconfig").exists())

    def test_reject_missing_soc(self):
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Unsupported SOC_VERSION", result.stderr)

    def test_supported_soc_builds_and_installs(self):
        csrc = self.root / "csrc"
        (csrc / "third_party" / "catlass" / "include").mkdir(parents=True)
        (self.root / ".gitmodules").write_text('[submodule "csrc/third_party/catlass"]\ncommit = test\n')
        (csrc / "build.sh").write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" > build-args\n'
            "mkdir -p build\n"
            "cat > build/cann-ops-transformer-test.run <<'EOF'\n"
            '#!/bin/bash\nprintf "%s\\n" "$@" > install-args\n'
            "EOF\n"
        )
        for soc, target in (
            ("ascend310p1", "ascend310p"),
            ("ascend910b1", "ascend910b"),
            ("ascend910_9391", "ascend910_93"),
            ("ascend950", "ascend950"),
            ("ascend950pr_9599", "ascend950"),
        ):
            with self.subTest(soc=soc):
                (csrc / "build-args").unlink(missing_ok=True)
                (csrc / "install-args").unlink(missing_ok=True)
                result = self.run_build(soc)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"--soc={target}\n", (csrc / "build-args").read_text())
                self.assertEqual(
                    (csrc / "install-args").read_text().strip(),
                    f"--install-path={self.root}/vllm_ascend/_cann_ops_custom",
                )


if __name__ == "__main__":
    unittest.main()
