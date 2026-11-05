# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.check_wheel_dependencies import check_wheel_dependencies, read_requirements


class WheelDependenciesTest(unittest.TestCase):
    def check_metadata(
        self, dependencies, requirements="torch-npu==2.10.0.post4\nnumpy\n", extra_metadata=(), project=None
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text(requirements, encoding="utf-8")
            if project is not None:
                (root / "pyproject.toml").write_text(project, encoding="utf-8")
            wheel = root / "example.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "vllm_ascend-0.0.0.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: vllm_ascend\nVersion: 0.0.0\n"
                    + "".join(f"Requires-Dist: {dependency}\n" for dependency in dependencies),
                )
                for name in extra_metadata:
                    archive.writestr(name, "Name: embedded\nRequires-Dist: unrelated\n")
            check_wheel_dependencies(wheel, root / ("pyproject.toml" if project is not None else "requirements.txt"))

    def test_static_project_dependencies_are_the_authority(self):
        self.check_metadata(
            ["example>=1"], requirements="stale==99\n", project='[project]\ndependencies = ["example>=1"]\n'
        )

    def test_empty_static_project_dependencies(self):
        self.check_metadata([], project="[project]\ndependencies = []\n")

    def test_legacy_pyproject_uses_requirements_input(self):
        self.check_metadata(["torch-npu==2.10.0.post4", "numpy"], project='[build-system]\nrequires = ["build-only"]\n')

    def test_normalized_names_and_whitespace(self):
        self.check_metadata(["torch_npu == 2.10.0.post4", "numpy"])

    def test_nested_metadata_is_ignored(self):
        self.check_metadata(
            ["torch-npu==2.10.0.post4", "numpy"],
            extra_metadata=("tests/fixtures/embedded-1.0.dist-info/METADATA",),
        )

    def test_multiple_root_metadata_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check_metadata(
                ["torch-npu==2.10.0.post4", "numpy"], extra_metadata=("embedded-1.0.dist-info/METADATA",)
            )

    def test_missing_dependency(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.check_metadata(["numpy"])

    def test_added_numpy_pin(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.check_metadata(["torch-npu==2.10.0.post4", "numpy==1.26.4"])

    def test_duplicate_metadata(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.check_metadata(["torch-npu==2.10.0.post4", "numpy", "numpy"])

    def test_markers_and_extras_are_preserved(self):
        self.check_metadata(
            ['example[one,two]>=1; python_version < "3.13"'],
            'example[two,one]>=1; python_version < "3.13"\n',
        )
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.check_metadata(["example>=1"], 'example>=1; python_version < "3.13"\n')

    def test_invalid_requirement_fails(self):
        with self.assertRaises(ValueError):
            self.check_metadata(["numpy"], "not a valid requirement\n")

    def test_includes_comments_and_index_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "requirements.txt"
            path.write_text("# runtime\n-r common.txt\n\n--extra-index-url https://example.org\n", encoding="utf-8")
            (root / "common.txt").write_text("numpy\n", encoding="utf-8")
            self.assertEqual(read_requirements(path, root), [("numpy", (), "", "", None)])

    def test_inline_comment_preserves_url_fragment(self):
        self.check_metadata(["numpy"], "numpy # compatibility reason\n")
        self.check_metadata(
            ["example @ https://example.org/example.whl#sha256=abc"],
            "example @ https://example.org/example.whl#sha256=abc # provenance\n",
        )


if __name__ == "__main__":
    unittest.main()
