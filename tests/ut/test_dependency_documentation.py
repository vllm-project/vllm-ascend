# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DEPENDENCIES = ("torch", "torch-npu", "triton-ascend")
CPU_BUILD_DEPENDENCIES = (
    "torch",
    "torch-npu",
    "torchvision",
    "torchaudio",
    "triton-ascend",
)
MAIN_PACKAGE_VARIABLES = {
    "torch": "main_pytorch_version",
    "torch-npu": "main_torch_npu_version",
    "torchvision": "main_torchvision_version",
    "torchaudio": "main_torchaudio_version",
    "triton-ascend": "main_triton_ascend_version",
}


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _requirements_versions() -> dict[str, str]:
    versions = {}
    for name, version in re.findall(
        r"^(torch|torch-npu|torchvision|torchaudio|triton-ascend)==([^\s;]+)$",
        _read("requirements.txt"),
        flags=re.MULTILINE,
    ):
        versions[name] = version
    return versions


def _pyproject_versions() -> dict[str, str]:
    versions = {}
    for name, version in re.findall(
        r'"(torch|torch-npu|triton-ascend)==([^";]+)"',
        _read("pyproject.toml"),
    ):
        versions[name] = version
    return versions


def _mkdocs_main_versions() -> dict[str, str]:
    mkdocs = _read("mkdocs.yml")
    versions = {}
    for package, variable in MAIN_PACKAGE_VARIABLES.items():
        match = re.search(
            rf'^\s*{variable}:\s*["\']?([^"\'\s#]+)',
            mkdocs,
            flags=re.MULTILINE,
        )
        if match is None:
            return {}
        versions[package] = match.group(1)
    return versions


def _mkdocs_stable_version() -> str:
    mkdocs = _read("mkdocs.yml")
    match = re.search(
        r'^\s*stable_vllm_ascend_version:\s*["\']?([^"\'\s#]+)',
        mkdocs,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else ""


class DependencyDocumentationTest(unittest.TestCase):
    def test_stable_documentation_version_is_explicit(self):
        self.assertTrue(_mkdocs_stable_version())
        main_html = _read("docs/overrides/main.html")
        self.assertIn("config.extra.stable_vllm_ascend_version", main_html)
        self.assertNotIn("config.extra.vllm_ascend_version", main_html)

    def test_main_dependency_versions_match_repository_metadata(self):
        requirements = _requirements_versions()
        core_requirements = {package: requirements[package] for package in CORE_DEPENDENCIES}
        self.assertEqual(set(requirements), set(CPU_BUILD_DEPENDENCIES))
        self.assertEqual(_pyproject_versions(), core_requirements)
        self.assertEqual(_mkdocs_main_versions(), requirements)

    def test_cpu_only_build_contract_is_documented(self):
        installation = _read("docs/source/getting_started/installation.md")
        section_start = installation.index("### CPU-only build verification")
        section_end = installation.index("### Multi-node deployment", section_start)
        cpu_section = installation[section_start:section_end]
        required_text = (
            ".github/vllm-main-verified.commit",
            "python -m venv .venv",
            '"setuptools>=64"',
            '"setuptools-scm>=8"',
            "wheel",
            'distribution("vllm-ascend")',
            "CPU-only package build: OK",
            "skips native extensions and custom kernels",
            "does not provide CPU inference or an environment for running the unit tests",
            "Runtime dependencies are intentionally absent",
            "python -m pip check",
            "../developer_guide/contribution/testing.md#set-up-a-test-environment",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, cpu_section)

        # Check the flags and target on the actual install command, rather than
        # accepting mentions elsewhere in the explanatory text.
        commands = cpu_section.replace("\\\n", " ")
        self.assertRegex(
            commands,
            r"COMPILE_CUSTOM_KERNELS=0\s+SOC_VERSION=\S+\s+"
            r"python -m pip install\s+--no-build-isolation\s+--no-deps\s+-e\s+\.",
        )


if __name__ == "__main__":
    unittest.main()
