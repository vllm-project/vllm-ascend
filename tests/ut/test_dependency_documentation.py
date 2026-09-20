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
            "### CPU-only build verification",
            ".github/vllm-main-verified.commit",
            "COMPILE_CUSTOM_KERNELS=0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
            "SOC_VERSION=",
            "--no-build-isolation",
            "https://download.pytorch.org/whl/cpu/",
            '"setuptools>=64"',
            '"setuptools-scm>=8"',
            "attrs",
            "googleapis-common-protos",
            "wheel",
            "ninja",
            "python -m pip check",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, cpu_section)

        for package, variable in MAIN_PACKAGE_VARIABLES.items():
            with self.subTest(package=package):
                self.assertIn(f"{package}=={{{{ {variable} }}}}", cpu_section)


if __name__ == "__main__":
    unittest.main()
