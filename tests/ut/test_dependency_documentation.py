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
    torch_pair = re.search(
        r'^\s*main_pytorch_torch_npu_version:\s*["\']?([^"\'\n]+)',
        mkdocs,
        flags=re.MULTILINE,
    )
    triton = re.search(
        r'^\s*main_triton_ascend_version:\s*["\']?([^"\'\s#]+)',
        mkdocs,
        flags=re.MULTILINE,
    )
    if torch_pair is None or triton is None:
        return {}
    torch_version, torch_npu_version = (value.strip() for value in torch_pair.group(1).split("/"))
    return {
        "torch": torch_version,
        "torch-npu": torch_npu_version,
        "triton-ascend": triton.group(1),
    }


class DependencyDocumentationTest(unittest.TestCase):
    def test_main_dependency_versions_match_repository_metadata(self):
        requirements = _requirements_versions()
        core_requirements = {package: requirements[package] for package in CORE_DEPENDENCIES}
        self.assertEqual(set(requirements), set(CPU_BUILD_DEPENDENCIES))
        self.assertEqual(_pyproject_versions(), core_requirements)
        self.assertEqual(_mkdocs_main_versions(), core_requirements)

    def test_cpu_only_build_contract_is_documented(self):
        installation = _read("docs/source/installation.md")
        section_start = installation.index("### CPU-only build verification")
        section_end = installation.index("\n!!! note", section_start)
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

        requirements = _requirements_versions()
        for package in CPU_BUILD_DEPENDENCIES:
            with self.subTest(package=package):
                self.assertIn(f"{package}=={requirements[package]}", cpu_section)

    def test_ascend_toolkit_home_is_set_before_nnal_installation(self):
        installation = _read("docs/source/installation.md")
        manual_install_start = installation.index("??? \"Click here to see 'Install CANN manually'\"")
        manual_install_end = installation.index('=== "Before using docker"', manual_install_start)
        manual_install = installation[manual_install_start:manual_install_end]
        export_position = manual_install.index("export ASCEND_TOOLKIT_HOME=")
        nnal_install = re.search(r"^\s*\./Ascend-cann-nnal_[^\n]+\.run --install$", manual_install, flags=re.MULTILINE)
        assert nnal_install is not None
        nnal_install_position = nnal_install.start()
        self.assertLess(export_position, nnal_install_position)


if __name__ == "__main__":
    unittest.main()
