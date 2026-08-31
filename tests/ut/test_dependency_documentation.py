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
ARCTIC_INFERENCE_REQUIREMENT = "arctic-inference==0.1.1"


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
        cpu_only_script = _read("tools/cpu_only_build.sh")
        cpu_only_dockerfile = _read("Dockerfile.cpu")
        cpu_only_workflow = _read(".github/workflows/cpu_only_build.yaml")
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
            ARCTIC_INFERENCE_REQUIREMENT,
            "part of the normal vLLM Ascend dependency",
            "bash tools/cpu_only_build.sh all",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, cpu_section)

        requirements = _requirements_versions()
        for package in CPU_BUILD_DEPENDENCIES:
            with self.subTest(package=package):
                self.assertIn(f"{package}=={requirements[package]}", cpu_section)

        self.assertIn(ARCTIC_INFERENCE_REQUIREMENT, _read("requirements.txt"))
        self.assertIn("write_cpu_only_requirements", cpu_only_script)
        self.assertIn('grep -Fvx "${ARCTIC_INFERENCE_REQUIREMENT}"', cpu_only_script)
        self.assertIn("trap restore_requirements EXIT", cpu_only_script)
        self.assertIn('--extra-index-url "${PYTORCH_CPU_INDEX_URL}"', cpu_only_script)
        self.assertIn("torch==2.10.0+cpu", cpu_only_script)
        self.assertIn("RUN bash tools/cpu_only_build.sh verify", cpu_only_dockerfile)
        self.assertIn('ARG CPU_BASE_IMAGE="openeuler/openeuler:24.03-lts"', cpu_only_dockerfile)
        self.assertIn("FROM ${CPU_BASE_IMAGE}", cpu_only_dockerfile)
        self.assertIn("uses: docker/build-push-action@v7", cpu_only_workflow)
        self.assertIn("swr.cn-north-4.myhuaweicloud.com/ddn-k8s/", cpu_only_workflow)
        self.assertIn("DNF_MIRROR=http://cache-service.nginx-pypi-cache", cpu_only_workflow)
        self.assertNotIn("docker run", cpu_only_workflow)

    def test_arctic_inference_contract_is_documented_for_suffix_decoding(self):
        for path in (
            "docs/source/tutorials/features/suffix_speculative_decoding.md",
            "docs/source/user_guide/feature_guide/speculative_decoding.md",
        ):
            with self.subTest(path=path):
                documentation = _read(path)
                self.assertIn(ARCTIC_INFERENCE_REQUIREMENT, documentation)
                self.assertNotIn("arctic-inference>=0.2.0", documentation)

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
