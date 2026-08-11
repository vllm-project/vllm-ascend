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


def _mkdocs_scalar(name: str) -> str:
    match = re.search(
        rf'^\s*{re.escape(name)}:\s*["\']?([^"\'\n#]+)',
        _read("mkdocs.yml"),
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1).strip()


def _release_matrix_row(version: str) -> list[str]:
    policy = _read("docs/source/community/versioning_policy.md")
    match = re.search(rf"^\|\s*{re.escape(version)}\s*\|(.+)$", policy, flags=re.MULTILINE)
    assert match is not None, version
    return [version, *(cell.strip() for cell in match.group(1).split("|") if cell.strip())]


class DependencyDocumentationTest(unittest.TestCase):
    def test_main_dependency_versions_match_repository_metadata(self):
        requirements = _requirements_versions()
        core_requirements = {package: requirements[package] for package in CORE_DEPENDENCIES}
        self.assertEqual(set(requirements), set(CPU_BUILD_DEPENDENCIES))
        self.assertEqual(_pyproject_versions(), core_requirements)
        self.assertEqual(_mkdocs_main_versions(), core_requirements)

    def test_default_release_stack_matches_compatibility_matrix(self):
        vllm_ascend = _mkdocs_scalar("vllm_ascend_version")
        row = _release_matrix_row(vllm_ascend)
        expected = [
            vllm_ascend,
            _mkdocs_scalar("vllm_version"),
            _mkdocs_scalar("release_python_version"),
            _mkdocs_scalar("release_cann_version"),
            _mkdocs_scalar("release_pytorch_torch_npu_version"),
            _mkdocs_scalar("release_triton_ascend_version"),
        ]
        self.assertEqual(row[:6], expected)
        self.assertEqual(vllm_ascend.removeprefix("v"), _mkdocs_scalar("pip_vllm_ascend_version"))
        self.assertEqual(
            _mkdocs_scalar("vllm_version").removeprefix("v"),
            _mkdocs_scalar("pip_vllm_version"),
        )

    def test_release_cann_paths_use_release_stack(self):
        software_stack = _read("docs/source/getting_started/software_stack.inc.md")
        install_cann = _read("docs/source/getting_started/installation/install_cann.inc.md")

        self.assertIn("{{ release_cann_version }}", software_stack)
        self.assertIn("{{ release_cann_version }}", install_cann)
        self.assertNotIn("{{ main_cann_version }}", install_cann)

    def test_getting_started_uses_one_release_baseline(self):
        getting_started = REPO_ROOT / "docs/source/getting_started"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(getting_started.rglob("*.md"))
            if not path.is_relative_to(getting_started / "locale") and not path.is_relative_to(getting_started / "zh")
        )

        self.assertNotIn("v0.23.0rc1", source)
        self.assertNotIn(
            "separate software version matrices",
            source.replace(
                "It does not maintain separate software version matrices for different hardware products.",
                "",
            ),
        )
        self.assertIn("{{ vllm_ascend_version }}", source)
        self.assertIn("{{ vllm_version }}", source)

    def test_quickstart_structure_and_shared_flows(self):
        quickstart = _read("docs/source/getting_started/quick_start.md")
        prebuilt = _read("docs/source/getting_started/installation/prebuilt_image.inc.md")
        image_matrix_include = '{% include "getting_started/image_matrix.inc.md" %}'
        self.assertEqual(quickstart.count(image_matrix_include), 0)
        self.assertEqual(prebuilt.count(image_matrix_include), 1)

        hardware_files = (
            "atlas-a2.inc.md",
            "atlas-a3.inc.md",
            "atlas-300i-duo.inc.md",
            "atlas-200i-pro.inc.md",
            "atlas-950dt.inc.md",
        )
        for hardware_file in hardware_files:
            include = '{% include "getting_started/quick_start/' + hardware_file + '" %}'
            with self.subTest(hardware_file=hardware_file):
                self.assertEqual(quickstart.count(include), 1)

        container_verification = '{% include "getting_started/quick_start/container_verification.inc.md" %}'
        qwen3_inference = '{% include "getting_started/quick_start/qwen3_inference.inc.md" %}'
        qwen35_serving = '{% include "getting_started/quick_start/qwen35_serving.inc.md" %}'

        for hardware_file in hardware_files:
            hardware = _read(f"docs/source/getting_started/quick_start/{hardware_file}")
            self.assertEqual(hardware.count(container_verification), 1)
            self.assertIn("docker pull", hardware)
            self.assertIn("{{ vllm_ascend_version }}", hardware)

        for hardware_file in hardware_files[:2]:
            hardware = _read(f"docs/source/getting_started/quick_start/{hardware_file}")
            self.assertEqual(hardware.count(qwen3_inference), 1)

        for hardware_file in hardware_files[2:4]:
            hardware = _read(f"docs/source/getting_started/quick_start/{hardware_file}")
            self.assertEqual(hardware.count(qwen35_serving), 1)

        self.assertIn(
            "torch.npu.is_available()", _read("docs/source/getting_started/quick_start/container_verification.inc.md")
        )
        self.assertIn(
            '=== "Offline batch inference"', _read("docs/source/getting_started/quick_start/qwen3_inference.inc.md")
        )
        qwen35 = _read("docs/source/getting_started/quick_start/qwen35_serving.inc.md")
        self.assertIn("Qwen/Qwen3.5-2B", qwen35)
        self.assertIn('=== "Offline batch inference"', qwen35)
        self.assertIn('=== "Online serving"', qwen35)
        self.assertIn("Currently unsupported", qwen35)

        model_download = _read("docs/source/getting_started/quick_start/model_download.inc.md")
        self.assertIn("VLLM_USE_MODELSCOPE=True", model_download)
        self.assertIn("modelscope>=1.18.1,<1.38", model_download)

    def test_getting_started_navigation_and_prerequisites(self):
        quickstart = _read("docs/source/getting_started/quick_start.md")
        installation = _read("docs/source/getting_started/installation.md")
        qwen3 = _read("docs/source/getting_started/quick_start/qwen3_inference.inc.md")
        cann = _read("docs/source/getting_started/installation/cann_environment.inc.md")
        install_cann = _read("docs/source/getting_started/installation/install_cann.inc.md")
        hardware_source = "\n".join(
            _read(f"docs/source/getting_started/quick_start/{name}")
            for name in (
                "atlas-a2.inc.md",
                "atlas-a3.inc.md",
                "atlas-300i-duo.inc.md",
                "atlas-200i-pro.inc.md",
                "atlas-950dt.inc.md",
            )
        )

        software_stack_include = '{% include "getting_started/software_stack.inc.md" %}'
        self.assertEqual(quickstart.count(software_stack_include), 1)
        self.assertEqual(installation.count(software_stack_include), 1)
        self.assertIn("release-compatibility-matrix", quickstart)
        self.assertIn("release-compatibility-matrix", installation)
        self.assertIn("https://docs.docker.com/engine/install/", quickstart)
        self.assertIn("https://docs.docker.com/engine/install/", installation)

        for anchor in (
            "quickstart-atlas-a2-container",
            "quickstart-atlas-a3-container",
            "quickstart-atlas-300i-duo-container",
            "quickstart-atlas-200i-pro-container",
            "quickstart-atlas-950dt-container",
        ):
            self.assertIn(anchor, hardware_source)

        self.assertIn("vi example.py", qwen3)
        self.assertIn(":wq", qwen3)
        self.assertIn("CANN base image", cann)
        self.assertIn("910b-ubuntu22.04-py3.12", cann)
        self.assertIn("a3-ubuntu22.04-py3.12", cann)
        self.assertIn("950-ubuntu22.04-py3.12", cann)
        self.assertIn("Ascend-cann-nnal_", install_cann)
        self.assertIn("$CANN_NNAL_RUN", install_cann)

    def test_installation_has_three_end_to_end_paths(self):
        installation = _read("docs/source/getting_started/installation.md")
        prebuilt = _read("docs/source/getting_started/installation/prebuilt_image.inc.md")
        cann = _read("docs/source/getting_started/installation/cann_environment.inc.md")
        base = _read("docs/source/getting_started/installation/base_environment.inc.md")

        path_includes = (
            "prebuilt_image.inc.md",
            "cann_environment.inc.md",
            "base_environment.inc.md",
        )
        for path_include in path_includes:
            include = '{% include "getting_started/installation/' + path_include + '" %}'
            with self.subTest(path_include=path_include):
                self.assertEqual(installation.count(include), 1)

        self.assertTrue(prebuilt.startswith('=== "Pre-built image (recommended)"'))
        self.assertTrue(cann.startswith('=== "Existing CANN environment"'))
        self.assertTrue(base.startswith('=== "Base environment"'))
        self.assertIn("Completion Criteria", prebuilt)
        self.assertIn("Completion Criteria", cann)
        self.assertIn("Completion Criteria", base)

        install_include = '{% include "getting_started/installation/install_vllm_ascend.inc.md" %}'
        verify_include = '{% include "getting_started/installation/verify_installation.inc.md" %}'
        cann_include = '{% include "getting_started/installation/install_cann.inc.md" %}'
        self.assertEqual(cann.count(install_include), 1)
        self.assertEqual(cann.count(verify_include), 1)
        self.assertEqual(base.count(cann_include), 1)
        self.assertEqual(base.count(install_include), 1)
        self.assertEqual(base.count(verify_include), 1)

        for path in (cann, base):
            self.assertNotIn("return to the previous section", path.lower())
            self.assertNotIn("continue reading the sections below on this page", path.lower())

        for shared_fragment in (
            "install_cann.inc.md",
            "install_vllm_ascend.inc.md",
            "verify_installation.inc.md",
        ):
            shared = _read("docs/source/getting_started/installation/" + shared_fragment)
            for line_number, line in enumerate(shared.splitlines(), start=1):
                if line:
                    with self.subTest(
                        shared_fragment=shared_fragment,
                        line_number=line_number,
                    ):
                        self.assertTrue(line.startswith("    "), line)

    def test_wheel_and_source_installation_contract(self):
        installation = _read("docs/source/getting_started/installation/install_vllm_ascend.inc.md")

        self.assertIn('=== "Standard pip wheel"', installation)
        self.assertIn("Standard pip installation currently supports Atlas A2 only", installation)
        self.assertIn('assert __device_type__ == "A2"', installation)
        self.assertIn('=== "uv-wheelnext"', installation)
        self.assertIn("uv pip install --system", installation)
        self.assertIn(
            "--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            installation,
        )
        self.assertIn(
            "https://mirrors.huaweicloud.com/ascend/repos/pypi/variant",
            installation,
        )
        self.assertIn("from vllm_ascend._build_info import __device_type__", installation)
        self.assertIn("--branch {{ vllm_ascend_version }}", installation)
        self.assertIn("--branch {{ vllm_version }}", installation)
        self.assertIn("VLLM_TARGET_DEVICE=empty pip install", installation)
        self.assertIn("pip install --no-deps --no-build-isolation -e .", installation)

        wheel_start = installation.index('=== "Standard pip wheel"')
        source_start = installation.index('=== "Source installation"')
        wheel_section = installation[wheel_start:source_start]
        self.assertNotIn("pip uninstall -y triton triton-ascend", wheel_section)

        plugin_install = installation.index("pip install --no-deps --no-build-isolation -e .")
        triton_cleanup = installation.index("pip uninstall -y triton triton-ascend")
        triton_install = installation.index(
            '"$TRITON_ASCEND_REQUIREMENT"',
            triton_cleanup,
        )
        self.assertLess(plugin_install, triton_cleanup)
        self.assertLess(triton_cleanup, triton_install)

    def test_installation_verification_stops_before_model_inference(self):
        verification = _read("docs/source/getting_started/installation/verify_installation.inc.md")

        self.assertIn("PYTHONPATH= pip check", verification)
        self.assertIn("torch.npu.is_available()", verification)
        self.assertIn("NPU tensor operation: PASS", verification)
        self.assertIn("vLLM Ascend plugin", verification)
        self.assertNotIn("from vllm import LLM", verification)
        self.assertNotIn("Qwen/Qwen3-0.6B", verification)

    def test_cpu_only_build_contract_is_documented(self):
        cpu_build = _read("docs/source/getting_started/installation/cpu_only_build.inc.md")
        required_text = (
            "CPU-only build",
            "COMPILE_CUSTOM_KERNELS=0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
            "SOC_VERSION=ascend910b1",
            "SOC_VERSION=ascend910_9391",
            "SOC_VERSION=ascend310p1",
            "--no-build-isolation",
            '"setuptools>=64"',
            '"setuptools-scm>=8"',
            "attrs",
            "googleapis-common-protos",
            "pip check",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, cpu_build)

    def test_advanced_installation_topics_are_included_once(self):
        installation = _read("docs/source/getting_started/installation.md")
        for fragment in (
            "cpu_only_build.inc.md",
            "multi_node.inc.md",
            "troubleshooting.inc.md",
        ):
            include = '{% include "getting_started/installation/' + fragment + '" %}'
            with self.subTest(fragment=fragment):
                self.assertEqual(installation.count(include), 1)

        troubleshooting = _read("docs/source/getting_started/installation/troubleshooting.inc.md")
        self.assertIn("modelscope>=1.18.1,<1.38", troubleshooting)
        self.assertIn("Device Type Mismatch After pip / Wheel Installation", troubleshooting)


if __name__ == "__main__":
    unittest.main()
