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


BUILD_ONLY_DEPENDENCIES = {
    "attrs": (
        "Needed by the arctic-inference build backend inside the PEP 517 isolated "
        "environment; see the CPU-only build verification section of installation.md."
    ),
    "googleapis-common-protos": (
        "Same as attrs: required to build arctic-inference from source when no compatible wheel is available."
    ),
}

RUNTIME_ONLY_DEPENDENCIES = {
    "regex": ("Used by collect_env.py and .github/workflows/scripts/*.py; never imported during the build."),
    "torchaudio": (
        "Pinned so the PyTorch CPU index install stays internally consistent. It is not "
        "imported anywhere under vllm_ascend/ and is not needed to configure or compile."
    ),
    "memfabric-hybrid": (
        "Published on Huawei OBS rather than PyPI (see "
        ".github/workflows/scripts/install_daily_deps.sh). Listing it in "
        "build-system.requires would make every isolated build fail to resolve."
    ),
    "memcache-hybrid": ("Same as memfabric-hybrid: OBS-hosted wheel, not resolvable from a package index."),
}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_requirement(spec: str) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", spec.strip())
    if match is None:
        raise AssertionError(f"Unparsable requirement: {spec!r}")
    return _canonical(match.group(1)), match.group(2).strip()


def _requirements_specs() -> dict[str, str]:
    specs = {}
    for line in _read("requirements.txt").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, constraint = _split_requirement(line)
            specs[name] = constraint
    return specs


def _pyproject_specs() -> dict[str, str]:
    block = re.search(
        r"^\[build-system\].*?^requires\s*=\s*\[(.*?)\]",
        _read("pyproject.toml"),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert block is not None, "Could not locate [build-system] requires in pyproject.toml"
    specs = {}
    for entry in re.findall(r'"([^"]+)"', block.group(1)):
        name, constraint = _split_requirement(entry)
        specs[name] = constraint
    return specs


class DependencyListConsistencyTest(unittest.TestCase):
    """Guards the contract between requirements.txt and build-system.requires.

    The two lists are deliberately *not* identical: one describes the runtime
    environment, the other the PEP 517 build-isolation environment. Every
    asymmetry must be an intentional, documented one.
    """

    def test_shared_dependencies_use_identical_specifiers(self):
        requirements = _requirements_specs()
        pyproject = _pyproject_specs()
        mismatched = {
            name: (pyproject[name], requirements[name])
            for name in sorted(set(pyproject) & set(requirements))
            if pyproject[name] != requirements[name]
        }
        self.assertEqual(
            mismatched,
            {},
            "Packages listed in both files must carry the same version specifier. "
            "Mismatches are reported as {package: (pyproject.toml, requirements.txt)}.",
        )

    def test_build_only_dependencies_are_allowlisted(self):
        only_in_pyproject = sorted(set(_pyproject_specs()) - set(_requirements_specs()))
        self.assertEqual(
            only_in_pyproject,
            sorted(BUILD_ONLY_DEPENDENCIES),
            "A package appears in build-system.requires but not in requirements.txt. "
            "Either add it to requirements.txt, or record it in BUILD_ONLY_DEPENDENCIES "
            "with the reason it is build-time only.",
        )

    def test_runtime_only_dependencies_are_allowlisted(self):
        only_in_requirements = sorted(set(_requirements_specs()) - set(_pyproject_specs()))
        self.assertEqual(
            only_in_requirements,
            sorted(RUNTIME_ONLY_DEPENDENCIES),
            "A package appears in requirements.txt but not in build-system.requires. "
            "Either add it to build-system.requires, or record it in "
            "RUNTIME_ONLY_DEPENDENCIES with the reason it is runtime only.",
        )

    def test_every_allowlisted_asymmetry_has_a_reason(self):
        for package, reason in {**BUILD_ONLY_DEPENDENCIES, **RUNTIME_ONLY_DEPENDENCIES}.items():
            with self.subTest(package=package):
                self.assertTrue(reason.strip(), f"{package} needs a documented reason")


if __name__ == "__main__":
    unittest.main()
