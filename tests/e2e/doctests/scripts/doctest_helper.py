#!/usr/bin/env python3

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Extract registered documentation blocks or preview the fixed CI plan.
Usage from the repository root (these commands do not execute the blocks):
  python3 tests/e2e/doctests/scripts/doctest_helper.py extract MARKER
  python3 tests/e2e/doctests/scripts/doctest_helper.py extract --ref REF MARKER
  python3 tests/e2e/doctests/scripts/doctest_helper.py extract --expand-macros MARKER
  python3 tests/e2e/doctests/scripts/doctest_helper.py plan [selection] [image repositories]
Raw extraction preserves macros; expansion and planning require PyYAML.
Diff planning compares Git refs but uses working-tree MkDocs values for image tags.
Manual selections default to none; non-none choices cannot be combined with diff selection.
Use plan --check-resources for PR planning; explicit manual runs remain strict.
Run the actual tests with scripts/run_doctests.sh under tests/e2e/doctests/.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[4]
MKDOCS_PATH = "mkdocs.yml"
# Matches a doctest marker comment, for example:
# <!-- doctest: quickstart-modelscope -->
DOCTEST_MARKER_RE = re.compile(r"^[ \t]*<!--\s*doctest:\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*-->[ \t]*$")
# Matches the canonical opening fence: ```bash or ```python.
# Do not allow a space between ``` and the language name.
DOCTEST_CODE_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)```(?:bash|python)[ \t]*$")
# Matches a simple MkDocs macro backed by mkdocs.yml -> extra,
# for example: {{ vllm_ascend_version }}
MKDOCS_EXTRA_MACRO_RE = re.compile(r"{{\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*}}")

DOCTEST_OSES = ("ubuntu", "openeuler")
INSTALLATION_OS_IMAGE_TAGS = {
    "ubuntu": "ubuntu22.04",
    "openeuler": "openeuler24.03",
}
VLLM_ASCEND_REPOSITORY_URL = "https://github.com/vllm-project/vllm-ascend.git"
REGISTRY_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

# Fixed marker locations; adding a marker does not automatically add a runtime step.
DOCTEST_MARKERS_BY_FILE = {
    "docs/source/getting_started/quick_start.md": ("quickstart-modelscope",),
    "docs/source/getting_started/quick_start/ascend_image/verify_container.inc.md": ("quickstart-container-verify",),
    "docs/source/getting_started/quick_start/ascend_image/atlas-a2.inc.md": (
        "quickstart-image-a2-ubuntu",
        "quickstart-image-a2-openeuler",
    ),
    "docs/source/getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md": (
        "quickstart-image-300i-duo-ubuntu",
        "quickstart-image-300i-duo-openeuler",
    ),
    "docs/source/getting_started/quick_start/offline/qwen3-0.6b.inc.md": (
        "quickstart-standard-offline",
        "quickstart-standard-offline-run",
    ),
    "docs/source/getting_started/quick_start/online/qwen3-0.6b.inc.md": (
        "quickstart-standard-online-serve",
        "quickstart-standard-online-model-list",
        "quickstart-standard-online-completion",
        "quickstart-standard-online-stop",
    ),
    "docs/source/getting_started/quick_start/offline/qwen3-0.6b-310p.inc.md": (
        "quickstart-300i-duo-offline",
        "quickstart-300i-duo-offline-run",
    ),
    "docs/source/getting_started/quick_start/online/qwen3-0.6b-310p.inc.md": (
        "quickstart-300i-duo-online-serve",
        "quickstart-300i-duo-online-model-list",
        "quickstart-300i-duo-online-completion",
        "quickstart-300i-duo-online-stop",
    ),
    "docs/source/getting_started/installation/cann_image/atlas-a2.inc.md": (
        "installation-cann-image-a2-ubuntu",
        "installation-cann-image-a2-openeuler",
    ),
    "docs/source/getting_started/installation/install_vllm_ascend.inc.md": (
        "installation-common-prerequisites-ubuntu",
        "installation-common-prerequisites-openeuler",
        "installation-pip-install",
        "installation-pip-device-check",
        "installation-uv-bootstrap",
        "installation-uv-install",
        "installation-uv-device-check",
        "installation-source-install",
        "installation-post-standard",
    ),
}
DOCTEST_FILE_BY_MARKER = {marker: path for path, markers in DOCTEST_MARKERS_BY_FILE.items() for marker in markers}

# Groups describe which cases a block affects, independently of its file location.
QUICKSTART_COMMON_MARKERS = (
    "quickstart-modelscope",
    "quickstart-container-verify",
)

QUICKSTART_A2_MARKERS = (
    "quickstart-image-a2-ubuntu",
    "quickstart-image-a2-openeuler",
    "quickstart-standard-offline",
    "quickstart-standard-offline-run",
    "quickstart-standard-online-serve",
    "quickstart-standard-online-model-list",
    "quickstart-standard-online-completion",
    "quickstart-standard-online-stop",
)

QUICKSTART_310P_MARKERS = (
    "quickstart-image-300i-duo-ubuntu",
    "quickstart-image-300i-duo-openeuler",
    "quickstart-300i-duo-offline",
    "quickstart-300i-duo-offline-run",
    "quickstart-300i-duo-online-serve",
    "quickstart-300i-duo-online-model-list",
    "quickstart-300i-duo-online-completion",
    "quickstart-300i-duo-online-stop",
)

INSTALLATION_COMMON_MARKERS = (
    "installation-cann-image-a2-ubuntu",
    "installation-cann-image-a2-openeuler",
    "installation-common-prerequisites-ubuntu",
    "installation-common-prerequisites-openeuler",
    "installation-post-standard",
)

INSTALLATION_PIP_MARKERS = (
    "installation-pip-install",
    "installation-pip-device-check",
)

INSTALLATION_UV_MARKERS = (
    "installation-uv-bootstrap",
    "installation-uv-install",
    "installation-uv-device-check",
)

INSTALLATION_SOURCE_MARKERS = ("installation-source-install",)

SHARED_DOCTEST_PATHS = {
    ".github/workflows/schedule_doctest.yaml",
    "tests/e2e/doctests/scripts/common.sh",
    "tests/e2e/doctests/scripts/run_doctests.sh",
    "tests/e2e/doctests/scripts/doctest_helper.py",
}
QUICKSTART_TEST_SCRIPT = "tests/e2e/doctests/001-quickstart-test.sh"
INSTALLATION_TEST_SCRIPT = "tests/e2e/doctests/002-installation-test.sh"


class DoctestError(ValueError):
    pass


def extract_doctest_block(text: str, marker: str, source: str = "input") -> str | None:
    """Extract one marked Bash/Python block, or return None when the marker is absent."""
    lines = text.splitlines()
    marker_lines = [
        index
        for index, line in enumerate(lines)
        if (match := DOCTEST_MARKER_RE.match(line)) and match.group(1) == marker
    ]
    if not marker_lines:
        return None
    if len(marker_lines) > 1:
        raise DoctestError(f"Duplicate doctest marker '{marker}' in {source}.")

    opening_index = marker_lines[0] + 1
    while opening_index < len(lines) and not lines[opening_index].strip():
        opening_index += 1
    opening_fence = DOCTEST_CODE_FENCE_RE.match(lines[opening_index]) if opening_index < len(lines) else None
    if opening_fence is None:
        raise DoctestError(
            f"Doctest marker '{marker}' in {source} must be followed by a ```bash or ```python code block."
        )

    indent = opening_fence.group("indent")
    body_lines: list[str] = []
    for line in lines[opening_index + 1 :]:
        if line.strip() == "```":
            content = "\n".join(line[len(indent) :] if line.startswith(indent) else line for line in body_lines)
            return content + ("\n" if body_lines else "")
        body_lines.append(line)
    raise DoctestError(f"Code block for doctest marker '{marker}' in {source} is not closed.")


def parse_mkdocs_extra(text: str) -> dict[str, str]:
    """Parse scalar MkDocs extra values without reading files or resolving custom YAML tags."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise DoctestError("PyYAML is required to read mkdocs.yml.") from error

    try:
        data = yaml.load(text, Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        raise DoctestError(f"Cannot parse mkdocs.yml: {error}") from error
    if not isinstance(data, dict):
        raise DoctestError("mkdocs.yml does not contain a mapping.")
    mkdocs_extra = data.get("extra")
    if not isinstance(mkdocs_extra, dict):
        raise DoctestError("mkdocs.yml does not contain an extra mapping.")
    return {key: value for key, value in mkdocs_extra.items() if isinstance(key, str) and isinstance(value, str)}


def extract_release_values(mkdocs_extra: dict[str, str]) -> dict[str, str]:
    """Select the release-stack values whose changes require broader doctest coverage."""
    return {
        key: value
        for key, value in mkdocs_extra.items()
        if key.startswith("release_") or key in {"vllm_version", "vllm_ascend_version"}
    }


def release_config_changed(base_text: str, head_text: str) -> bool:
    """Compare release-stack configuration between two MkDocs documents."""
    return extract_release_values(parse_mkdocs_extra(base_text)) != extract_release_values(
        parse_mkdocs_extra(head_text)
    )


def expand_mkdocs_macros(content: str, mkdocs_extra: dict[str, str], marker: str) -> str:
    """Replace simple MkDocs macros in a block, rejecting unknown names."""

    def replace(match: re.Match[str]) -> str:
        """Resolve one matched macro to its configured scalar value."""
        name = match.group("name")
        if name not in mkdocs_extra:
            raise DoctestError(f"Unknown mkdocs.yml macro '{{{{ {name} }}}}' in doctest marker '{marker}'.")
        return mkdocs_extra[name]

    return MKDOCS_EXTRA_MACRO_RE.sub(replace, content)


def read_repo_text(path: str, ref: str | None = None, *, allow_missing: bool = False) -> str | None:
    """Read a repository file from the working tree or a Git ref."""
    if ref is None:
        absolute = REPO_ROOT / path
        if not absolute.is_file():
            if allow_missing:
                return None
            raise DoctestError(f"File not found: {path}")
        return absolute.read_text(encoding="utf-8")

    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        if allow_missing:
            return None
        raise DoctestError(f"Cannot read '{path}' at git ref '{ref}'.")
    return result.stdout


def doctest_block_changed(base_text: str | None, head_text: str | None, marker: str, source: str) -> bool:
    """Compare a marked block across refs, rejecting markers missing from both."""
    base_block = extract_doctest_block(base_text, marker, f"{source} at base") if base_text is not None else None
    head_block = extract_doctest_block(head_text, marker, f"{source} at head") if head_text is not None else None
    if base_block is None and head_block is None:
        raise DoctestError(f"Doctest marker '{marker}' does not exist at either ref in {source}.")
    return base_block != head_block


def get_changed_paths(base: str, head: str) -> set[str]:
    """Return changed repository paths and fail if the Git refs cannot be compared."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, head, "--"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise DoctestError(result.stderr.strip() or f"Cannot compare {base} and {head}.")
    return set(result.stdout.splitlines())


def any_doctest_blocks_changed(
    base: str,
    head: str,
    markers: tuple[str, ...],
    content_cache: dict[str, tuple[str | None, str | None]],
) -> bool:
    """Check every marker in a group, sharing cached base/head file contents."""
    blocks_changed = False
    for marker in markers:
        path = DOCTEST_FILE_BY_MARKER[marker]
        if path not in content_cache:
            content_cache[path] = (
                read_repo_text(path, base, allow_missing=True),
                read_repo_text(path, head, allow_missing=True),
            )
        base_text, head_text = content_cache[path]
        if doctest_block_changed(base_text, head_text, marker, path):
            blocks_changed = True
    return blocks_changed


def select_doctests(base: str, head: str) -> dict[str, list[str]]:
    """Select affected devices and installation methods using the fixed change rules."""
    changed_paths = get_changed_paths(base, head)
    content_cache: dict[str, tuple[str | None, str | None]] = {}
    run_a2 = run_310p = QUICKSTART_TEST_SCRIPT in changed_paths
    if any_doctest_blocks_changed(base, head, QUICKSTART_COMMON_MARKERS, content_cache):
        run_a2 = run_310p = True
    if any_doctest_blocks_changed(base, head, QUICKSTART_A2_MARKERS, content_cache):
        run_a2 = True
    if any_doctest_blocks_changed(base, head, QUICKSTART_310P_MARKERS, content_cache):
        run_310p = True

    pip_changed = any_doctest_blocks_changed(base, head, INSTALLATION_PIP_MARKERS, content_cache)
    uv_changed = any_doctest_blocks_changed(base, head, INSTALLATION_UV_MARKERS, content_cache)
    source_changed = any_doctest_blocks_changed(base, head, INSTALLATION_SOURCE_MARKERS, content_cache)
    installation_common_changed = any_doctest_blocks_changed(base, head, INSTALLATION_COMMON_MARKERS, content_cache)
    installation_script_changed = INSTALLATION_TEST_SCRIPT in changed_paths
    run_pip = installation_script_changed or pip_changed
    run_uv = installation_script_changed or uv_changed
    run_source = installation_script_changed or source_changed
    # Common-only changes exercise pip; method-specific changes already cover common setup.
    if installation_common_changed and not (pip_changed or uv_changed or source_changed):
        run_pip = True

    base_text = read_repo_text(MKDOCS_PATH, base)
    head_text = read_repo_text(MKDOCS_PATH, head)
    assert base_text is not None and head_text is not None
    # Shared tooling or release changes add both devices and pip, retaining other selections.
    if release_config_changed(base_text, head_text) or bool(changed_paths & SHARED_DOCTEST_PATHS):
        run_a2 = run_310p = run_pip = True

    quickstart_devices = []
    if run_a2:
        quickstart_devices.append("a2")
    if run_310p:
        quickstart_devices.append("310p")
    installation_methods = []
    if run_source:
        installation_methods.append("source")
    if run_uv:
        installation_methods.append("uv")
    if run_pip:
        installation_methods.append("pip")
    return {"quickstart": quickstart_devices, "installation": installation_methods}


def require_doctest_block(marker: str, ref: str | None = None) -> str:
    """Resolve a registered marker and require its code block to exist."""
    try:
        path = DOCTEST_FILE_BY_MARKER[marker]
    except KeyError as error:
        raise DoctestError(f"Unknown doctest marker '{marker}'.") from error
    text = read_repo_text(path, ref)
    assert text is not None
    content = extract_doctest_block(text, marker, f"{path}{f' at {ref}' if ref else ''}")
    if content is None:
        raise DoctestError(f"Doctest marker '{marker}' was not found in {path}{f' at {ref}' if ref else ''}.")
    return content


def require_mkdocs_extra_value(mkdocs_extra: dict[str, str], key: str) -> str:
    """Require a named scalar value from the parsed MkDocs configuration."""
    try:
        return mkdocs_extra[key]
    except KeyError as error:
        raise DoctestError(f"No simple scalar named '{key}' under mkdocs.yml extra.") from error


def load_mkdocs_extra(ref: str | None = None) -> dict[str, str]:
    """Load scalar MkDocs extra values from the working tree or a Git ref."""
    text = read_repo_text(MKDOCS_PATH, ref)
    assert text is not None
    return parse_mkdocs_extra(text)


def build_quickstart_matrix_entries(
    devices: list[str], vllm_ascend_version: str, image_repository: str
) -> list[dict[str, str]]:
    """Expand each operating system across selected devices and construct vLLM Ascend images."""
    entries = []
    image_repository = image_repository.rstrip("/")
    for os_name in DOCTEST_OSES:
        for device in devices:
            image_tag = vllm_ascend_version
            if device == "310p":
                image_tag += "-310p"
            if os_name == "openeuler":
                image_tag += "-openeuler"
            entries.append(
                {
                    "device": device,
                    "os": os_name,
                    "image": f"{image_repository}:{image_tag}",
                }
            )
    return entries


def build_installation_matrix_entries(
    methods: list[str], cann_version: str, python_version: str, image_repository: str
) -> list[dict[str, str]]:
    """Expand each operating system across selected methods and construct CANN images."""
    entries = []
    image_repository = image_repository.rstrip("/")
    for os_name, os_image_tag in INSTALLATION_OS_IMAGE_TAGS.items():
        for method in methods:
            image_tag = f"{cann_version}-910b-{os_image_tag}-py{python_version}"
            entries.append(
                {
                    "method": method,
                    "os": os_name,
                    "image": f"{image_repository}:{image_tag}",
                }
            )
    return entries


def build_doctest_plan(
    quickstart_devices: list[str],
    installation_methods: list[str],
    quickstart_image_repository: str,
    installation_image_repository: str,
) -> dict[str, object]:
    """Build CI matrices and run flags using the working tree's release versions."""
    mkdocs_extra = load_mkdocs_extra()
    quickstart_entries = build_quickstart_matrix_entries(
        quickstart_devices,
        require_mkdocs_extra_value(mkdocs_extra, "vllm_ascend_version"),
        quickstart_image_repository,
    )
    installation_entries = build_installation_matrix_entries(
        installation_methods,
        require_mkdocs_extra_value(mkdocs_extra, "release_cann_version"),
        require_mkdocs_extra_value(mkdocs_extra, "release_image_python_version"),
        installation_image_repository,
    )
    return {
        "quickstart": {"include": quickstart_entries},
        "installation": {"include": installation_entries},
        "run_quickstart": bool(quickstart_entries),
        "run_installation": bool(installation_entries),
        "skipped": [],
    }


def registry_image_exists(image: str) -> bool:
    """Return whether an SWR image manifest exists, treating only HTTP 404 as missing."""
    repository, tag = image.rsplit(":", 1)
    registry, separator, repository_path = repository.partition("/")
    if not separator:
        raise DoctestError(f"Invalid container image reference: {image}")

    token_query = urlencode(
        {
            "service": "dockyard",
            "scope": f"repository:{repository_path}:pull",
        }
    )
    token_url = f"https://{registry}/swr/auth/v2/registry/auth/?{token_query}"
    with urlopen(token_url, timeout=30) as response:
        token = json.load(response).get("token")
    if not token:
        raise DoctestError(f"SWR did not return an anonymous pull token for {repository}.")

    manifest_url = f"https://{registry}/v2/{quote(repository_path, safe='/')}/manifests/{quote(tag, safe='')}"
    request = Request(
        manifest_url,
        method="HEAD",
        headers={
            "Accept": REGISTRY_MANIFEST_ACCEPT,
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urlopen(request, timeout=30):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise


def source_ref_exists(ref: str) -> bool:
    """Return whether the documented vLLM Ascend clone ref exists as a branch or tag."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            "--tags",
            VLLM_ASCEND_REPOSITORY_URL,
            f"refs/heads/{ref}",
            f"refs/tags/{ref}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        return False
    raise DoctestError(result.stderr.strip() or f"Cannot check vLLM Ascend source ref '{ref}'.")


def filter_missing_images(entries: list[dict[str, str]], name: str, skipped: list[str]) -> list[dict[str, str]]:
    """Remove entries whose image is explicitly absent from the registry."""
    images = list(dict.fromkeys(entry["image"] for entry in entries))
    missing = {image for image in images if not registry_image_exists(image)}
    skipped.extend(f"{name} skipped: image not found: {image}" for image in images if image in missing)
    return [entry for entry in entries if entry["image"] not in missing]


def check_plan_resources(plan: dict[str, object]) -> dict[str, object]:
    """Filter entries whose required image or source ref is not available yet."""
    skipped = plan["skipped"]
    assert isinstance(skipped, list)

    quickstart = plan["quickstart"]
    assert isinstance(quickstart, dict)
    quickstart_entries = quickstart["include"]
    assert isinstance(quickstart_entries, list)
    quickstart["include"] = filter_missing_images(quickstart_entries, "Quick Start", skipped)

    installation = plan["installation"]
    assert isinstance(installation, dict)
    installation_entries = installation["include"]
    assert isinstance(installation_entries, list)
    installation_entries = filter_missing_images(installation_entries, "Installation", skipped)
    source_selected = any(entry["method"] == "source" for entry in installation_entries)
    if source_selected:
        source_ref = require_mkdocs_extra_value(load_mkdocs_extra(), "vllm_ascend_version")
        if not source_ref_exists(source_ref):
            installation_entries = [entry for entry in installation_entries if entry["method"] != "source"]
            skipped.append(f"Installation source skipped: source ref not found: {source_ref}")
    installation["include"] = installation_entries

    plan["run_quickstart"] = bool(quickstart["include"])
    plan["run_installation"] = bool(installation["include"])
    return plan


def parse_args() -> argparse.Namespace:
    """Parse extraction or planning arguments and reject conflicting selection modes."""
    parser = argparse.ArgumentParser(description="Extract marked documentation code blocks and plan doctest CI jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract a marked bash or python doctest block.")
    extract_parser.add_argument("--ref")
    extract_parser.add_argument(
        "--expand-macros", action="store_true", help="Expand {{ name }} values from mkdocs.yml extra before output."
    )
    extract_parser.add_argument("marker")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base")
    plan_parser.add_argument("--head")
    plan_parser.add_argument("--quickstart", choices=("none", "a2", "310p"), default="none")
    plan_parser.add_argument("--installation", choices=("none", "pip", "uv", "source"), default="none")
    plan_parser.add_argument("--quickstart-image-repository", required=True)
    plan_parser.add_argument("--installation-image-repository", required=True)
    plan_parser.add_argument("--check-resources", action="store_true")
    args = parser.parse_args()
    if args.command == "plan" and (args.base is not None or args.head is not None):
        if args.base is None or args.head is None:
            parser.error("--base and --head must be used together")
        if args.quickstart != "none" or args.installation != "none":
            parser.error("manual doctest options cannot be used with --base/--head")
    return args


def main() -> int:
    """Print extracted code or a JSON execution plan without executing doctest commands."""
    args = parse_args()
    if args.command == "extract":
        content = require_doctest_block(args.marker, args.ref)
        if args.expand_macros:
            content = expand_mkdocs_macros(content, load_mkdocs_extra(args.ref), args.marker)
        sys.stdout.write(content)
        return 0
    if args.command == "plan":
        if args.base is not None:
            selection = select_doctests(args.base, args.head)
            plan = build_doctest_plan(
                selection["quickstart"],
                selection["installation"],
                args.quickstart_image_repository,
                args.installation_image_repository,
            )
        else:
            quickstart_devices = [] if args.quickstart == "none" else [args.quickstart]
            installation_methods = [] if args.installation == "none" else [args.installation]
            plan = build_doctest_plan(
                quickstart_devices,
                installation_methods,
                args.quickstart_image_repository,
                args.installation_image_repository,
            )
        if args.check_resources:
            plan = check_plan_resources(plan)
        json.dump(plan, sys.stdout)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DoctestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
