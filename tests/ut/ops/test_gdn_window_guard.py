# SPDX-License-Identifier: Apache-2.0
"""Keep the recurrent GDN dispatch in step with the installed fla_npu OPP.

``vllm_ascend/ops/gdn.py`` carries a local per-sequence cap for the pre-V2
AscendC recurrent kernel: that kernel consumed a single eight-token window and
returned without writing output above it. ``fla_npu`` #1070 replaced it with an
architecture-split, window-loop implementation that has no per-sequence cap.

Nothing keeps the Python constant and the installed kernel in step: a fla_npu
version bump changes the kernel silently, and the framework would keep routing
wide verify windows to Triton. This test is that reminder. It reads the
operator that is actually installed (or, when fla_npu is not installed, the
version this repository pins) and fails when the two disagree, with the exact
edit required. It checks the width comparison as well as the constant, since
either half can reintroduce the cap on its own.

The test runs no device code, so it is valid in the CPU unit-test job.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[3]
GDN_SOURCE = REPO_ROOT / "vllm_ascend" / "ops" / "gdn.py"
PROFILE_SOURCE = REPO_ROOT / "vllm_ascend" / "device" / "hardware_profile.py"
OP_NAME = "recurrent_gated_delta_rule"

# fla_npu version -> largest per-sequence verify window its OPP accepts.
# ``None`` means the OPP is the window-loop kernel and imposes no cap.
# Add an entry whenever the pins below move; ``test_declared_pins_agree``
# fails until you do, which is the point of the guard.
KNOWN_OPP_WINDOWS: dict[str, int | None] = {
    "26.9.1": 8,
}

UNKNOWN = object()

_MAX_MTP = re.compile(r"\bMAX_MTP\s*=\s*(\d+)\s*;")
_CODE_CAP = re.compile(r"^_MAX_RECURRENT_GDN_TOKENS\s*=\s*(\d+)\s*$", re.MULTILINE)
_CAP_NAME = "_MAX_RECURRENT_GDN_TOKENS"
# The width gate #16292 added: ``spec_state_indices_tensor.shape[1] > 8``.
# Match the use site as well as the definition. Deleting the constant but
# inlining the number still caps the dispatch; deleting the gate but keeping
# the constant sends wide windows to the capped kernel. Both are this bug.
_WIDTH_GATE = re.compile(r"shape\[1\]\s*>\s*([A-Za-z_]\w*|\d+)")
_PROFILE_OVERRIDE = re.compile(r"gdn_fixed_spec_recurrent_fallback_min_query_len\s*=\s*(\d+)")
_DOCKER_PIN = re.compile(r'ARG\s+FLA_VERSION="([^"]+)"')
_WORKFLOW_PIN = re.compile(r"flash-linear-attention-npu(?:-[^=\s\"]*)?==+\s*([0-9][\w.\-+]*)")

REMOVE_THE_CAP = """\
The installed fla_npu OPP is the window-loop kernel: {source} ships one
directory per architecture and no fixed window, so it accepts any per-sequence
verify length. vllm_ascend still caps the dispatch {limit}, which keeps
k >= {threshold} on the Triton path forever.

Delete both:
  - vllm_ascend/ops/gdn.py: _MAX_RECURRENT_GDN_TOKENS and the whole
    `spec_state_indices_tensor.shape[1] > ...` clause in
    AscendGatedDeltaNetAttention._forward_core (do not inline the number)
  - vllm_ascend/device/hardware_profile.py: the A2
    gdn_fixed_spec_recurrent_fallback_min_query_len=9 workaround, which was
    observed on the single-window kernel that this OPP replaced
then update the guard branch in this test.
"""

RESTORE_THE_CAP = """\
The installed fla_npu OPP still caps a sequence at {window} tokens ({source}),
but vllm_ascend no longer caps the dispatch. On this pairing a verify window of
{window_plus} tokens reaches the kernel, which returns without writing output:
the request produces garbage instead of an error.

Either install the fla_npu release that ships the window-loop kernel, or
restore both _MAX_RECURRENT_GDN_TOKENS = {window} and its
`spec_state_indices_tensor.shape[1] > _MAX_RECURRENT_GDN_TOKENS` clause in
vllm_ascend/ops/gdn.py, plus the A2
gdn_fixed_spec_recurrent_fallback_min_query_len workaround.
"""

UNKNOWN_LAYOUT = """\
Cannot tell which recurrent GDN kernel is installed.

The operator directory {where} has neither an architecture directory (V2
window-loop kernel) nor a header carrying MAX_MTP (legacy fixed window). A
fla_npu layout change has silently disabled this guard: inspect the installed
tree and update the detector rather than skipping the test.
"""


def _ordered_vendors(vendors_root: Path) -> list[Path]:
    """Vendor trees in the order the OPP loader visits them.

    ``vendors/config.ini`` carries ``load_priority``; a wheel ships exactly one
    vendor, so this only matters when a second tree has been overlaid.
    """
    priority: list[str] = []
    config = vendors_root / "config.ini"
    if config.is_file():
        for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "load_priority":
                priority = [name.strip() for name in value.split(",") if name.strip()]
    rank = {name: index for index, name in enumerate(priority)}
    vendors = [path for path in vendors_root.glob("*") if path.is_dir()]
    return sorted(vendors, key=lambda path: (rank.get(path.name, len(rank)), path.name))


def _installed_vendor_dirs() -> list[Path]:
    """Vendor trees fla_npu installs, without importing the package.

    ``tests/ut/conftest.py`` replaces ``fla_npu`` with a stub, so import it is
    not an option here; the distribution metadata names the real tree.
    """
    roots: list[Path] = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name not in {"fla-npu", "fla_npu"} and not name.startswith("flash-linear-attention-npu"):
            continue
        package = Path(str(dist.locate_file("fla_npu")))
        roots.extend(_ordered_vendors(package / "opp" / "vendors"))
    return roots


def _candidate_vendors() -> list[Path]:
    """The installed vendor trees, in loader order.

    Overlay trees are deliberately out of scope: the supported deployment is a
    matching fla_npu wheel, and the guard must describe the kernel that wheel
    installs. An unrecognised base tree fails instead of falling back.
    """
    return _installed_vendor_dirs()


def _op_dirs(vendor: Path) -> list[Path]:
    root = vendor / "op_impl" / "ai_core" / "tbe"
    return sorted(path for path in root.glob(f"*/ascendc/{OP_NAME}") if path.is_dir())


def _window_of(op_dir: Path) -> int | None | object:
    # Check the architecture directories first: a V2 tree that still shipped a
    # root header would otherwise be read as the legacy fixed window.
    if (op_dir / "arch22").is_dir() or (op_dir / "arch35").is_dir():
        return None
    header = op_dir / f"{OP_NAME}.h"
    if header.is_file():
        match = _MAX_MTP.search(header.read_text(encoding="utf-8", errors="replace"))
        if match is not None:
            return int(match.group(1))
    return UNKNOWN


def _installed_window() -> tuple[int | None | object, str]:
    for vendor in _candidate_vendors():
        op_dirs = _op_dirs(vendor)
        if not op_dirs:
            continue
        # The vendor the loader visits first is the one whose kernel runs, so a
        # lower-priority vendor cannot describe it.
        return _window_of(op_dirs[0]), str(op_dirs[0])
    return UNKNOWN, "no installed fla_npu OPP was found"


def _declared_pins() -> dict[Path, str]:
    pins: dict[Path, str] = {}
    for path in sorted(REPO_ROOT.glob("Dockerfile*")):
        if not path.is_file():
            continue
        for match in _DOCKER_PIN.finditer(path.read_text(encoding="utf-8")):
            pins[path] = match.group(1)
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.y*ml")):
        if not path.is_file():
            continue
        for match in _WORKFLOW_PIN.finditer(path.read_text(encoding="utf-8")):
            pins[path] = match.group(1)
    return pins


def _python_cap(source: Path = GDN_SOURCE) -> int | None:
    match = _CODE_CAP.search(source.read_text(encoding="utf-8"))
    return None if match is None else int(match.group(1))


def _width_gate(source: Path = GDN_SOURCE) -> str | None:
    """What ``_forward_core`` compares the verify width against, if anything."""
    match = _WIDTH_GATE.search(source.read_text(encoding="utf-8"))
    return None if match is None else match.group(1)


def _profile_override() -> int | None:
    match = _PROFILE_OVERRIDE.search(PROFILE_SOURCE.read_text(encoding="utf-8"))
    return None if match is None else int(match.group(1))


def _declared_window() -> tuple[int | None | object, str]:
    pins = _declared_pins()
    if not pins:
        return UNKNOWN, "no fla_npu pin was found in this repository"
    versions = sorted(set(pins.values()))
    if len(versions) > 1:
        where = ", ".join(f"{path.relative_to(REPO_ROOT)}={version}" for path, version in pins.items())
        return UNKNOWN, f"conflicting fla_npu pins: {where}"
    version = versions[0]
    if version not in KNOWN_OPP_WINDOWS:
        return UNKNOWN, (
            f"fla_npu {version} is not in KNOWN_OPP_WINDOWS; confirm whether that "
            "release contains flash-linear-attention-npu #1070 and add the entry"
        )
    return KNOWN_OPP_WINDOWS[version], f"fla_npu {version}"


def _observed_window() -> tuple[int | None | object, str]:
    window, source = _installed_window()
    if window is not UNKNOWN:
        return window, source
    # An installed but unrecognisable tree is a broken guard, not a reason to
    # fall back to the pin: fail instead of pretending the pin describes it.
    if _candidate_vendors():
        return UNKNOWN, source
    return _declared_window()


def test_declared_pins_agree() -> None:
    """Every place that pins fla_npu must name the same release."""
    pins = _declared_pins()
    # A scanner that silently stops matching one kind of file would let a
    # disagreeing pin hide, so require both targets to be seen.
    assert any(path.name.startswith("Dockerfile") for path in pins), "no Dockerfile pin was found"
    assert any(path.suffix in {".yml", ".yaml"} for path in pins), "no workflow pin was found"
    versions = sorted(set(pins.values()))
    assert len(versions) <= 1, f"fla_npu pins disagree: {versions}"


def test_workflow_pin_accepts_both_specifiers() -> None:
    """``==`` and ``===`` both pin; missing either would hide a workflow."""
    cases = {
        "uv pip install --no-deps flash-linear-attention-npu-a3===26.9.1": "26.9.1",
        "uv pip install --no-deps flash-linear-attention-npu-a3==26.9.1": "26.9.1",
        "flash-linear-attention-npu==26.10.0": "26.10.0",
    }
    for line, expected in cases.items():
        match = _WORKFLOW_PIN.search(line)
        assert match is not None, line
        assert match.group(1) == expected, line


def test_declared_window_is_known() -> None:
    """A pin bump must classify the release before anything else can run."""
    window, source = _declared_window()
    assert window is not UNKNOWN, source


def test_dispatch_cap_matches_installed_kernel() -> None:
    """The Python cap must describe the kernel that is actually installed."""
    window, source = _observed_window()
    assert window is not UNKNOWN, (
        source if "fla_npu" in source and "not in KNOWN" in source else UNKNOWN_LAYOUT.format(where=source)
    )

    cap = _python_cap()
    gate = _width_gate()
    if window is None:
        limit = f"at {cap} tokens" if cap is not None else f"with `shape[1] > {gate}`"
        message = REMOVE_THE_CAP.format(source=source, limit=limit, threshold=cap if cap is not None else gate)
        assert cap is None, message
        assert gate is None, message
        assert _profile_override() is None, message
        return

    if not isinstance(window, int):
        raise AssertionError(UNKNOWN_LAYOUT.format(where=source))

    assert cap == window, RESTORE_THE_CAP.format(window=window, source=source, window_plus=window + 1)
    # Deleting the gate while keeping the constant would send wide windows to
    # the capped kernel with nothing left to stop them.
    assert gate == _CAP_NAME, RESTORE_THE_CAP.format(window=window, source=source, window_plus=window + 1)
    # The numerical workaround belongs to the single-window kernel; it must
    # stay while that kernel is the one installed.
    assert _profile_override() is not None, RESTORE_THE_CAP.format(window=window, source=source, window_plus=window + 1)


class TestDetector:
    """The detector itself, so a layout change fails here first."""

    def test_legacy_window_is_read_from_the_kernel_header(self, tmp_path: Path) -> None:
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        (op_dir / f"{OP_NAME}.h").write_text("constexpr uint64_t MAX_MTP = 8;\n", encoding="utf-8")
        assert _window_of(op_dir) == 8

    def test_a_wider_legacy_window_is_not_hard_coded(self, tmp_path: Path) -> None:
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        (op_dir / f"{OP_NAME}.h").write_text("constexpr uint64_t MAX_MTP = 16;\n", encoding="utf-8")
        assert _window_of(op_dir) == 16

    def test_architecture_directory_means_no_cap(self, tmp_path: Path) -> None:
        op_dir = tmp_path / "op"
        (op_dir / "arch22").mkdir(parents=True)
        # A V2 tree that still shipped the dead header must still read as V2.
        (op_dir / f"{OP_NAME}.h").write_text("constexpr uint64_t MAX_MTP = 8;\n", encoding="utf-8")
        assert _window_of(op_dir) is None

    def test_arch35_alone_is_the_same_v2_kernel(self, tmp_path: Path) -> None:
        op_dir = tmp_path / "op"
        (op_dir / "arch35").mkdir(parents=True)
        assert _window_of(op_dir) is None

    def test_unrecognised_layout_is_unknown(self, tmp_path: Path) -> None:
        op_dir = tmp_path / "op"
        op_dir.mkdir()
        (op_dir / "something_else.h").write_text("// nothing\n", encoding="utf-8")
        assert _window_of(op_dir) is UNKNOWN

    def test_first_vendor_that_ships_the_operator_decides(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy" / "op_impl" / "ai_core" / "tbe" / "impl" / "ascendc" / OP_NAME
        legacy.mkdir(parents=True)
        (legacy / f"{OP_NAME}.h").write_text("constexpr uint64_t MAX_MTP = 8;\n", encoding="utf-8")
        v2 = tmp_path / "v2" / "op_impl" / "ai_core" / "tbe" / "impl" / "ascendc" / OP_NAME
        (v2 / "arch22").mkdir(parents=True)

        assert _op_dirs(tmp_path / "legacy") == [legacy]
        assert _window_of(_op_dirs(tmp_path / "legacy")[0]) == 8
        assert _window_of(_op_dirs(tmp_path / "v2")[0]) is None

    def test_vendor_order_follows_the_packaged_config(self, tmp_path: Path) -> None:
        vendors = tmp_path / "vendors"
        for name in ("aaa", "mmm", "zzz"):
            (vendors / name).mkdir(parents=True)
        (vendors / "config.ini").write_text("load_priority=zzz,aaa\n", encoding="utf-8")
        assert [path.name for path in _ordered_vendors(vendors)] == ["zzz", "aaa", "mmm"]

        (vendors / "config.ini").unlink()
        assert [path.name for path in _ordered_vendors(vendors)] == ["aaa", "mmm", "zzz"]

    def test_width_gate_reports_the_comparison_operand(self, tmp_path: Path) -> None:
        source = tmp_path / "gdn.py"
        source.write_text(
            "use_upstream_recurrent = flag or (\n"
            "    indices.ndim == 2 and indices.shape[1] > _MAX_RECURRENT_GDN_TOKENS\n"
            ")\n",
            encoding="utf-8",
        )
        assert _width_gate(source) == _CAP_NAME

        source.write_text("use_upstream_recurrent = flag or (indices.shape[1] > 8)\n", encoding="utf-8")
        assert _width_gate(source) == "8"

        source.write_text("use_upstream_recurrent = flag\n", encoding="utf-8")
        assert _width_gate(source) is None


def test_repository_state_is_classified() -> None:
    """This repository's own sources must match one of the known states."""
    cap = _python_cap()
    assert cap in (None, 8), (
        "The dispatch cap was changed without updating this guard; record the new "
        "state in KNOWN_OPP_WINDOWS and the assertions above."
    )
