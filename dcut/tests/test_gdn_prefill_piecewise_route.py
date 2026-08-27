# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


DCUT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = DCUT_DIR / "patch_runner.py"
GDN_PATH = DCUT_DIR / "patch_gdn_v023.py"


def test_prefill_does_not_override_outer_runtime_selection() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "_dcut_force_prefill_eager" not in runner
    assert "_FORCE_EAGER_ARG_POSITION" not in runner
    assert "_orig_determine_batch_execution_and_padding" not in runner
    assert "R._determine_batch_execution_and_padding" not in runner


def test_prefill_still_routes_only_gdn_to_native_core() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    gdn = GDN_PATH.read_text(encoding="utf-8")

    assert "_dcut_execute_with_gdn_prefill_route" in runner
    assert 'attr = "_dcut_gdn_scheduler_has_prefill"' in runner
    assert "forward_context._dcut_gdn_native_batch = native_gdn_batch" in runner
    assert '"gdn_native_path": _has_prefill' in runner
    assert '"_dcut_gdn_native_batch"' in gdn
    assert "return bool(native_batch)" in gdn


def test_eager_log_uses_actual_outer_runtime_mode() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "_is_eager = _runtime_mode in" in runner
    assert "_is_eager = _has_prefill or" not in runner
    assert '"runtime_mode": _runtime_mode' in runner
    assert '"is_eager": _is_eager' in runner
