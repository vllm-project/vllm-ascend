import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github" / "workflows" / "scripts" / "collect_merged_new_tests.py"
MODULE_NAME = "collect_merged_new_tests_under_test"


def _load_script():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_result(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_resolve_pr_diff_refs_uses_merge_base_of_pr_base_and_head():
    """Stale main vs PR head must merge-base those two refs, not HEAD vs base."""
    mod = _load_script()
    stale_base = "afdcd487c76c6fc61f9e7dede76d89c9ef78db6e"
    pr_head = "975912a2dc700c45fe01047ad8ab896405f053c4"
    branch_point = "branchpointsha"

    def git_run(args, *, check):
        if args[:2] == ["git", "rev-parse"]:
            return _git_result(args[-1] + "\n")
        if args[:2] == ["git", "merge-base"]:
            assert args[2:] == [stale_base, pr_head]
            return _git_result(f"{branch_point}\n")
        raise AssertionError(args)

    with patch.object(mod, "_git_run", side_effect=git_run):
        old_ref, new_ref = mod.resolve_pr_diff_refs(stale_base, pr_head)

    assert old_ref == branch_point
    assert new_ref == pr_head


def test_git_name_status_uses_three_dot_pr_range():
    mod = _load_script()
    with patch.object(mod, "_git_run", return_value=_git_result("A\ttests/ut/worker/v2/test_eplb_controller.py\n")) as git_run:
        changes = mod._git_name_status("branchpointsha", "prheadsha")

    assert changes == [
        ("A", "tests/ut/worker/v2/test_eplb_controller.py", "tests/ut/worker/v2/test_eplb_controller.py")
    ]
    diff_args = git_run.call_args.args[0]
    assert "branchpointsha...prheadsha" in diff_args
    assert "HEAD" not in diff_args


def test_collect_from_changes_records_only_added_test_file():
    mod = _load_script()
    items = mod.collect_from_changes(
        [("A", "tests/ut/worker/v2/test_eplb_controller.py", "tests/ut/worker/v2/test_eplb_controller.py")],
        read_head=lambda path: "",
        read_old=lambda path: None,
    )
    assert items == [("tests/ut/worker/v2/test_eplb_controller.py", "new test file")]


def test_main_collects_three_dot_pr_diff(tmp_path: Path):
    mod = _load_script()
    output = tmp_path / "merged_new_tests.yaml"
    with (
        patch.object(mod, "resolve_pr_diff_refs", return_value=("branchpointsha", "prheadsha")) as resolve,
        patch.object(
            mod,
            "collect_required_changes",
            return_value=[("tests/ut/worker/v2/test_eplb_controller.py", "new test file")],
        ) as collect,
    ):
        assert (
            mod.main(
                [
                    "--pr-base",
                    "stale-main",
                    "--pr-head",
                    "pr-head",
                    "--output",
                    str(output),
                    "--pr-number",
                    "14929",
                    "--merge-sha",
                    "abc",
                ]
            )
            == 0
        )

    resolve.assert_called_once_with("stale-main", "pr-head")
    collect.assert_called_once_with("branchpointsha", "prheadsha")
    text = output.read_text(encoding="utf-8")
    assert "tests/ut/worker/v2/test_eplb_controller.py" in text
    assert text.count("- path:") == 1
