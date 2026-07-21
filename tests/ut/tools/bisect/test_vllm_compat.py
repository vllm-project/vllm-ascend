from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.vllm_compat import _compare, expected_vllm_tag, expected_vllm_tag_at


@pytest.mark.parametrize(
    ("expected", "installed", "compatible", "message"),
    [
        ("v0.9.1", "0.9.1+local", True, "vllm matches"),
        ("0.9.2", "0.9.1", False, "vllm version mismatch"),
        (None, "0.9.1", True, "skipping vllm compat check"),
        ("not-a-version", "0.9.1", True, "cannot parse vllm versions"),
    ],
)
def test_compare_vllm_versions(expected: str | None, installed: str | None, compatible: bool, message: str):
    actual_compatible, actual_message = _compare(expected, installed)

    assert actual_compatible is compatible
    assert message in actual_message


def test_expected_vllm_tag_reads_current_checkout_file(tmp_path: Path):
    tag_file = tmp_path / ".github" / "vllm-release-tag.commit"
    tag_file.parent.mkdir()
    tag_file.write_text("v0.9.1\n", encoding="utf-8")

    assert expected_vllm_tag(tmp_path) == "v0.9.1"


def test_expected_vllm_tag_at_reads_file_without_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(git_ops, "file_at_commit", lambda repo, commit, rel_path: "v0.9.2\n")

    assert expected_vllm_tag_at(tmp_path, "a" * 40) == "v0.9.2"
