#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
"""Unit tests for .github/workflows/scripts/codeowner_gate.py."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codeowner_gate as gate  # noqa: E402

SAMPLE = """
# comment, ignored
/vllm_ascend/device @alice @bob
/vllm_ascend/ops/*.py @carol
# TODO(owner): /vllm_ascend/eplb
# /vllm_ascend/eplb @dave
# SKIP(owner): /tests/ut/
/docs @erin
"""


@pytest.fixture
def rules():
    return gate.parse_codeowners(SAMPLE)


class TestParse:
    def test_active_rules(self, rules):
        by_pattern = {r.pattern: r for r in rules}
        assert by_pattern["vllm_ascend/device"].owners == ["@alice", "@bob"]
        assert by_pattern["vllm_ascend/device"].status == "active"

    def test_todo_rule_with_owners(self, rules):
        todo = [r for r in rules if r.status == "todo"]
        assert len(todo) == 1
        assert todo[0].pattern == "vllm_ascend/eplb"
        assert todo[0].owners == ["@dave"]

    def test_skip_rule(self, rules):
        skip = [r for r in rules if r.status == "skip"]
        assert len(skip) == 1
        assert skip[0].pattern == "tests/ut"

    def test_comments_ignored(self, rules):
        assert all("#" not in r.pattern for r in rules)


class TestMatching:
    def test_dir_prefix(self, rules):
        rule = gate.match_rule(rules, "vllm_ascend/device/foo.py")
        assert rule is not None and rule.pattern == "vllm_ascend/device"

    def test_glob(self, rules):
        rule = gate.match_rule(rules, "vllm_ascend/ops/foo.py")
        assert rule is not None and rule.owners == ["@carol"]
        assert gate.match_rule(rules, "vllm_ascend/ops/sub/foo.py") is None

    def test_last_match_wins(self):
        rules = gate.parse_codeowners("/a @x\n/a/b @y\n")
        rule = gate.match_rule(rules, "a/b/c.py")
        assert rule is not None and rule.owners == ["@y"]
        other = gate.match_rule(rules, "a/z.py")
        assert other is not None and other.owners == ["@x"]


class TestApprovals:
    def test_any_owner_approve_passes(self, rules):
        failures = gate.check_approvals(1, "author", rules, ["vllm_ascend/device/a.py"], {"bob": "APPROVED"})
        assert failures == []

    def test_non_owner_approve_does_not_count(self, rules):
        failures = gate.check_approvals(1, "author", rules, ["vllm_ascend/device/a.py"], {"zoe": "APPROVED"})
        assert failures

    def test_author_excluded_even_if_owner(self, rules):
        failures = gate.check_approvals(1, "alice", rules, ["vllm_ascend/device/a.py"], {"alice": "APPROVED"})
        assert failures

    def test_changes_requested_not_counted(self, rules):
        failures = gate.check_approvals(1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "CHANGES_REQUESTED"})
        assert failures

    def test_unowned_file_fails(self, rules):
        failures = gate.check_approvals(1, "author", rules, ["unknown/x.py"], {})
        assert "no CODEOWNERS owner" in failures[0]

    def test_todo_owner_approval_counts(self, rules):
        # dave is a TODO owner; the check accepts their approve (gate is
        # permission-model independent on purpose).
        failures = gate.check_approvals(1, "author", rules, ["vllm_ascend/eplb/x.py"], {"dave": "APPROVED"})
        assert failures == []


class TestCoverage:
    def test_uncovered_file(self, rules):
        failures = gate.check_coverage(rules, ["vllm_ascend/device/a.py", "orphan/x.py"])
        assert any("orphan/x.py" in f for f in failures)
        assert not any("vllm_ascend/device/a.py" in f for f in failures)

    def test_skip_excluded_from_coverage(self):
        rules = gate.parse_codeowners("# SKIP(owner): /tests/ut/\n# /tests/ut/\n")
        assert gate.check_coverage(rules, ["tests/ut/anything.py"]) == []

    def test_stale_active_entry(self):
        rules = gate.parse_codeowners("/deleted_dir @alice\n")
        failures = gate.check_coverage(rules, ["real/file.py"])
        assert any("stale" in f and "deleted_dir" in f for f in failures)

    def test_todo_entry_not_stale_checked(self):
        # TODO entries are inactive; they must not trigger stale errors.
        rules = gate.parse_codeowners("# TODO(owner): /some_dir\n# /some_dir @a\n")
        failures = gate.check_coverage(rules, ["other/file.py"])
        assert failures == ["file has no CODEOWNERS owner: other/file.py"]

    def test_ownerless_todo_does_not_count_as_coverage(self):
        # Strict mode: a file covered only by an ownerless TODO entry is
        # reported as unowned - nobody is accountable for it.
        rules = gate.parse_codeowners("# TODO(owner): /some_dir\n# /some_dir\n")
        failures = gate.check_coverage(rules, ["some_dir/file.py"])
        assert failures == ["file has no CODEOWNERS owner: some_dir/file.py"]

    def test_todo_with_owner_counts_as_owned(self):
        # A TODO entry that already names owners counts as owned (it becomes
        # real the moment it is uncommented).
        rules = gate.parse_codeowners("# TODO(owner): /some_dir @pending\n# /some_dir @pending\n")
        assert gate.check_coverage(rules, ["some_dir/file.py"]) == []

    def test_coverage_passes(self, rules):
        tracked = ["vllm_ascend/device/a.py", "vllm_ascend/ops/b.py", "docs/x.md", "tests/ut/t.py"]
        assert gate.check_coverage(rules, tracked) == []


class TestAuthorOnlyOwnership:
    def test_author_is_sole_owner(self, rules):
        rules = gate.parse_codeowners("/mymod @solo\n")
        failures = gate.check_approvals(1, "solo", rules, ["mymod/x.py"], {"solo": "APPROVED"})
        assert "only owner is the PR author" in failures[0]

    def test_waiting_message_lists_eligible_owners(self, rules):
        failures = gate.check_approvals(1, "zoe", rules, ["vllm_ascend/device/a.py"], {})
        assert "waiting for approval from one of: @alice, @bob" in failures[0]


class TestOwnerlessActiveRule:
    def test_flagged_in_coverage(self):
        rules = gate.parse_codeowners("/orphan_dir\n/real @a\n")
        failures = gate.check_coverage(rules, ["real/f.py", "orphan_dir/f.py"])
        assert any("no owner and is ignored" in f for f in failures)

    def test_no_false_coverage_from_ownerless_rule(self):
        # Files only matched by an ownerless rule must be reported as
        # unowned, because GitHub ignores ownerless entries.
        rules = gate.parse_codeowners("/orphan_dir\n/real @a\n")
        failures = gate.check_coverage(rules, ["orphan_dir/f.py"])
        assert any("orphan_dir/f.py" in f and "no CODEOWNERS owner" in f for f in failures)


class TestNestedOwners:
    """Union semantics: parent owner covers submodules; a more specific
    rule adds owners."""

    RULES_TEXT = "/model_loader @yilunh998\n/model_loader/rfork/ @yangsonglin13\n"

    def rules(self):
        return gate.parse_codeowners(self.RULES_TEXT)

    def test_owner_pool_unions_parent_and_child(self):
        owners, skipped = gate.resolve_owners(self.rules(), "model_loader/rfork/x.py")
        assert owners == ["@yilunh998", "@yangsonglin13"] and not skipped

    def test_either_owner_approve_passes(self):
        for reviewer in ("yilunh998", "yangsonglin13"):
            failures = gate.check_approvals(
                1, "author", self.rules(), ["model_loader/rfork/x.py"], {reviewer: "APPROVED"}
            )
            assert failures == [], (reviewer, failures)

    def test_parent_veto_blocks_submodule(self):
        failures = gate.check_approvals(
            1,
            "author",
            self.rules(),
            ["model_loader/rfork/x.py"],
            {"yangsonglin13": "APPROVED", "yilunh998": "CHANGES_REQUESTED"},
        )
        assert len(failures) == 1 and "yilunh998" in failures[0]

    def test_child_veto_blocks_even_if_parent_approves(self):
        failures = gate.check_approvals(
            1,
            "author",
            self.rules(),
            ["model_loader/rfork/x.py"],
            {"yilunh998": "APPROVED", "yangsonglin13": "CHANGES_REQUESTED"},
        )
        assert len(failures) == 1 and "yangsonglin13" in failures[0]

    def test_parent_owner_still_owns_uncovered_files(self):
        failures = gate.check_approvals(
            1, "author", self.rules(), ["model_loader/netloader/x.py"], {"yilunh998": "APPROVED"}
        )
        assert failures == []

    def test_changes_requested_blocks_despite_approval(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "CHANGES_REQUESTED", "bob": "APPROVED"}
        )
        assert any("changes requested by @alice" in f for f in failures)

    def test_blocking_owner_re_review_clears_gate(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "APPROVED", "bob": "APPROVED"}
        )
        assert failures == []

    def test_dismissed_review_does_not_block(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "DISMISSED", "bob": "APPROVED"}
        )
        assert failures == []

    def test_author_requested_changes_not_counted(self):
        rules = gate.parse_codeowners("/a @solo\n")
        failures = gate.check_approvals(1, "solo", rules, ["a/x.py"], {"solo": "CHANGES_REQUESTED"})
        assert any("only owner is the PR author" in f for f in failures)


class TestGhApi:
    """gh_api uses urllib with GH_TOKEN; no gh CLI dependency."""

    def _mock_urlopen(self, monkeypatch, pages):
        """pages: list of (body, link_header) served in order."""
        import io

        responses = iter(pages)

        class FakeResponse:
            def __init__(self, body, link):
                self._io = io.BytesIO(json.dumps(body).encode())
                self.headers = {"Link": link}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return self._io.read()

        def fake_urlopen(request):
            body, link = next(responses)
            return FakeResponse(body, link)

        monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)

    def test_flattens_paginated_arrays(self, monkeypatch):
        self._mock_urlopen(
            monkeypatch,
            [(["r1", "r2"], '<https://api.github.com/x?page=2>; rel="next"'), (["r3"], "")],
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setenv("GH_TOKEN", "t")
        assert gate.gh_api("repos/{owner}/{repo}/x") == ["r1", "r2", "r3"]

    def test_object_endpoint_returns_last_page(self, monkeypatch):
        self._mock_urlopen(
            monkeypatch,
            [({"a": 1}, '<https://api.github.com/y?page=2>; rel="next"'), ({"a": 2}, "")],
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        assert gate.gh_api("repos/{owner}/{repo}/y") == {"a": 2}

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            gate.gh_api("repos/{owner}/{repo}/z")


class TestSkipInApprovals:
    def test_skip_overrides_active_rule(self):
        # /tests is actively owned, but the later SKIP(owner) /tests/ut/
        # carves out a deliberately-unowned subtree.
        rules = gate.parse_codeowners("/tests @wenjun91\n# SKIP(owner): /tests/ut/\n# SKIP(owner): /tests/ut/\n")
        failures = gate.check_approvals(1, "author", rules, ["tests/ut/foo.py"], {})
        assert failures == []

    def test_active_outside_skip_still_gated(self):
        rules = gate.parse_codeowners("/tests @wenjun91\n# SKIP(owner): /tests/ut/\n")
        failures = gate.check_approvals(1, "author", rules, ["tests/e2e/foo.py"], {})
        assert any("wenjun91" in f for f in failures)

    def test_skip_before_active_does_not_win(self):
        # An active rule after the SKIP re-owns the subtree.
        rules = gate.parse_codeowners("# SKIP(owner): /tests/ut/\n# /tests/ut/\n/tests @wenjun91\n")
        owners, skipped = gate.resolve_owners(rules, "tests/ut/foo.py")
        assert not skipped and owners == ["@wenjun91"]

    def test_todo_owner_approval_counts(self, rules):
        failures = gate.check_approvals(1, "author", rules, ["vllm_ascend/eplb/x.py"], {"dave": "APPROVED"})
        assert failures == []


class TestChangesRequested:
    def test_changes_requested_blocks_despite_approval(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "CHANGES_REQUESTED", "bob": "APPROVED"}
        )
        assert any("changes requested by @alice" in f for f in failures)

    def test_blocking_owner_re_review_clears_gate(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "APPROVED", "bob": "APPROVED"}
        )
        assert failures == []

    def test_dismissed_review_does_not_block(self, rules):
        failures = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"alice": "DISMISSED", "bob": "APPROVED"}
        )
        assert failures == []

    def test_author_requested_changes_not_counted(self):
        rules = gate.parse_codeowners("/a @solo\n")
        failures = gate.check_approvals(1, "solo", rules, ["a/x.py"], {"solo": "CHANGES_REQUESTED"})
        assert any("only owner is the PR author" in f for f in failures)
