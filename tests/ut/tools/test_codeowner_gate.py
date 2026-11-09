#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
"""Unit tests for .github/workflows/scripts/codeowner_gate.py."""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".github", "workflows", "scripts"),
)

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
        assert rule.pattern == "vllm_ascend/device"

    def test_glob(self, rules):
        assert gate.match_rule(rules, "vllm_ascend/ops/foo.py").owners == ["@carol"]
        assert gate.match_rule(rules, "vllm_ascend/ops/sub/foo.py") is None

    def test_last_match_wins(self):
        rules = gate.parse_codeowners("/a @x\n/a/b @y\n")
        assert gate.match_rule(rules, "a/b/c.py").owners == ["@y"]
        assert gate.match_rule(rules, "a/z.py").owners == ["@x"]


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
        assert failures == ["file has no CODEOWNERS entry: other/file.py"]

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
        # Files only matched by an ownerless rule must still be reported as
        # uncovered, because GitHub ignores ownerless entries.
        rules = gate.parse_codeowners("/orphan_dir\n/real @a\n")
        failures = gate.check_coverage(rules, ["orphan_dir/f.py"])
        assert any("orphan_dir/f.py" in f and "no CODEOWNERS entry" in f for f in failures)


class TestActiveOwnersFor:
    def test_active_overrides_todo(self):
        rules = gate.parse_codeowners("# TODO(owner): /a\n# /a @pending\n/a @real\n")
        assert gate.active_owners_for(rules, "a/x.py") == ["@real"]

    def test_todo_fills_when_no_active(self):
        rules = gate.parse_codeowners("# TODO(owner): /a\n# /a @pending\n")
        assert gate.active_owners_for(rules, "a/x.py") == ["@pending"]


class TestGhApiSlurp:
    def test_flattens_paginated_arrays(self, monkeypatch):
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            out = '[["r1", "r2"], ["r3"]]'
            import types

            return types.SimpleNamespace(stdout=out, returncode=0)

        monkeypatch.setattr(gate.subprocess, "run", fake_run)
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        result = gate.gh_api("repos/{owner}/{repo}/x")
        assert result == ["r1", "r2", "r3"]

    def test_object_endpoint_returns_last_page(self, monkeypatch):
        import types

        def fake_run(cmd, capture_output, text, check):
            return types.SimpleNamespace(stdout='[{"a":1}, {"a":2}]', returncode=0)

        monkeypatch.setattr(gate.subprocess, "run", fake_run)
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        assert gate.gh_api("repos/{owner}/{repo}/y") == {"a": 2}
