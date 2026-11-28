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
# TODO(owner): /vllm_ascend/example_feature
# /vllm_ascend/example_feature @dave
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
        assert todo[0].pattern == "vllm_ascend/example_feature"
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
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"bob": "APPROVED"}
        )
        assert failures == []

    def test_non_owner_approve_does_not_count(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {"zoe": "APPROVED"}
        )
        assert failures

    def test_author_excluded_even_if_owner(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "alice", rules, ["vllm_ascend/device/a.py"], {"alice": "APPROVED"}
        )
        assert failures

    def test_changes_requested_not_counted(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "CHANGES_REQUESTED"},
        )
        assert failures

    def test_unowned_file_fails(self, rules):
        # Policy: unclaimed files are not gate failures; they are surfaced
        # as a notice (PR comment / warnings) instead.
        failures, unclaimed, _self, _ignored = gate.check_approvals(1, "author", rules, ["unknown/x.py"], {})
        assert failures == []
        assert unclaimed == ["unknown/x.py"]

    def test_todo_owner_approval_counts(self, rules):
        # dave is a TODO owner; the check accepts their approve (gate is
        # permission-model independent on purpose).
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/example_feature/x.py"], {"dave": "APPROVED"}
        )
        assert failures == []


class TestCoverage:
    def test_uncovered_file(self, rules):
        errors, warnings = gate.check_coverage(rules, ["vllm_ascend/device/a.py", "orphan/x.py"])
        assert any("orphan/x.py" in w for w in warnings)
        assert not any("vllm_ascend/device/a.py" in f for f in errors + warnings)

    def test_skip_excluded_from_coverage(self):
        rules = gate.parse_codeowners("# SKIP(owner): /tests/ut/\n# /tests/ut/\n")
        assert gate.check_coverage(rules, ["tests/ut/anything.py"]) == ([], [])

    def test_stale_active_entry(self):
        rules = gate.parse_codeowners("/deleted_dir @alice\n")
        errors, warnings = gate.check_coverage(rules, ["real/file.py"])
        assert any("stale" in f and "deleted_dir" in f for f in errors)

    def test_todo_entry_not_stale_checked(self):
        # TODO entries are inactive; they must not trigger stale errors.
        rules = gate.parse_codeowners("# TODO(owner): /some_dir\n# /some_dir @a\n")
        errors, warnings = gate.check_coverage(rules, ["other/file.py"])
        assert errors == [] and warnings == ["file has no CODEOWNERS owner: other/file.py"]

    def test_ownerless_todo_does_not_count_as_coverage(self):
        # Strict mode: a file covered only by an ownerless TODO entry is
        # reported as unowned - nobody is accountable for it.
        rules = gate.parse_codeowners("# TODO(owner): /some_dir\n# /some_dir\n")
        errors, warnings = gate.check_coverage(rules, ["some_dir/file.py"])
        assert errors == [] and warnings == ["file has no CODEOWNERS owner: some_dir/file.py"]

    def test_todo_with_owner_counts_as_owned(self):
        # A TODO entry that already names owners counts as owned (it becomes
        # real the moment it is uncommented).
        rules = gate.parse_codeowners("# TODO(owner): /some_dir @pending\n# /some_dir @pending\n")
        assert gate.check_coverage(rules, ["some_dir/file.py"]) == ([], [])

    def test_coverage_passes(self, rules):
        tracked = [
            "vllm_ascend/device/a.py",
            "vllm_ascend/ops/b.py",
            "docs/x.md",
            "tests/ut/t.py",
        ]
        assert gate.check_coverage(rules, tracked) == ([], [])


class TestAuthorOnlyOwnership:
    def test_author_is_sole_owner(self, rules):
        rules = gate.parse_codeowners("/mymod @solo\n")
        failures, _unclaimed, self_owned, _ignored = gate.check_approvals(
            1, "solo", rules, ["mymod/x.py"], {"solo": "APPROVED"}
        )
        assert failures == [] and self_owned == ["mymod/x.py"]

    def test_waiting_message_lists_eligible_owners(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "zoe", rules, ["vllm_ascend/device/a.py"], {}
        )
        assert "waiting for approval from one of: @alice, @bob" in failures[0]


class TestOwnerlessActiveRule:
    def test_flagged_in_coverage(self):
        rules = gate.parse_codeowners("/orphan_dir\n/real @a\n")
        errors, warnings = gate.check_coverage(rules, ["real/f.py", "orphan_dir/f.py"])
        assert any("no owner and is ignored" in f for f in errors)

    def test_no_false_coverage_from_ownerless_rule(self):
        # Files only matched by an ownerless rule must be reported as
        # unowned, because GitHub ignores ownerless entries.
        rules = gate.parse_codeowners("/orphan_dir\n/real @a\n")
        errors, warnings = gate.check_coverage(rules, ["orphan_dir/f.py"])
        assert any("orphan_dir/f.py" in w and "no CODEOWNERS owner" in w for w in warnings)


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
            failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
                1,
                "author",
                self.rules(),
                ["model_loader/rfork/x.py"],
                {reviewer: "APPROVED"},
            )
            assert failures == [], (reviewer, failures)

    def test_parent_veto_blocks_submodule(self):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            self.rules(),
            ["model_loader/rfork/x.py"],
            {"yangsonglin13": "APPROVED", "yilunh998": "CHANGES_REQUESTED"},
        )
        assert len(failures) == 1 and "yilunh998" in failures[0]

    def test_child_veto_blocks_even_if_parent_approves(self):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            self.rules(),
            ["model_loader/rfork/x.py"],
            {"yilunh998": "APPROVED", "yangsonglin13": "CHANGES_REQUESTED"},
        )
        assert len(failures) == 1 and "yangsonglin13" in failures[0]

    def test_parent_owner_still_owns_uncovered_files(self):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            self.rules(),
            ["model_loader/netloader/x.py"],
            {"yilunh998": "APPROVED"},
        )
        assert failures == []

    def test_changes_requested_blocks_despite_approval(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "CHANGES_REQUESTED", "bob": "APPROVED"},
        )
        assert any("changes requested by @alice" in f for f in failures)

    def test_blocking_owner_re_review_clears_gate(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "APPROVED", "bob": "APPROVED"},
        )
        assert failures == []

    def test_dismissed_review_does_not_block(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "DISMISSED", "bob": "APPROVED"},
        )
        assert failures == []

    def test_author_requested_changes_not_counted(self):
        rules = gate.parse_codeowners("/a @solo\n")
        failures, _unclaimed, self_owned, _ignored = gate.check_approvals(
            1, "solo", rules, ["a/x.py"], {"solo": "CHANGES_REQUESTED"}
        )
        assert failures == [] and self_owned == ["a/x.py"]


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
            [
                (["r1", "r2"], '<https://api.github.com/x?page=2>; rel="next"'),
                (["r3"], ""),
            ],
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setenv("GH_TOKEN", "t")
        assert gate.gh_api("repos/{owner}/{repo}/x") == ["r1", "r2", "r3"]

    def test_object_endpoint_returns_last_page(self, monkeypatch):
        self._mock_urlopen(
            monkeypatch,
            [
                ({"a": 1}, '<https://api.github.com/y?page=2>; rel="next"'),
                ({"a": 2}, ""),
            ],
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
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(1, "author", rules, ["tests/ut/foo.py"], {})
        assert failures == []

    def test_active_outside_skip_still_gated(self):
        rules = gate.parse_codeowners("/tests @wenjun91\n# SKIP(owner): /tests/ut/\n")
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(1, "author", rules, ["tests/e2e/foo.py"], {})
        assert any("wenjun91" in f for f in failures)

    def test_skip_before_active_does_not_win(self):
        # An active rule after the SKIP re-owns the subtree.
        rules = gate.parse_codeowners("# SKIP(owner): /tests/ut/\n# /tests/ut/\n/tests @wenjun91\n")
        owners, skipped = gate.resolve_owners(rules, "tests/ut/foo.py")
        assert not skipped and owners == ["@wenjun91"]

    def test_todo_owner_approval_counts(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/example_feature/x.py"], {"dave": "APPROVED"}
        )
        assert failures == []


class TestChangesRequested:
    def test_changes_requested_blocks_despite_approval(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "CHANGES_REQUESTED", "bob": "APPROVED"},
        )
        assert any("changes requested by @alice" in f for f in failures)

    def test_blocking_owner_re_review_clears_gate(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "APPROVED", "bob": "APPROVED"},
        )
        assert failures == []

    def test_dismissed_review_does_not_block(self, rules):
        failures, _unclaimed, _self_owned, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "DISMISSED", "bob": "APPROVED"},
        )
        assert failures == []

    def test_author_requested_changes_not_counted(self):
        rules = gate.parse_codeowners("/a @solo\n")
        failures, _unclaimed, self_owned, _ignored = gate.check_approvals(
            1, "solo", rules, ["a/x.py"], {"solo": "CHANGES_REQUESTED"}
        )
        assert failures == [] and self_owned == ["a/x.py"]


class TestContentlessExemption:
    """Zero-byte files (empty __init__.py, .gitkeep) are exempt."""

    def test_empty_file_is_exempt(self, tmp_path, monkeypatch):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "__init__.py").write_bytes(b"")
        (tmp_path / "sub" / "real.py").write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        rules = gate.parse_codeowners("/sub @a\n")
        # only the empty file would otherwise be unowned; real file owned
        assert gate.check_coverage(rules, ["sub/real.py"]) == ([], [])

    def test_unowned_empty_file_still_exempt(self, tmp_path, monkeypatch):
        (tmp_path / "__init__.py").write_bytes(b"")
        monkeypatch.chdir(tmp_path)
        rules = gate.parse_codeowners("# TODO(owner): /pkg\n# /pkg\n")
        assert gate.check_coverage(rules, ["__init__.py"]) == ([], [])

    def test_nonempty_file_needs_owner(self, tmp_path, monkeypatch):
        (tmp_path / "module.py").write_text("# license header\n")
        monkeypatch.chdir(tmp_path)
        rules = gate.parse_codeowners("# TODO(owner): /pkg\n# /pkg\n")
        errors, warnings = gate.check_coverage(rules, ["module.py"])
        assert errors == [] and warnings == ["file has no CODEOWNERS owner: module.py"]

    def test_missing_file_is_not_exempt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rules = gate.parse_codeowners("# TODO(owner): /pkg\n# /pkg\n")
        errors, warnings = gate.check_coverage(rules, ["ghost.py"])
        assert errors == [] and warnings == ["file has no CODEOWNERS owner: ghost.py"]


class TestNoRulePollution:
    """resolve_owners must never mutate the parsed rules (regression: an
    aliasing bug made resolving one path inject owners into ownerless TODO
    rules, polluting all later resolutions)."""

    def test_todo_rule_not_polluted_by_active_rule(self):
        rules = gate.parse_codeowners("# TODO(owner): /pkg\n# /pkg\n/pkg/sub @alice\n")
        snapshot = [(r.pattern, list(r.owners)) for r in rules]
        # resolve a path that matches both the TODO and the active rule
        gate.resolve_owners(rules, "pkg/sub/x.py")
        after = [(r.pattern, list(r.owners)) for r in rules]
        assert snapshot == after

    def test_later_resolutions_independent(self):
        rules = gate.parse_codeowners("# TODO(owner): /pkg\n# /pkg\n/pkg/sub @alice\n")
        # first resolve pollutes in the buggy version
        gate.resolve_owners(rules, "pkg/sub/x.py")
        # a path covered ONLY by the ownerless TODO must stay unowned
        owners, skipped = gate.resolve_owners(rules, "pkg/other.py")
        assert owners == [] and not skipped


class TestGetPrReviews:
    def test_ghost_user_review_skipped(self, monkeypatch):
        data = [
            {"user": None, "state": "APPROVED"},
            {"user": {"login": "alice"}, "state": "APPROVED"},
        ]
        monkeypatch.setattr(gate, "gh_api", lambda endpoint, **kw: data)
        assert gate.get_pr_reviews(1) == {"alice": "APPROVED"}


class TestDismissalSurvivesPush:
    """Approvals survive pushes: the repo ruleset auto-dismisses reviews on
    push and gate authors cannot change that setting, so DISMISSED reviews
    are ignored and the previous state persists."""

    def test_approval_survives_dismissal(self, monkeypatch):
        # GitHub rewrites the review state to DISMISSED on push; the gate
        # must treat a dismissed review as an approval (repo policy).
        data = [
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "alice"}, "state": "DISMISSED"},  # auto-dismiss on push
        ]
        monkeypatch.setattr(gate, "gh_api", lambda endpoint, **kw: data)
        assert gate.get_pr_reviews(1) == {"alice": "APPROVED"}

    def test_dismissed_only_review_counts_as_approval(self, monkeypatch):
        # The API no longer exposes the original APPROVED record once
        # dismissed; the DISMISSED entry is all we have.
        data = [{"user": {"login": "bob"}, "state": "DISMISSED"}]
        monkeypatch.setattr(gate, "gh_api", lambda endpoint, **kw: data)
        assert gate.get_pr_reviews(1) == {"bob": "APPROVED"}

    def test_changes_requested_overrides_approval(self, monkeypatch):
        data = [
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "alice"}, "state": "CHANGES_REQUESTED"},
        ]
        monkeypatch.setattr(gate, "gh_api", lambda endpoint, **kw: data)
        assert gate.get_pr_reviews(1) == {"alice": "CHANGES_REQUESTED"}

    def test_reapproval_after_changes_requested(self, monkeypatch):
        data = [
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "alice"}, "state": "CHANGES_REQUESTED"},
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "alice"}, "state": "DISMISSED"},
        ]
        monkeypatch.setattr(gate, "gh_api", lambda endpoint, **kw: data)
        assert gate.get_pr_reviews(1) == {"alice": "APPROVED"}

    def test_approved_still_counts_after_dismissal_in_gate(self, rules):
        # DISMISSED entries in the reviews dict no longer occur (they are
        # filtered upstream), but the gate must remain robust to them.
        failures, _unclaimed, _self, _ignored = gate.check_approvals(
            1,
            "author",
            rules,
            ["vllm_ascend/device/a.py"],
            {"alice": "DISMISSED", "bob": "APPROVED"},
        )
        assert failures == []


class TestIgnoreOwner:
    """`/ignore-owner` waive: a write+ maintainer's command comment skips the
    approval gate for matching paths; the coverage health check is unaffected."""

    def _approvals(self, rules, files, reviews, ignore):
        return gate.check_approvals(1, "author", rules, files, reviews, ignore)

    def test_ignored_path_skipped(self, rules):
        failures, _un, _self, ignored = self._approvals(
            rules,
            ["vllm_ascend/device/a.py"],
            {},
            ("paths", ["vllm_ascend/device/a.py"], "alice"),
        )
        assert failures == [] and ignored == ["vllm_ascend/device/a.py"]

    def test_partial_ignore_still_gates_others(self, rules):
        failures, _un, _self, ignored = self._approvals(
            rules,
            ["vllm_ascend/device/a.py", "vllm_ascend/ops/b.py"],
            {},
            ("paths", ["vllm_ascend/device/a.py"], "alice"),
        )
        assert ignored == ["vllm_ascend/device/a.py"]
        assert len(failures) == 1 and "ops/b.py" in failures[0]

    def test_all_ignore_passes_everything(self, rules):
        failures, _un, _self, ignored = self._approvals(
            rules,
            ["vllm_ascend/device/a.py", "vllm_ascend/ops/b.py"],
            {},
            ("all", [], "alice"),
        )
        assert failures == [] and sorted(ignored) == [
            "vllm_ascend/device/a.py",
            "vllm_ascend/ops/b.py",
        ]

    def test_dir_pattern_ignores_subtree(self, rules):
        failures, _un, _self, ignored = self._approvals(
            rules,
            ["vllm_ascend/device/sub/a.py"],
            {},
            ("paths", ["vllm_ascend/device"], "alice"),
        )
        assert failures == [] and ignored == ["vllm_ascend/device/sub/a.py"]

    def test_glob_pattern_matches_single_segment(self, rules):
        failures, unclaimed, _self, ignored = self._approvals(
            rules,
            ["vllm_ascend/ops/a.py", "vllm_ascend/ops/sub/b.py"],
            {},
            ("paths", ["vllm_ascend/ops/*.py"], "alice"),
        )
        assert ignored == ["vllm_ascend/ops/a.py"]
        # sub/b.py never matched the single-segment glob, and it is unowned
        # in SAMPLE, so it is a notice rather than a failure.
        assert failures == [] and unclaimed == ["vllm_ascend/ops/sub/b.py"]

    def test_none_spec_does_not_ignore(self, rules):
        failures, _un, _self, ignored = self._approvals(rules, ["vllm_ascend/device/a.py"], {}, ("none", [], ""))
        assert failures and ignored == []

    def test_clear_spec_does_not_ignore(self, rules):
        failures, _un, _self, ignored = self._approvals(rules, ["vllm_ascend/device/a.py"], {}, ("clear", [], ""))
        assert failures and ignored == []

    def test_ignore_does_not_affect_coverage(self, rules):
        # The waive is per-PR for the approval gate only; CODEOWNERS health
        # has no /ignore-owner escape hatch.
        tracked = [
            "vllm_ascend/device/a.py",
            "vllm_ascend/ops/b.py",
            "docs/x.md",
            "tests/ut/t.py",
        ]
        errors, _warnings = gate.check_coverage(rules, tracked)
        assert errors == []


class TestGetIgnoreSpec:
    """Parsing of /ignore-owner comments: the latest valid human command wins;
    the commenter needs write+; bot/deleted/permissionless comments are skipped
    rather than revoking an earlier valid waiver."""

    def _comment(self, body, login="alice", user_type="User"):
        return {"user": {"login": login, "type": user_type}, "body": body}

    def _run(self, monkeypatch, comments, role="write", roles=None):
        # gh_api serves the comments list for the comments endpoint and the
        # permission object for the collaborators endpoint. `role` is the
        # fallback for any commenter without a per-user entry in `roles`.
        role_map = roles or {}

        def fake_api(endpoint, **kw):
            if "comments" in endpoint:
                return comments
            return {"role_name": role_map.get(kw.get("user"), role)}

        monkeypatch.setattr(gate, "gh_api", fake_api)
        return gate.get_ignore_spec(1)

    def test_no_command(self, monkeypatch):
        assert self._run(monkeypatch, [self._comment("looks good")]) == ("none", [], "")

    def test_paths_command(self, monkeypatch):
        kind, paths, by = self._run(monkeypatch, [self._comment("/ignore-owner docs/a.md docs/b.md")])
        assert (kind, paths, by) == ("paths", ["docs/a.md", "docs/b.md"], "alice")

    def test_bare_command_ignores_all(self, monkeypatch):
        assert self._run(monkeypatch, [self._comment("/ignore-owner")]) == (
            "all",
            [],
            "alice",
        )

    def test_clear_command(self, monkeypatch):
        assert self._run(monkeypatch, [self._comment("/ignore-owner-clear")]) == (
            "clear",
            [],
            "",
        )

    def test_command_with_prose_below(self, monkeypatch):
        kind, paths, _by = self._run(monkeypatch, [self._comment("/ignore-owner docs/a.md\nsmall docs fix")])
        assert kind == "paths" and paths == ["docs/a.md"]

    def test_bot_comment_skipped(self, monkeypatch):
        comments = [self._comment("/ignore-owner", login="bot", user_type="Bot")]
        assert self._run(monkeypatch, comments) == ("none", [], "")

    def test_latest_command_replaces_earlier(self, monkeypatch):
        comments = [
            self._comment("/ignore-owner docs/a.md"),
            self._comment("/ignore-owner"),
        ]
        assert self._run(monkeypatch, comments)[0] == "all"

    def test_other_slash_commands_do_not_match(self, monkeypatch):
        # Only /ignore-owner and /ignore-owner-clear count; other /-prefixed
        # lines (e.g. someone commenting /run later) keep the ignore intact.
        comments = [
            self._comment("/ignore-owner docs/a.md"),
            self._comment("/run"),
        ]
        assert self._run(monkeypatch, comments) == ("paths", ["docs/a.md"], "alice")

    def test_driveby_command_does_not_revoke_valid_ignore(self, monkeypatch):
        # Alice (write) waives docs/a.md; a triage-level Bob then comments a
        # bare /ignore-owner. Bob is skipped, so Alice's waiver survives - a
        # permissionless commenter can neither grant nor revoke an ignore.
        comments = [
            self._comment("/ignore-owner docs/a.md", login="alice"),
            self._comment("/ignore-owner", login="bob"),
        ]
        assert self._run(monkeypatch, comments, roles={"alice": "write", "bob": "triage"}) == (
            "paths",
            ["docs/a.md"],
            "alice",
        )

    def test_write_permission_required(self, monkeypatch):
        for role in ("read", "triage"):
            assert self._run(monkeypatch, [self._comment("/ignore-owner")], role=role)[0] == "none"

    def test_write_maintain_admin_allowed(self, monkeypatch):
        for role in ("write", "maintain", "admin"):
            assert self._run(monkeypatch, [self._comment("/ignore-owner")], role=role)[0] == "all"

    def test_legacy_permission_field_counts(self, monkeypatch):
        # A custom role whose `role_name` is not a standard name but whose
        # legacy `permission` maps to write still grants the waiver.
        def fake_api(endpoint, **kw):
            if "comments" in endpoint:
                return [self._comment("/ignore-owner")]
            return {"role_name": "custom-auditor", "permission": "write"}

        monkeypatch.setattr(gate, "gh_api", fake_api)
        assert gate.get_ignore_spec(1) == ("all", [], "alice")

    def test_non_collaborator_commander_is_invalid(self, monkeypatch):
        # Anyone can comment on a public repo; the collaborators API returns
        # 404 for a non-collaborator and that must be "no ignore", never a
        # gate crash.
        import urllib.error

        def fake_api(endpoint, **kw):
            if "comments" in endpoint:
                return [self._comment("/ignore-owner", login="driveby")]
            raise urllib.error.HTTPError("url", 404, "not found", {}, None)

        monkeypatch.setattr(gate, "gh_api", fake_api)
        assert gate.get_ignore_spec(1) == ("none", [], "")

    def test_deleted_account_commander_is_invalid(self, monkeypatch):
        # `user` is null for a deleted account: the permission check must not
        # be skipped, or the ignore would resurrect without an owner.
        comment = {"user": None, "body": "/ignore-owner"}
        assert self._run(monkeypatch, [comment]) == ("none", [], "")

    def test_path_is_ignored_normalizes_leading_slash(self):
        assert gate.path_is_ignored("docs/a.md", "paths", ["/docs/a.md"])
        assert gate.path_is_ignored("docs/sub/b.md", "paths", ["docs/"])


class TestIsWritePlus:
    """is_write_plus reads the collaborators permission endpoint: write+ only,
    404 (non-collaborator) is False, 403 warns instead of failing silently."""

    def _run(self, monkeypatch, payload=None, error=None):
        def fake_api(endpoint, **kw):
            assert "permission" in endpoint
            if error is not None:
                raise error
            return payload

        monkeypatch.setattr(gate, "gh_api", fake_api)
        return gate.is_write_plus("alice")

    def test_write_maintain_admin_allowed(self, monkeypatch):
        for role in ("write", "maintain", "admin"):
            assert self._run(monkeypatch, {"role_name": role}) is True

    def test_read_and_triage_denied(self, monkeypatch):
        for role in ("read", "triage", "none"):
            assert self._run(monkeypatch, {"role_name": role}) is False

    def test_legacy_permission_field_counts(self, monkeypatch):
        assert self._run(monkeypatch, {"role_name": "custom", "permission": "write"})
        assert self._run(monkeypatch, {"role_name": "custom", "permission": "admin"})

    def test_non_collaborator_404_is_false(self, monkeypatch):
        import urllib.error

        error = urllib.error.HTTPError("url", 404, "not found", {}, None)
        assert self._run(monkeypatch, error=error) is False

    def test_forbidden_403_warns_and_is_false(self, monkeypatch, capsys):
        import urllib.error

        error = urllib.error.HTTPError("url", 403, "forbidden", {}, None)
        assert self._run(monkeypatch, error=error) is False
        assert "::warning::" in capsys.readouterr().err

    def test_other_http_errors_raise(self, monkeypatch):
        import urllib.error

        error = urllib.error.HTTPError("url", 500, "boom", {}, None)
        with pytest.raises(urllib.error.HTTPError):
            self._run(monkeypatch, error=error)


class TestIgnoreOwnerIntegration:
    """Regression: the real chain get_ignore_spec (3-tuple) -> check_approvals
    -> path_is_ignored. A tuple-shape mismatch here once crashed the gate for
    every PR (ValueError: too many values to unpack)."""

    def test_ignore_spec_feeds_check_approvals(self, monkeypatch, rules):
        def fake_api(endpoint, **kw):
            if "comments" in endpoint:
                return [
                    {
                        "user": {"login": "alice", "type": "User"},
                        "body": "/ignore-owner vllm_ascend/device",
                    }
                ]
            if "permission" in endpoint:
                return {"role_name": "write"}
            raise AssertionError(f"unexpected endpoint: {endpoint}")

        monkeypatch.setattr(gate, "gh_api", fake_api)
        ignore_spec = gate.get_ignore_spec(1)
        assert len(ignore_spec) == 3
        failures, _un, _self, ignored = gate.check_approvals(
            1, "author", rules, ["vllm_ascend/device/a.py"], {}, ignore_spec
        )
        assert failures == [] and ignored == ["vllm_ascend/device/a.py"]
