#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
"""Unit tests for .github/workflows/scripts/ai_review.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_review as ai  # noqa: E402


class TestSignature:
    def test_round_trip(self):
        state = ai.State(state="reviewing", round=2, total=6)
        sig = state.signature("abc123def456")
        assert sig == ("<!-- ai-review: state=reviewing | round=2/5 | total=6 | head=abc123def456 -->")

    def test_parse(self):
        comment = "some findings ...\n\n<!-- ai-review: state=waiting-human | round=6/5 | total=7 | head=deadbeef -->"
        m = ai.SIGNATURE_RE.search(comment)
        fields = dict(ai.FIELD_RE.findall(m.group(1)))
        assert fields["state"] == "waiting-human"
        assert fields["round"] == "6/5"
        assert fields["total"] == "7"

    def test_parse_ignores_comments_without_signature(self):
        assert ai.SIGNATURE_RE.search("a normal comment") is None


class TestParseAllowlist:
    def test_names_and_comments(self):
        text = "\n# comment, ignored\nalice # inline note\n\nbob\n#carol-is-not-one\n"
        assert ai.parse_allowlist(text) == {"alice", "bob"}

    def test_empty(self):
        assert ai.parse_allowlist("") == set()


class TestParseFindings:
    def test_wellformed_block(self):
        text = (
            "some prose\n\n"
            "```finding\n"
            "path: vllm_ascend/foo.py\n"
            "line: 42\n"
            "severity: MUST-FIX\n"
            "rule: L1#no-item-sync\n"
            "body: sync in hot path\n"
            "```\n\n"
            "<!-- ai-review: state=reviewing | round=1/5 | total=1 | head=abc -->"
        )
        findings = ai.parse_findings(text)
        assert len(findings) == 1
        assert findings[0] == {
            "path": "vllm_ascend/foo.py",
            "line": 42,
            "severity": "MUST-FIX",
            "rule": "L1#no-item-sync",
            "body": "sync in hot path",
        }

    def test_multiline_body(self):
        text = "```finding\npath: a.py\nline: 1\nseverity: NIT\nrule: general\nbody: line one\nline two\n```\n"
        assert ai.parse_findings(text)[0]["body"] == "line one\nline two"

    @pytest.mark.parametrize(
        "block",
        [
            "```finding\nline: 1\nseverity: NIT\nrule: general\nbody: x\n```",  # no path
            "```finding\npath: a.py\nline: abc\nseverity: NIT\nrule: general\nbody: x\n```",  # bad line
            "```finding\npath: a.py\nline: 0\nseverity: NIT\nrule: general\nbody: x\n```",  # zero line
            "```finding\npath: a.py\nline: 1\nseverity: NIT\n```",  # missing rule/body
        ],
    )
    def test_malformed_blocks_dropped(self, block):
        assert ai.parse_findings(block) == []

    def test_no_findings(self):
        assert ai.parse_findings("plain comment, no blocks") == []


class TestResolveDomains:
    REGISTRY = """
domains:
  - name: ci
    description: CI workflows
    paths:
      - ".github/**"
      - "tests/**"
  - name: patch
    description: patch framework
    paths:
      - "vllm_ascend/patch/**"
"""

    def test_matches_coarse_globs(self):
        domains = ai.resolve_domains(self.REGISTRY, [".github/workflows/pr_test.yaml"])
        assert [d["name"] for d in domains] == ["ci"]

    def test_first_match_wins_and_multiple_domains(self):
        domains = ai.resolve_domains(self.REGISTRY, ["vllm_ascend/patch/platform/foo.py", ".github/x.yml"])
        # returned in registry order, not file order
        assert [d["name"] for d in domains] == ["ci", "patch"]

    def test_no_match(self):
        assert ai.resolve_domains(self.REGISTRY, ["vllm_ascend/ops/foo.py"]) == []

    def test_invalid_yaml_is_empty_not_crash(self):
        assert ai.resolve_domains("domains: [broken", ["any"]) == []


class TestKnowledgeBaseMap:
    def test_entry_ids_extracted_from_checklists(self, tmp_path):
        kb = tmp_path / ".github" / "agent-review"
        (kb / "L1").mkdir(parents=True)
        (kb / "L2" / "ci").mkdir(parents=True)
        (kb / "L1" / "checklist.md").write_text(
            "# L1\n\n- `L1#no-item-sync`: avoid item()\n- `L1#top-level-imports`: imports\n"
        )
        (kb / "L2" / "ci" / "checklist.md").write_text("# L2\n\n- `L2-ci#container-contract`: declare containers\n")
        (kb / "instructions.md").write_text("# instructions")

        orch = object.__new__(ai.Orchestrator)
        orch.workspace = str(tmp_path)
        kb_map = orch._knowledge_base_map()

        assert "`L1#no-item-sync`" in kb_map
        assert "`L2-ci#container-contract`" in kb_map
        assert "instructions.md" not in kb_map  # not a rule file
        # map lists relative paths the agent can read from the workspace
        assert ".github/agent-review/L1/checklist.md" in kb_map


class TestAgentOutputContract:
    def test_missing_signature_is_appended(self):
        output = "findings here, no signature"

        new_state = ai.State(state="reviewing", round=1, total=1)
        if not ai.SIGNATURE_RE.search(output):
            output += "\n\n" + new_state.signature("abc123")
        assert ai.SIGNATURE_RE.search(output)
        fields = dict(ai.FIELD_RE.findall(ai.SIGNATURE_RE.search(output).group(1)))
        assert fields["state"] == "reviewing"
        assert fields["head"] == "abc123"

    def test_existing_signature_is_replaced(self):
        output = "body\n\n<!-- ai-review: state=reviewing | round=9/5 | total=9 | head=old -->"
        new_state = ai.State(state="passed", round=2, total=10)
        output = ai.SIGNATURE_RE.sub(new_state.signature("newhead"), output, count=1)
        fields = dict(ai.FIELD_RE.findall(ai.SIGNATURE_RE.search(output).group(1)))
        assert fields["state"] == "passed"
        assert fields["head"] == "newhead"


@pytest.mark.parametrize(
    "body,expected",
    [
        ("@ai-review please re-check", True),
        ("just chatting with humans", False),
        ("@ai-reviewer is someone else", True),  # substring mention is fine for gate
        ("", False),
    ],
)
def test_mention_gate(body, expected):
    assert (ai.MENTION_TOKEN in body) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        ("/ai-review-stop", True),
        ("text /ai-review-stop trailing", True),
        ("/ai-review stopped by history", False),
    ],
)
def test_stop_gate(body, expected):
    assert (ai.STOP_COMMAND in body) == expected
