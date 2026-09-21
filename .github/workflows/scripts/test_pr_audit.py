import importlib.util
import pathlib
import sys
import unittest


SCRIPT = pathlib.Path(__file__).with_name("pr_audit.py")
SPEC = importlib.util.spec_from_file_location("pr_audit", SCRIPT)
pr_audit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = pr_audit
SPEC.loader.exec_module(pr_audit)


class FakeClient:
    def __init__(self, issues):
        self.issues = issues

    def get_issue(self, number):
        return self.issues.get(number)


class AuditPullRequestTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "title_min_length": 5,
            "body_min_length": 15,
            "max_added_lines": 1000,
            "issue_required_title_prefixes": ["[BugFix]", "[Feature]"],
            "issue_keywords": ["fixes", "closes", "resolves", "refs"],
            "allow_milestone": True,
            "code_extensions": [".py"],
            "test_path_substrings": ["test"],
            "forbidden_links": ["https://forbidden.example"],
        }

    def _audit(self, title="[BugFix] Correct issue", body="Fixes #123\nEnough description.", files=None):
        return pr_audit.audit_pull_request(
            {"title": title, "body": body},
            {"milestone": None}, files or [{"filename": "tests/test_fix.py", "additions": 2}],
            self.config, FakeClient({123: {"number": 123}}),
        )

    def test_linked_issue_satisfies_bugfix_rule(self):
        results = self._audit()
        self.assertTrue(next(result for result in results if result.name == "linked_issue").passed)

    def test_missing_issue_fails_bugfix_rule(self):
        results = self._audit(body="Enough description without a reference.")
        self.assertFalse(next(result for result in results if result.name == "linked_issue").passed)

    def test_missing_tests_is_warning(self):
        results = self._audit(files=[{"filename": "vllm_ascend/fix.py", "additions": 2}])
        test_result = next(result for result in results if result.name == "tests")
        self.assertFalse(test_result.passed)
        self.assertEqual(test_result.severity, "warning")

    def test_forbidden_link_fails(self):
        results = self._audit(body="Fixes #123\nhttps://forbidden.example")
        self.assertFalse(next(result for result in results if result.name == "forbidden_links").passed)

    def test_comment_contains_single_marker(self):
        comment = pr_audit.audit_comment("octocat", self._audit())
        self.assertEqual(comment.count(pr_audit.COMMENT_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
