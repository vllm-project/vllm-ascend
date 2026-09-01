from pathlib import Path

import pytest

from tools.check_log_quality import Rules, check_file, load_rules


def _write_sample(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _issues_for(source: str, tmp_path: Path, rel_path: str = "vllm_ascend/sample.py") -> list:
    sample = tmp_path / rel_path
    _write_sample(sample, source)
    rules = load_rules(Path("tools/log_quality_rules.toml"))
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        return check_file(sample, rules, incremental=False)
    finally:
        os.chdir(old_cwd)


def test_privacy_sensitive_argument_fails(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error('failed', messages)\n",
        tmp_path,
    )
    assert any(issue.rule == "LQ001-privacy-in-argument" for issue in issues)


def test_privacy_allowlist_num_tokens_passes(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.info('batch size', num_tokens)\n",
        tmp_path,
    )
    assert not any(issue.rule.startswith("LQ001") for issue in issues)


def test_privacy_allowlist_prompt_hash_passes(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.info('hash', prompt_hash)\n",
        tmp_path,
    )
    assert not any(issue.rule.startswith("LQ001") for issue in issues)


def test_privacy_allowlist_does_not_bypass_other_fields(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error('num_tokens=%s password=%s', num_tokens, password)\n",
        tmp_path,
    )
    assert any(issue.rule == "LQ001-privacy-in-message" for issue in issues)


def test_dynamic_log_message_is_not_flagged_empty(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error(err_msg)\n",
        tmp_path,
    )
    assert not any(issue.rule == "LQ004-empty-log-message" for issue in issues)


def test_short_fstring_with_placeholder_passes(tmp_path: Path):
    issues = _issues_for(
        'from vllm.logger import logger\n\nlogger.error(f"NPU error: {code}")\n',
        tmp_path,
    )
    assert not any(issue.rule == "LQ004-short-log-message" for issue in issues)


def test_non_logger_error_method_is_ignored(tmp_path: Path):
    issues = _issues_for(
        "class Parser:\n    def error(self, msg):\n        pass\n\nParser().error('failed')\n",
        tmp_path,
    )
    assert issues == []


def test_vague_error_message_fails(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error('failed')\n",
        tmp_path,
    )
    assert any(issue.rule == "LQ003-vague-log-message" for issue in issues)


def test_error_with_parameter_passes(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error('load model failed: %s', err)\n",
        tmp_path,
    )
    assert not any(issue.level == "error" for issue in issues)


def test_incremental_skips_unchanged_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = "\n".join(
        [
            "from vllm.logger import logger",
            "",
            "logger.error('failed')",
            "logger.error('load model failed: %s', err)",
        ]
    )
    sample = tmp_path / "vllm_ascend/sample.py"
    _write_sample(sample, source)
    rules = Rules()
    monkeypatch.setattr(
        "tools.check_log_quality.parse_staged_changed_lines",
        lambda _rel_path: {4},
    )
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        issues = check_file(sample, rules, incremental=True)
    finally:
        os.chdir(old_cwd)
    assert not any(issue.rule == "LQ003-vague-log-message" for issue in issues)
    assert not issues


def test_out_of_scope_file_is_skipped(tmp_path: Path):
    issues = _issues_for(
        "from vllm.logger import logger\n\nlogger.error('failed')\n",
        tmp_path,
        rel_path="tests/sample.py",
    )
    assert issues == []
