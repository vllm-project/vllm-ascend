# SPDX-License-Identifier: Apache-2.0
"""Tripwires against publishing timings from mismatched prompts or corrupt data."""

import hashlib
import json
import sys

import pytest

from benchmarks.uno import run_math_eval
from benchmarks.uno.sglang_math_data import DATASET_REVISIONS, get_benchmark


def test_prompt_formatter_preserves_caller_and_applies_math_instruction():
    messages = [{"role": "system", "content": "Existing instruction"}, {"role": "user", "content": "2 + 2?"}]
    before = json.loads(json.dumps(messages))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, chat, *, tokenize, **options):
            calls.append((chat, options))
            return [1, 2, 3] if tokenize else "rendered prompt"

    ids, rendered = run_math_eval.format_prompt(Tokenizer(), messages, get_benchmark("math500"))
    assert messages == before
    assert ids == [1, 2, 3] and rendered == "rendered prompt"
    assert calls[0][0][0]["content"].endswith("\n\nExisting instruction")
    assert "\\boxed{}" in calls[0][0][0]["content"]
    assert calls[0][1]["reasoning_effort"] == "high"
    assert calls[0][1]["add_generation_prompt"] is True


@pytest.mark.parametrize("fault", ["hash", "revision", "count"])
def test_corrupt_data_fails_before_loading_any_model(tmp_path, monkeypatch, fault):
    data = b'{"row": 0}\n'
    (tmp_path / "math500.jsonl").write_bytes(data)
    record = dict(
        benchmark="math500",
        repo="HuggingFaceH4/MATH-500",
        revision=DATASET_REVISIONS["HuggingFaceH4/MATH-500"],
        prepared_sha256=hashlib.sha256(data).hexdigest(),
    )
    if fault == "hash":
        record["prepared_sha256"] = "incorrect"
    elif fault == "revision":
        record["revision"] = "main"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([record]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_math_eval",
            "--model-path",
            "missing-model",
            "--mode",
            "ar",
            "--benchmark",
            "math500",
            "--data-root",
            str(tmp_path),
            "--data-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-running-requests",
            "1",
        ],
    )
    with pytest.raises(AssertionError):
        run_math_eval.main()
    assert not (tmp_path / "out").exists()
