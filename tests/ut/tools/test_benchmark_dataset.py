from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmark_dataset import generate_benchmark_dataset, get_prefix_dataset_path


class FakeTokenizer:
    """Whitespace tokenizer with a reversible vocabulary for generator tests."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            token_ids.append(self.token_to_id[token])
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


def _read_questions(dataset_dir: str) -> list[str]:
    dataset_file = Path(dataset_dir) / "test.jsonl"
    return [json.loads(line)["question"] for line in dataset_file.read_text(encoding="utf-8").splitlines()]


def test_generate_fixed_dataset(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    tokenizer = FakeTokenizer()
    config = {
        "type": "fixed",
        "input_len": 16,
        "num_samples": 5,
        "seed": 7,
    }

    dataset_dir = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=tokenizer,
    )

    questions = _read_questions(dataset_dir)
    assert len(questions) == 5
    assert len(set(questions)) == 5
    assert all(len(tokenizer.encode(question)) == 16 for question in questions)
    assert (Path(dataset_dir) / "metadata.json").is_file()


def test_generate_prefix_dataset(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    tokenizer = FakeTokenizer()
    config = {
        "type": "prefix",
        "input_len": 20,
        "num_samples": 6,
        "prefix_ratio": 0.5,
        "prefix_num": 2,
        "seed": 11,
    }

    dataset_dir = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=tokenizer,
    )

    tokenized = [tokenizer.encode(question) for question in _read_questions(dataset_dir)]
    assert all(len(token_ids) == 20 for token_ids in tokenized)
    assert tokenized[0][:10] == tokenized[2][:10] == tokenized[4][:10]
    assert tokenized[1][:10] == tokenized[3][:10] == tokenized[5][:10]
    assert tokenized[0][:10] != tokenized[1][:10]
    assert tokenized[0][10:] != tokenized[2][10:]


def test_generate_full_prefix_dataset(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    tokenizer = FakeTokenizer()
    config = {
        "type": "prefix",
        "input_len": 12,
        "num_samples": 4,
        "prefix_ratio": 1.0,
        "prefix_num": 2,
    }

    dataset_dir = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=tokenizer,
    )

    questions = _read_questions(dataset_dir)
    assert questions[0] == questions[2]
    assert questions[1] == questions[3]
    assert questions[0] != questions[1]
    assert all(len(tokenizer.encode(question)) == 12 for question in questions)


def test_generate_prefix_prewarm_dataset(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    tokenizer = FakeTokenizer()
    config = {
        "type": "prefix",
        "input_len": 20,
        "num_samples": 6,
        "prefix_ratio": 0.5,
        "prefix_num": 2,
        "prewarm": True,
        "dp": 3,
        "seed": 11,
    }

    dataset_dir = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=tokenizer,
    )

    prefix_questions = _read_questions(get_prefix_dataset_path(dataset_dir))
    assert len(prefix_questions) == 6
    assert prefix_questions[0] == prefix_questions[1] == prefix_questions[2]
    assert prefix_questions[3] == prefix_questions[4] == prefix_questions[5]
    assert prefix_questions[0] != prefix_questions[3]
    assert all(len(tokenizer.encode(question)) == 10 for question in prefix_questions)


def test_reuse_complete_cached_dataset(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    tokenizer = FakeTokenizer()
    config = {"type": "fixed", "input_len": 8, "num_samples": 2}
    first_path = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=tokenizer,
    )

    second_path = generate_benchmark_dataset(
        model_path="test-model",
        config=config,
        cache_root=tmp_path,
        tokenizer=object(),
    )

    assert second_path == first_path


@pytest.mark.parametrize(
    "config, error",
    [
        ({"type": "variable", "input_len": 8, "num_samples": 2}, "type"),
        ({"type": "fixed", "input_len": 0, "num_samples": 2}, "input_len"),
        ({"type": "fixed", "input_len": 8, "num_samples": 0}, "num_samples"),
        ({"type": "fixed", "input_len": 8, "num_samples": 2, "prefix_ratio": 0.5}, "prefix_ratio"),
        ({"type": "fixed", "input_len": 8, "num_samples": 2, "prewarm": True}, "prewarm"),
        ({"type": "prefix", "input_len": 8, "num_samples": 2, "prefix_ratio": 0.5, "dp": 0}, "dp"),
        ({"type": "prefix", "input_len": 8, "num_samples": 2, "prefix_ratio": 0.5, "prewarm": "yes"}, "boolean"),
        ({"type": "prefix", "input_len": 8, "num_samples": 2, "prefix_ratio": 0}, "prefix_ratio"),
        ({"type": "prefix", "input_len": 2, "num_samples": 2, "prefix_ratio": 0.1}, "too small"),
        ({"type": "prefix", "input_len": 8, "num_samples": 2, "prefix_ratio": 1.1}, "prefix_ratio"),
    ],
)
def test_reject_invalid_generator_config(tmpdir, config: dict, error: str) -> None:
    tmp_path = Path(str(tmpdir))
    with pytest.raises((KeyError, ValueError), match=error):
        generate_benchmark_dataset(
            model_path="test-model",
            config=config,
            cache_root=tmp_path,
            tokenizer=FakeTokenizer(),
        )
