# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline fixture authoring only. The regression test never tokenizes inputs."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from tests.e2e.glm53_flash.checkpoint import file_hash, write_json

BOUNDARY_LENGTHS = (127, 128, 129, 2047, 2048, 2051, 2052, 4097)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace reviewed prompt fixtures")
    tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
    texts = {
        "chinese": "请用中文解释为什么春天白天逐渐变长。回答：",
        "math": "Compute 17 * 23 and explain the calculation step by step. Answer:",
        "code": "Write a Python function that returns the greatest common divisor of two integers.\n```python\n",
    }
    cases = [
        {"name": name, "text": text, "token_ids": tokenizer.encode(text, add_special_tokens=False)}
        for name, text in texts.items()
    ]
    pattern = tokenizer.encode(
        "第七章：城市与河流。The river passes the library, a stone bridge, and a quiet garden. "
        "There are 17 red flowers and 23 blue flowers. Python: total = sum(range(19)).\n",
        add_special_tokens=False,
    )
    for length in BOUNDARY_LENGTHS:
        ids = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
        cases.append({"name": f"boundary_{length}", "token_ids": ids})
    write_json(
        args.output,
        {"schema_version": 1, "tokenizer_sha256": file_hash(args.source_dir / "tokenizer.json"), "cases": cases},
    )


if __name__ == "__main__":
    main()
