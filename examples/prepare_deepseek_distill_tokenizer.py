# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Create a separate tokenizer directory preserving checkpoint byte-level BPE.

Some Transformers 5 releases replace this checkpoint's byte-level pipeline with
a SentencePiece-style pipeline. Do not modify weights or the original download.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROBES = ("hello world", "two  spaces\nand a newline", "请用中文解释天空为什么是蓝色的。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    definition = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "llama" or metadata.get("tokenizer_class") != "LlamaTokenizerFast":
        parser.error("Expected a Llama checkpoint with LlamaTokenizerFast metadata")
    if definition.get("decoder", {}).get("type") != "ByteLevel":
        parser.error("This procedure is only for a ByteLevel tokenizer.json")

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True)
    for name in ("tokenizer.json", "special_tokens_map.json", "chat_template.jinja"):
        path = source / name
        if path.exists():
            shutil.copy2(path, args.output / name)
    metadata["tokenizer_class"] = "PreTrainedTokenizerFast"
    (args.output / "tokenizer_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reference = Tokenizer.from_file(str(source / "tokenizer.json"))
    actual = AutoTokenizer.from_pretrained(args.output)
    probes = []
    for text in PROBES:
        expected_ids = reference.encode(text).ids
        actual_ids = actual.encode(text)
        if not expected_ids or actual_ids != expected_ids or actual.decode(actual_ids) != text:
            raise RuntimeError(f"Tokenizer pipeline mismatch for {text!r}")
        probes.append({"text": text, "token_ids": actual_ids})
    if actual.chat_template != metadata.get("chat_template"):
        raise RuntimeError("Chat template was not preserved")
    messages = [{"role": "user", "content": PROBES[0]}]
    rendered = actual.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_ids = actual.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if chat_ids != reference.encode(rendered, add_special_tokens=False).ids:
        raise RuntimeError("Chat template tokenization differs from the checkpoint pipeline")
    manifest = {
        "source": str(source),
        "tokenizer_json_sha256": hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest(),
        "metadata_override": {"tokenizer_class": "PreTrainedTokenizerFast"},
        "probes": probes,
        "chat_token_ids": chat_ids,
    }
    (args.output / "compatibility-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "verified", "output": str(args.output), "probes": len(probes)}))


if __name__ == "__main__":
    main()
