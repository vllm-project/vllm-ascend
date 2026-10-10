# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit offline golden candidate capture; never imported by the CI test."""

import argparse
from pathlib import Path

from tests.e2e.glm53_flash.checkpoint import DERIVED_MANIFEST, read_json, validate_manifest, verify_files, write_json
from tests.e2e.glm53_flash.runtime import FIXTURES, contract, environment_info, load_prompts, run_suite

COLD_STARTS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ascend-commit", required=True)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.candidate.exists():
        parser.error("Refusing to overwrite a golden/candidate")
    prompts = load_prompts()
    projection = read_json(args.model / DERIVED_MANIFEST)
    source = read_json(FIXTURES / "source.json")
    validate_manifest(source)
    if projection["source_id"] != source["source_id"]:
        parser.error("Projection does not match the reviewed source")
    if prompts["tokenizer_sha256"] != source["files"]["tokenizer.json"]["sha256"]:
        parser.error("Prompt tokenizer does not match the reviewed checkpoint")
    verify_files(args.model, projection["files"])
    golden = {
        "schema_version": 1,
        "contract": contract(source, projection, prompts),
        "provenance": {
            "ascend_commit": args.ascend_commit,
            "vllm_commit": args.vllm_commit,
            "image": args.image,
            "environment": environment_info(),
        },
        "validation": [],
    }
    for cold_start in range(COLD_STARTS):
        result = run_suite(args.model, prompts, args.artifacts / f"cold-{cold_start}", golden if cold_start else None)
        if cold_start == 0:
            golden.update(cases=result["cases"], workers=result["workers"])
        golden["validation"].append(result["report"])
    write_json(args.candidate, golden)
    print(f"Three cold starts passed. Review candidate manually: {args.candidate}")


if __name__ == "__main__":
    main()
