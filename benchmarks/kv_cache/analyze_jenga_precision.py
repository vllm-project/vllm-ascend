#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(reference: dict, candidate: dict) -> dict:
    for field in (
        "seed",
        "output_tokens",
        "gdn_decode_backend",
        "speculative_config",
    ):
        if reference.get(field) != candidate.get(field):
            raise ValueError(
                f"precision runs use different {field}: {reference.get(field)!r} != {candidate.get(field)!r}"
            )

    reference_cases = {item["case"]: item for item in reference["results"]}
    candidate_cases = {item["case"]: item for item in candidate["results"]}
    if reference_cases.keys() != candidate_cases.keys():
        raise ValueError("precision runs contain different cases")

    rows = []
    for case in reference_cases:
        left = reference_cases[case]
        right = candidate_cases[case]
        if left["prompt_tokens_reported"] != right["prompt_tokens_reported"]:
            raise ValueError(f"prompt token count differs for {case}")
        if left["prompt_token_ids_sha256"] != right["prompt_token_ids_sha256"]:
            raise ValueError(f"prompt token IDs differ for {case}")

        token_pairs = list(zip(left["token_ids"], right["token_ids"]))
        first_token_mismatch = next(
            (index for index, pair in enumerate(token_pairs) if pair[0] != pair[1]),
            None,
        )
        if first_token_mismatch is None and len(left["token_ids"]) != len(right["token_ids"]):
            first_token_mismatch = len(token_pairs)

        rows.append(
            {
                "case": case,
                "prompt_tokens": left["prompt_tokens_reported"],
                "text_equal": left["text"] == right["text"],
                "token_ids_equal": left["token_ids"] == right["token_ids"],
                "first_token_mismatch": first_token_mismatch,
                "matching_prefix_tokens": (
                    first_token_mismatch if first_token_mismatch is not None else len(left["token_ids"])
                ),
            }
        )
    return {
        "reference": reference["run_label"],
        "candidate": candidate["run_label"],
        "all_text_equal": all(row["text_equal"] for row in rows),
        "all_token_ids_equal": all(row["token_ids_equal"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-a", type=Path, required=True)
    parser.add_argument("--static-b", type=Path, required=True)
    parser.add_argument("--address", type=Path, required=True)
    parser.add_argument("--address-b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    static_a = load(args.static_a)
    static_b = load(args.static_b)
    address = load(args.address)
    report = {
        "static_restart": compare(static_a, static_b),
        "address_vs_static": compare(static_a, address),
    }
    if args.address_b is not None:
        report["address_restart"] = compare(address, load(args.address_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
