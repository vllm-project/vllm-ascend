# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Compare ranked-logprob probes emitted by compare_qwen35_mtp.py."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(directory):
    records = []
    for runner in (1, 2):
        path = directory / f"v{runner}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"{path} is not complete")
        probe = result.get("logits_probe") or result.get("full_logits_probe")
        if not probe:
            raise ValueError(f"{path} does not contain a logits probe")
        records.append(probe[0])
    return records


def logprob_map(step):
    return {int(token): float(value["logprob"]) for token, value in step.items()}


def renormalized_probs(values):
    maximum = max(values.values())
    total = math.fsum(math.exp(value - maximum) for value in values.values())
    return {token: math.exp(value - maximum) / total for token, value in values.items()}


def kl_divergence(left, right):
    return math.fsum(
        probability * math.log(probability / right[token]) for token, probability in left.items() if probability > 0.0
    )


def top_k_overlap(left, right, k):
    left_top = set(sorted(left, key=left.get, reverse=True)[:k])
    right_top = set(sorted(right, key=right.get, reverse=True)[:k])
    return len(left_top & right_top) / k


def compare_positions(v1_steps, v2_steps):
    positions = []
    for index, (v1_step, v2_step) in enumerate(zip(v1_steps, v2_steps)):
        v1_values = logprob_map(v1_step)
        v2_values = logprob_map(v2_step)
        common_tokens = set(v1_values) & set(v2_values)
        if not common_tokens:
            raise ValueError(f"Position {index} has no overlapping tokens")
        v1_aligned = {token: v1_values[token] for token in common_tokens}
        v2_aligned = {token: v2_values[token] for token in common_tokens}

        v1_probs = renormalized_probs(v1_aligned)
        v2_probs = renormalized_probs(v2_aligned)
        returned_size = min(len(v1_values), len(v2_values))
        record = {
            "position": index,
            "v1_token_count": len(v1_values),
            "v2_token_count": len(v2_values),
            "common_token_count": len(common_tokens),
            "top_k_overlap": {
                k: top_k_overlap(v1_values, v2_values, k) for k in (20, 100, 1000, returned_size) if k <= returned_size
            },
            "kl_returned_set_renormalized": max(0.0, kl_divergence(v1_probs, v2_probs)),
        }
        positions.append(record)
    return positions


def mean_metric(positions, key, nested_key=None):
    values = [row[key][nested_key] if nested_key is not None else row[key] for row in positions if key in row]
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing v1.json and v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1, v2 = load_records(args.directory)
    if v1["token_ids"] != v2["token_ids"]:
        raise ValueError("Probe token histories differ; distribution positions are not comparable")

    positions = compare_positions(v1["logprobs"], v2["logprobs"])
    returned_size = min(
        min(len(step) for step in v1["logprobs"]),
        min(len(step) for step in v2["logprobs"]),
    )
    summary = {
        "scope": (
            f"KL uses at most {returned_size} returned logprobs per position, "
            "renormalized within the returned set; they are not full-vocabulary KL"
        ),
        "token_ids": v1["token_ids"],
        "positions": positions,
        "mean": {
            "top_k_overlap": {
                k: mean_metric(positions, "top_k_overlap", k)
                for k in (20, 100, 1000, returned_size)
                if k <= returned_size
            },
            "kl_returned_set_renormalized": mean_metric(positions, "kl_returned_set_renormalized"),
        },
        "inputs": {
            "v1_sha256": sha256(args.directory / "v1.json"),
            "v2_sha256": sha256(args.directory / "v2.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
