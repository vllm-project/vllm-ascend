#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Compare two ``dspark_v1_v2_collect.py`` dumps and emit a Markdown report.

What is compared, and what each result means:

* **Token-sequence identity.** Only meaningful under greedy decoding
  (``temperature=0``) with the same seed. There it is a hard equivalence
  check: any divergence is a regression. Under ``temperature>0`` the two
  runners draw independent randomness, so divergence is expected and the
  distribution metrics below are the meaningful signal instead.
* **Acceptance rates.** Per-position acceptance and mean acceptance length.
  Compared as distributions, not for exact equality.
* **Top-k overlap.** Mean fraction of shared token ids in the per-position
  top-k sets.
* **KL divergence.** Computed over the *union* of the two top-k token sets,
  renormalised to sum to one. The public API exposes only top-k logprobs, so
  this is a truncated-support approximation of the full-vocabulary KL; it is
  comparable across runs with the same ``--logprobs`` but is not an absolute
  full-vocabulary KL. Positions whose supports barely overlap yield large
  values by construction, so read it alongside the overlap column.

Usage::

    python dspark_v1_v2_compare.py --v1 v1_greedy.json --v2 v2_greedy.json \\
        --out report_greedy.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any

# Greedy runs must match exactly; this is the regression gate.
GREEDY_IDENTITY_REQUIRED = True
# Non-greedy acceptance-rate tolerance per position, matching the 0.1 band the
# repo's existing DSpark acceptance baselines use.
ACCEPTANCE_TOLERANCE = 0.1
# An output in which at least this fraction of the symbols is one single symbol (and that has at least
# DEGENERATE_MIN_CHARS symbols) is treated as degenerate, e.g. a prefix followed by '!!!!...'. Symbols are the
# generated token ids when the dump has them, otherwise the non-whitespace characters (so indentation and
# newlines never count). A degenerate *baseline* (v1) is not a valid reference: differences against it must not be
# read as a regression of the candidate. A degenerate *candidate* (v2) against a healthy baseline is a real
# regression signal. Token id 0 repeated can be caused by NaN logits under argmax; that is not verified here.
DEGENERATE_CHAR_FRACTION = 0.6
DEGENERATE_MIN_CHARS = 16
# Exit code used with --fail-on-regression when the comparison is invalid (distinct from a regression).
EXIT_INVALID_COMPARISON = 3


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _degenerate_prompts(dump: dict[str, Any]) -> list[str]:
    """prompt_ids whose output is dominated by one single symbol (see DEGENERATE_CHAR_FRACTION)."""
    bad: list[str] = []
    for output in dump.get("outputs", []):
        ids = output.get("token_ids")
        symbols = list(ids) if ids else [ch for ch in (output.get("text") or "") if not ch.isspace()]
        if len(symbols) < DEGENERATE_MIN_CHARS:
            continue
        if Counter(symbols).most_common(1)[0][1] / len(symbols) >= DEGENERATE_CHAR_FRACTION:
            bad.append(str(output["prompt_id"]))
    return bad


def _first_divergence(left, right) -> int | None:
    """Index of the first differing element, or None when one is a prefix of the
    other. Works for a list[int] of token ids (local mode) or a str of raw text
    (remote mode's fallback, see `_compare_outputs`) - both support zip() and
    element-wise `!=` the same way."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None


def _topk_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return float("nan")
    shared = set(left) & set(right)
    return len(shared) / max(len(left), len(right))


def _truncated_kl(left: dict[str, float], right: dict[str, float]) -> float:
    """KL(left || right) over the union of both top-k supports.

    Missing entries are floored at the smallest observed logprob in that
    distribution, which is an upper bound on their true probability (anything
    outside top-k ranks below the k-th entry). Both sides are renormalised
    afterwards, so the result is a bounded approximation rather than the exact
    full-vocabulary KL.
    """
    if not left or not right:
        return float("nan")
    union = set(left) | set(right)
    left_floor = min(left.values())
    right_floor = min(right.values())

    p = {token: math.exp(left.get(token, left_floor)) for token in union}
    q = {token: math.exp(right.get(token, right_floor)) for token in union}
    p_total = sum(p.values())
    q_total = sum(q.values())
    if p_total <= 0 or q_total <= 0:
        return float("nan")

    divergence = 0.0
    for token in union:
        p_i = p[token] / p_total
        q_i = q[token] / q_total
        if p_i > 0 and q_i > 0:
            divergence += p_i * math.log(p_i / q_i)
    return divergence


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _validate_comparable(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    if v1.get("mode") != v2.get("mode"):
        raise SystemExit(f"collection modes differ: {v1.get('mode')!r} vs {v2.get('mode')!r}")

    outputs1, outputs2 = v1["outputs"], v2["outputs"]
    if len(outputs1) != len(outputs2):
        raise SystemExit(f"output counts differ: v1={len(outputs1)} v2={len(outputs2)}")

    comparison_keys = (set(v1["config"]) | set(v2["config"])) - {"endpoint"}
    mismatched = [key for key in sorted(comparison_keys) if v1["config"].get(key) != v2["config"].get(key)]
    if mismatched:
        raise SystemExit(f"collection configs differ for {mismatched}; rerun both arms with identical settings")

    ids1 = [output["prompt_id"] for output in outputs1]
    ids2 = [output["prompt_id"] for output in outputs2]
    if len(ids1) != len(set(ids1)) or len(ids2) != len(set(ids2)):
        raise SystemExit("duplicate prompt_id values are ambiguous; collect each prompt id once per comparison")
    if ids1 != ids2:
        raise SystemExit(f"prompt ids or order differ: v1={ids1} v2={ids2}")

    per_pos1 = v1["acceptance"]["acceptance_per_pos"]
    per_pos2 = v2["acceptance"]["acceptance_per_pos"]
    if len(per_pos1) != len(per_pos2):
        raise SystemExit(f"acceptance position counts differ: v1={len(per_pos1)} v2={len(per_pos2)}")

    for output1, output2 in zip(outputs1, outputs2, strict=True):
        token_ids1, token_ids2 = output1.get("token_ids"), output2.get("token_ids")
        for label, output, token_ids in (
            ("v1", output1, token_ids1),
            ("v2", output2, token_ids2),
        ):
            if token_ids is not None and not isinstance(token_ids, list):
                raise SystemExit(f"{label} token ids for prompt {output['prompt_id']} are not a list")
            if token_ids == [] and output.get("text"):
                raise SystemExit(f"{label} prompt {output['prompt_id']} has nonempty text but empty token ids")
        if (token_ids1 is None) != (token_ids2 is None):
            raise SystemExit(
                f"comparison level differs for prompt {output1['prompt_id']}: token ids exist on only one side"
            )
        top1, top2 = output1.get("top_logprobs", []), output2.get("top_logprobs", [])
        if len(top1) != len(top2):
            raise SystemExit(
                f"top-logprob position counts differ for prompt {output1['prompt_id']}: v1={len(top1)} v2={len(top2)}"
            )


def _compare_outputs(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_comparable(v1, v2)
    rows = []
    for out1, out2 in zip(v1["outputs"], v2["outputs"], strict=True):
        if out1["prompt_id"] != out2["prompt_id"]:
            raise SystemExit(
                f"prompt order differs between the two dumps "
                f"({out1['prompt_id']} vs {out2['prompt_id']}); rerun both with the same --prompts/--repeat"
            )
        tokens1, tokens2 = out1["token_ids"], out2["token_ids"]
        # Remote mode requests generated token IDs when the server supports them.
        # Fall back to character-level comparison when either dump lacks IDs. This
        # is coarser than token identity because different tokenizations can produce
        # the same text, so the report makes the comparison level explicit.
        token_level = tokens1 is not None and tokens2 is not None
        if token_level:
            divergence = _first_divergence(tokens1, tokens2)
            identical = divergence is None and len(tokens1) == len(tokens2)
            len1, len2 = len(tokens1), len(tokens2)
        else:
            text1, text2 = out1.get("text", ""), out2.get("text", "")
            divergence = _first_divergence(text1, text2)
            identical = divergence is None and len(text1) == len(text2)
            len1, len2 = len(text1), len(text2)

        overlaps, kls = [], []
        for step1, step2 in zip(out1["top_logprobs"], out2["top_logprobs"], strict=True):
            overlaps.append(_topk_overlap(step1, step2))
            kls.append(_truncated_kl(step1, step2))

        rows.append(
            {
                "prompt_id": out1["prompt_id"],
                "prompt_num_tokens": out1["prompt_num_tokens"],
                "comparison_level": "token" if token_level else "char",
                "len_v1": len1,
                "len_v2": len2,
                "identical": identical,
                "first_divergence": divergence,
                "mean_topk_overlap": _mean(overlaps),
                "mean_kl": _mean(kls),
                "compared_positions": len(overlaps),
            }
        )
    return rows


def _markdown(v1: dict[str, Any], v2: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, bool]:
    config = v1["config"]
    greedy = config["temperature"] == 0.0
    lines: list[str] = []

    lines.append("# DSpark Model Runner V2 vs V1 comparison\n")
    lines.append("## Configuration\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key in (
        "model",
        "draft_model",
        "num_speculative_tokens",
        "draft_sample_method",
        "enforce_eager_draft",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "logprobs",
        "tensor_parallel_size",
        "data_parallel_size",
        "cudagraph_mode",
        "batch_size",
    ):
        lines.append(f"| {key} | `{config.get(key)}` |")
    lines.append("")

    if v1.get("mode") == "remote":
        verification1 = v1.get("runner_verification", {"status": "legacy_unverified", "verified": False})
        verification2 = v2.get("runner_verification", {"status": "legacy_unverified", "verified": False})
        if not verification1.get("verified") or not verification2.get("verified"):
            lines.append(
                "> WARNING: Remote runner identity is not verified by server-start evidence in both dumps. "
                f"v1={verification1.get('status')}, v2={verification2.get('status')}. "
                "Treat runner labels as client claims unless startup logs are attached.\n"
            )

    degenerate_v1, degenerate_v2 = _degenerate_prompts(v1), _degenerate_prompts(v2)
    # Only a degenerate *baseline* (v1) invalidates the comparison. A degenerate candidate (v2) next to a
    # healthy baseline is a genuine regression signal.
    invalid = bool(degenerate_v1)
    candidate_degenerate = bool(degenerate_v2) and not degenerate_v1
    if invalid:
        lines.append(
            "> **Invalid comparison: degenerate output detected in the V1 baseline.** "
            f"Degenerate V1 prompts={degenerate_v1}; "
            f"degenerate V2 prompts={degenerate_v2 or 'none'}. "
            f"At least {DEGENERATE_CHAR_FRACTION:.0%} of the generated symbols "
            "(token IDs when available, otherwise non-whitespace characters) are "
            "identical. A degenerate baseline is not a valid reference, so the "
            "identity and acceptance differences below must not be classified as "
            "a V2 regression.\n"
        )
    elif candidate_degenerate:
        lines.append(
            "> **Candidate-side V2 degeneration detected next to a healthy V1 "
            "baseline; this is a valid V2 regression signal.** "
            f"Degenerate V2 prompts={degenerate_v2}; at least "
            f"{DEGENERATE_CHAR_FRACTION:.0%} of the generated symbols are identical.\n"
        )

    comparison_levels = {row["comparison_level"] for row in rows}
    char_level = comparison_levels == {"char"}
    lines.append(
        "## 1. Generated text identity (character level)\n" if char_level else "## 1. Generated token identity\n"
    )
    if greedy:
        if char_level:
            lines.append(
                "Under greedy decoding (`temperature=0`), generated text should "
                "match. Character-level comparison is a coarser fallback.\n"
            )
        else:
            lines.append("Under greedy decoding (`temperature=0`), generated token sequences should match exactly.\n")
    else:
        unit = "text" if char_level else "token sequences"
        lines.append(
            "Under non-greedy decoding (`temperature>0`), runners use independent "
            f"random draws, so identical {unit} is not required. Use the "
            "distribution metrics in sections 2 and 3 for the verdict.\n"
        )
    if comparison_levels == {"char"}:
        lines.append(
            "> This comparison is character-level because generated token IDs "
            "were unavailable. Output lengths are characters, not tokens.\n"
        )
    elif comparison_levels == {"token", "char"}:
        lines.append(
            "> The dump contains both token- and character-level rows. Output "
            "length units differ and are comparable only within each row.\n"
        )
    lines.append("| Prompt | Prompt tokens | V1 output length | V2 output length | Identical | First divergence |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        divergence = "-" if row["first_divergence"] is None else str(row["first_divergence"])
        lines.append(
            f"| {row['prompt_id']} | {row['prompt_num_tokens']} | {row['len_v1']} | {row['len_v2']} | "
            f"{'yes' if row['identical'] else 'no'} | {divergence} |"
        )
    identical_count = sum(1 for row in rows if row["identical"])
    lines.append(f"\nIdentical outputs: {identical_count}/{len(rows)}.\n")

    lines.append("## 2. Speculative acceptance\n")
    acc1, acc2 = v1["acceptance"], v2["acceptance"]
    lines.append("| Position | V1 acceptance | V2 acceptance | Delta |")
    lines.append("| --- | --- | --- | --- |")
    per_pos_ok = True
    for pos, (a, b) in enumerate(zip(acc1["acceptance_per_pos"], acc2["acceptance_per_pos"], strict=True)):
        delta = b - a
        if abs(delta) > ACCEPTANCE_TOLERANCE:
            per_pos_ok = False
        lines.append(f"| {pos} | {a:.4f} | {b:.4f} | {delta:+.4f} |")
    lines.append("")
    lines.append("| Metric | V1 | V2 |")
    lines.append("| --- | --- | --- |")
    for key in (
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "mean_acceptance_length",
    ):
        value1, value2 = acc1[key], acc2[key]
        formatted = (f"{value1:.4f}", f"{value2:.4f}") if isinstance(value1, float) else (str(value1), str(value2))
        lines.append(f"| {key} | {formatted[0]} | {formatted[1]} |")
    lines.append("")

    # The u=0.0 defect (#13730, fixed by #13470) manifested as acceptance
    # pinned at 1.0 for every position under temperature>0. Flag a recurrence.
    if not greedy:
        suspicious = [pos for pos, rate in enumerate(acc2["acceptance_per_pos"]) if rate >= 0.999]
        if len(suspicious) == len(acc2["acceptance_per_pos"]) and acc2["num_drafts"] > 0:
            lines.append(
                "> V2 acceptance is 1.0 at every position under non-greedy "
                "sampling. This matches the #13730 (`u=0.0`) signature fixed by "
                "#13470 and should be reported if reproduced on fixed code.\n"
            )

    lines.append("## 3. Logit distribution (top-k logprobs)\n")
    logprobs_available = config.get("logprobs", 0) > 0 and any(row["compared_positions"] > 0 for row in rows)
    if not logprobs_available:
        lines.append(
            "**UNAVAILABLE**: no valid top-k logprob positions were collected, so "
            "overlap and KL are not computed or replaced with placeholder NaNs.\n"
        )
    else:
        lines.append(
            f"Each position retains top-{config['logprobs']}. KL is computed after "
            "renormalizing the union of both top-k supports. It is a truncated-support "
            "approximation, not full-vocabulary KL.\n"
        )
        lines.append("| Prompt | Compared positions | Mean top-k overlap | Mean KL(V1||V2) |")
        lines.append("| --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['prompt_id']} | {row['compared_positions']} | "
                f"{row['mean_topk_overlap']:.4f} | {row['mean_kl']:.6f} |"
            )
        lines.append("")

    lines.append("## 4. Collection throughput (informational)\n")
    is_remote = v1.get("timing", {}).get("remote_mode") or v2.get("timing", {}).get("remote_mode")
    if is_remote:
        lines.append(
            "> Throughput uses client wall time from HTTP submission through the "
            "last response, including network and queueing. Do not compare its "
            "absolute value with local mode. Use `dspark_v1_v2_perf.sh` for TTFT, "
            "TPOT, and end-to-end latency.\n"
        )
    else:
        lines.append(
            "> Throughput uses the wall time of offline `LLM.generate` and includes "
            "batch scheduling effects. Use `dspark_v1_v2_perf.sh` for formal "
            "performance measurements.\n"
        )
    lines.append("| Metric | V1 | V2 | V2/V1 |")
    lines.append("| --- | --- | --- | --- |")
    t1, t2 = v1["timing"], v2["timing"]
    for key in (
        "generate_wall_seconds",
        "total_output_tokens",
        "output_tokens_per_second",
    ):
        a, b = t1[key], t2[key]
        ratio = f"{b / a:.3f}" if isinstance(a, (int, float)) and a else "-"
        if isinstance(a, float):
            lines.append(f"| {key} | {a:.3f} | {b:.3f} | {ratio} |")
        else:
            lines.append(f"| {key} | {a} | {b} | {ratio} |")
    lines.append("")

    lines.append("## 5. Verdict\n")
    verdict_ok = True
    if greedy and GREEDY_IDENTITY_REQUIRED:
        comparison_name = "character-level text" if char_level else "token sequences"
        if identical_count == len(rows):
            lines.append(f"- Greedy identity: all {len(rows)} {comparison_name} match.")
        else:
            verdict_ok = False
            failed = [row["prompt_id"] for row in rows if not row["identical"]]
            lines.append(
                f"- Greedy identity: {len(rows) - identical_count} {comparison_name} "
                f"differ ({failed}). "
                + (
                    "The comparison is invalid because the baseline is degenerate; "
                    "this is not classified as a regression."
                    if invalid
                    else "This is an accuracy regression signal."
                )
            )
    else:
        unit = "character-level text" if char_level else "token sequences"
        lines.append(f"- Greedy identity: not applicable to this non-greedy {unit} comparison.")

    if per_pos_ok:
        lines.append(f"- Acceptance: every per-position delta is within +/-{ACCEPTANCE_TOLERANCE}.")
    else:
        verdict_ok = False
        lines.append(
            f"- Acceptance: at least one per-position delta exceeds "
            f"+/-{ACCEPTANCE_TOLERANCE}. "
            + (
                "The comparison is invalid because the baseline is degenerate."
                if invalid
                else "Investigate before classifying the difference."
            )
        )

    if invalid:
        # Even identical degenerate outputs do not establish equivalence.
        verdict_ok = False
        lines.append(
            "- Overall: invalid comparison because the V1 baseline contains "
            "degenerate output; individual pass/fail markers are inconclusive."
        )
    elif candidate_degenerate:
        verdict_ok = False
        lines.append("- Overall: degenerate V2 output is a regression signal.")

    lines.append("")
    return "\n".join(lines), verdict_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--v1",
        required=True,
        help="JSON dump collected with VLLM_USE_V2_MODEL_RUNNER=0.",
    )
    parser.add_argument(
        "--v2",
        required=True,
        help="JSON dump collected with VLLM_USE_V2_MODEL_RUNNER=1.",
    )
    parser.add_argument("--out", default=None, help="Write the Markdown report here (default: stdout).")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when the verdict is a regression (for CI use).",
    )
    args = parser.parse_args()

    v1, v2 = _load(args.v1), _load(args.v2)
    if v1["runner"] != "v1" or v2["runner"] != "v2":
        raise SystemExit(f"expected a v1 and a v2 dump, got runner={v1['runner']!r} and runner={v2['runner']!r}")

    rows = _compare_outputs(v1, v2)
    report, verdict_ok = _markdown(v1, v2, rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"[compare] wrote {args.out} (verdict_ok={verdict_ok})")
    else:
        print(report)

    if args.fail_on_regression and not verdict_ok:
        invalid = bool(_degenerate_prompts(v1))
        raise SystemExit(EXIT_INVALID_COMPARISON if invalid else 1)


if __name__ == "__main__":
    main()
