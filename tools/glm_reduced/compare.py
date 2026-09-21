# SPDX-License-Identifier: Apache-2.0
"""Numerical comparison of fixed-workload logprob dumps.

Pure comparison logic for the precision gate. ``run_logits_dump.py`` (vLLM/NPU
boundary) produces JSONL dumps; this module compares a candidate dump against
a baseline dump. A comparison is only meaningful between two runs of the
**same checkpoint** (identified by ``checkpoint_id``, the reduction manifest
digest or an explicit content id) across a differing axis such as run mode,
runtime revision or dtype. It fails closed: anything that cannot be proven
is a failure, never a silent pass.

Hard rules:
- both dumps carry identity metadata: checkpoint id, a per-run ``run_id``,
  runtime revisions and the actual engine settings;
- checkpoint ids must match; seeds must match; the workload shape must match;
- the two runs must be distinct runs (different ``run_id``); independent A/A
  runs with otherwise identical metadata are valid noise characterization and
  are not rejected;
- each dump must be complete: prompt_count records with unique prompt indices
  and exactly output_tokens produced tokens/logprob positions per record —
  a dump truncated on both sides cannot pass;
- identical prompt sets in identical order, with identical prompt token ids;
- greedy token sequences are compared and the first divergence is reported;
- the sampled token must be present in both dumps' top-k for every position,
  and at least one logprob pair must actually be compared;
- per-position top-k logprob differences must fit the caller-supplied
  tolerances (finite, non-negative; there is no default tolerance);
- every value must be finite (NaN and infinities alike).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

DUMP_FORMAT = "glm-reduced-logprobs-v2"
REQUIRED_META_KEYS = (
    "model",
    "checkpoint_id",
    "run_id",
    "dtype",
    "mode",
    "seed",
    "prompt_count",
    "output_tokens",
    "runtime",
    "engine",
)


@dataclass
class Dump:
    meta: dict
    records: list[dict]  # {"prompt_index", "prompt_token_ids", "token_ids", "top_logprobs"}


@dataclass
class ComparisonReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    compared_positions: int = 0
    diverged_prompts: list[int] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "problems": self.problems,
                "max_abs_diff": self.max_abs_diff,
                "max_rel_diff": self.max_rel_diff,
                "compared_positions": self.compared_positions,
                "diverged_prompts": self.diverged_prompts,
            },
            indent=2,
        )


def check_identity(
    baseline_meta: dict,
    candidate_meta: dict,
    workload_keys: tuple[str, ...],
    *,
    require_identical_engine: bool = False,
) -> list[str]:
    """Shared fail-closed identity/comparability checks for both gates.

    Identity comes from ``checkpoint_id`` (content digest), not from model
    paths. Two runs of the same checkpoint are comparable whenever they are
    *different runs*: ``run_id`` must differ (the CLI additionally rejects the
    exact same input file). Independent A/A runs with otherwise identical
    metadata are valid noise characterization and must not be rejected. With
    ``require_identical_engine`` (performance gate), dtype and the full engine
    settings must also match — only runtime revisions / run labels may differ,
    so TP/EP/caching-policy changes cannot masquerade as regressions.
    """
    problems = []
    for key in workload_keys:
        if baseline_meta.get(key) != candidate_meta.get(key):
            problems.append(
                f"workload mismatch: baseline {key}={baseline_meta.get(key)!r} vs "
                f"candidate {key}={candidate_meta.get(key)!r}"
            )
    base_id = baseline_meta.get("checkpoint_id")
    cand_id = candidate_meta.get("checkpoint_id")
    if not base_id or not cand_id:
        problems.append(
            "both runs must record a checkpoint_id (reduction manifest digest or explicit content id); "
            "a bare model path is not an identity"
        )
    elif base_id != cand_id:
        problems.append(
            "checkpoint_id mismatch: baseline and candidate were produced from different checkpoints; "
            "comparing arbitrary distinct checkpoints is not a valid gate"
        )
    if baseline_meta.get("seed") != candidate_meta.get("seed"):
        problems.append("seed mismatch between baseline and candidate")
    base_run = baseline_meta.get("run_id")
    cand_run = candidate_meta.get("run_id")
    if not base_run or not cand_run:
        problems.append("both runs must record a distinct run_id")
    elif base_run == cand_run:
        problems.append(
            "baseline and candidate carry the same run_id: they are the same run, not two independent measurements"
        )
    if require_identical_engine:
        if baseline_meta.get("dtype") != candidate_meta.get("dtype"):
            problems.append(
                f"dtype mismatch ({baseline_meta.get('dtype')!r} vs {candidate_meta.get('dtype')!r}); "
                "performance runs are only comparable at identical dtype"
            )
        if baseline_meta.get("engine") != candidate_meta.get("engine"):
            problems.append(
                f"engine settings mismatch ({baseline_meta.get('engine')!r} vs "
                f"{candidate_meta.get('engine')!r}); performance runs must use identical engine "
                "settings (TP/EP/caching policy/...)"
            )
    return problems


def validate_dump(dump: Dump, where: str = "dump") -> list[str]:
    """Structural completeness: declared workload vs actual records."""
    problems = []
    meta = dump.meta
    prompt_count = meta.get("prompt_count")
    output_tokens = meta.get("output_tokens")
    if not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count <= 0:
        return problems + [f"{where}: prompt_count={prompt_count!r} is not a positive integer"]
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens <= 0:
        return problems + [f"{where}: output_tokens={output_tokens!r} is not a positive integer"]
    if len(dump.records) != prompt_count:
        problems.append(
            f"{where}: {len(dump.records)} records for declared prompt_count={prompt_count}; "
            "a truncated dump cannot pass"
        )
        return problems
    indices = [record.get("prompt_index") for record in dump.records]
    if sorted(indices) != list(range(prompt_count)):
        problems.append(f"{where}: prompt_index values {indices} are not a unique 0..{prompt_count - 1} set")
    for record in dump.records:
        token_ids = record.get("token_ids")
        top_logprobs = record.get("top_logprobs")
        produced = len(token_ids) if isinstance(token_ids, list) else token_ids
        if not isinstance(token_ids, list) or len(token_ids) != output_tokens:
            problems.append(
                f"{where} prompt {record.get('prompt_index')}: produced {produced!r} tokens, declared "
                f"output_tokens={output_tokens}; truncated output cannot pass"
            )
        elif not isinstance(top_logprobs, list) or len(top_logprobs) != output_tokens:
            problems.append(
                f"{where} prompt {record.get('prompt_index')}: top_logprobs length does not match "
                f"declared output_tokens={output_tokens}"
            )
    return problems


def load_dump(path: str | Path) -> Dump:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"logprob dump {str(path)!r} does not exist")
    meta: dict | None = None
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if line_no == 1:
                if payload.get("format") != DUMP_FORMAT:
                    raise ValueError(f"{path.name}: first line must be a metadata header with format={DUMP_FORMAT!r}")
                meta = payload
                continue
            records.append(payload)
    if meta is None:
        raise ValueError(f"{path.name}: empty dump")
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{path.name}: metadata is missing keys {missing}")
    dump = Dump(meta=meta, records=records)
    problems = validate_dump(dump, where=path.name)
    if problems:
        raise ValueError("; ".join(problems))
    return dump


def _validate_record(record: dict, where: str) -> list[str]:
    problems = []
    for key in ("prompt_index", "prompt_token_ids", "token_ids", "top_logprobs"):
        if key not in record:
            problems.append(f"{where}: record missing {key!r}")
            return problems
    if len(record["token_ids"]) != len(record["top_logprobs"]):
        problems.append(f"{where}: token_ids/top_logprobs length mismatch")
        return problems
    for position, entries in enumerate(record["top_logprobs"]):
        if not isinstance(entries, list) or not entries:
            problems.append(f"{where}: position {position} has no top-k logprob entries")
            continue
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or "token_id" not in entry:
                problems.append(f"{where}: position {position} has a malformed top-k entry")
                break
            if entry["token_id"] in seen:
                problems.append(f"{where}: duplicate token id {entry['token_id']} at position {position}")
                break
            seen.add(entry["token_id"])
            value = entry.get("logprob")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{where}: non-finite logprob at position {position}: {value!r}")
                break
    return problems


def compare_dumps(baseline: Dump, candidate: Dump, *, atol: float, rtol: float) -> ComparisonReport:
    report = ComparisonReport(ok=True)
    if (
        not (isinstance(atol, (int, float)) and isinstance(rtol, (int, float)))
        or not (math.isfinite(atol) and math.isfinite(rtol))
        or atol < 0
        or rtol < 0
    ):
        report.problems.append(f"tolerances must be finite and non-negative, got atol={atol!r} rtol={rtol!r}")
        report.ok = False
        return report
    report.problems.extend(check_identity(baseline.meta, candidate.meta, ("prompt_count", "output_tokens")))
    report.problems.extend(validate_dump(baseline, "baseline"))
    report.problems.extend(validate_dump(candidate, "candidate"))
    if report.problems:
        report.ok = False
        return report

    for base_record, cand_record in zip(baseline.records, candidate.records):
        where = f"prompt {base_record.get('prompt_index')}"
        record_problems = _validate_record(base_record, f"baseline {where}")
        record_problems += _validate_record(cand_record, f"candidate {where}")
        report.problems.extend(record_problems)
        if record_problems:
            continue
        if base_record["prompt_token_ids"] != cand_record["prompt_token_ids"]:
            report.problems.append(f"{where}: prompt token ids differ between baseline and candidate")
            continue
        if base_record["token_ids"] != cand_record["token_ids"]:
            divergence = next(
                (i for i, (b, c) in enumerate(zip(base_record["token_ids"], cand_record["token_ids"])) if b != c),
                min(len(base_record["token_ids"]), len(cand_record["token_ids"])),
            )
            report.diverged_prompts.append(base_record["prompt_index"])
            report.problems.append(f"{where}: greedy token sequences diverge at position {divergence}")
            continue
        for position, (base_entries, cand_entries) in enumerate(
            zip(base_record["top_logprobs"], cand_record["top_logprobs"])
        ):
            sampled = base_record["token_ids"][position]
            base_by_token = {entry["token_id"]: entry["logprob"] for entry in base_entries}
            cand_by_token = {entry["token_id"]: entry["logprob"] for entry in cand_entries}
            if sampled not in base_by_token or sampled not in cand_by_token:
                report.problems.append(
                    f"{where} position {position}: sampled token {sampled} missing from the top-k of "
                    "one side; the dump does not cover the tokens that were actually generated"
                )
                continue
            for token, base_value in base_by_token.items():
                if token not in cand_by_token:
                    if token == sampled:
                        continue  # unreachable: sampled presence checked above
                    report.problems.append(
                        f"{where} position {position}: token {token} absent from candidate top-k; "
                        "top-k sets must overlap to prove numerical parity"
                    )
                    continue
                diff = abs(base_value - cand_by_token[token])
                rel = diff / max(abs(base_value), 1e-12)
                report.compared_positions += 1
                report.max_abs_diff = max(report.max_abs_diff, diff)
                report.max_rel_diff = max(report.max_rel_diff, rel)
                if diff > atol + rtol * abs(base_value):
                    report.problems.append(
                        f"{where} position {position} token {token}: logprob diff {diff:.6g} exceeds "
                        f"atol={atol} + rtol={rtol}*|{base_value:.6g}|"
                    )
    if report.compared_positions == 0 and not report.problems:
        report.problems.append("no logprob pairs were compared at all; an empty overlap cannot prove numerical parity")
    report.ok = not report.problems
    return report
