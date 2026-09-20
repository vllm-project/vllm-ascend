# SPDX-License-Identifier: Apache-2.0
"""Inspect a local host trace. This command does not execute/replay a model."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from vllm_ascend.dfx.recorder import SCHEMA


def decode_arrays(value):
    if isinstance(value, list):
        return [decode_arrays(item) for item in value]
    if isinstance(value, dict):
        if value.get("encoding") == "numpy-base64":
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biuf":
                raise ValueError("Only numeric host arrays are supported")
            raw = base64.b64decode(value["data"], validate=True)
            return np.frombuffer(raw, dtype=dtype).reshape(value["shape"]).tolist()
        return {key: decode_arrays(item) for key, item in value.items()}
    return value


def read_trace(path: Path, execution_id: int | None = None, request_id: str | None = None) -> dict:
    with path.open(encoding="utf-8") as source:
        trace = json.load(source)
    if trace.get("schema") != SCHEMA:
        raise ValueError("Unsupported DFX trace schema")
    records = trace["records"]
    if request_id is not None:
        selected = {
            record["execution_id"]
            for record in records
            if record["kind"] == "host_batch_prepared"
            and request_id in (record.get("payload") or {}).get("req_ids", [])
        }
        records = [record for record in records if record["execution_id"] in selected]
    if execution_id is not None:
        records = [record for record in records if record["execution_id"] == execution_id]
    # Preserve complete co-batches and scheduler records, not just the selected row.
    trace["records"] = decode_arrays(records)
    return trace


def decode_tensor(tensor):
    """Decode owned bytes without pickle or executing model code."""
    raw = np.asarray(tensor["data"], dtype=np.uint8).tobytes()
    layout = tensor["layout"]
    name = layout["dtype"].removeprefix("torch.")
    if name == "bfloat16":
        values = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        allowed = {"float16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"}
        if name not in allowed:
            raise ValueError(f"Unsupported comparison dtype: {name}; quantization scales/layout are required")
        values = np.frombuffer(raw, dtype=np.dtype(name))
    return values.reshape(layout["shape"])


def compare_device_snapshots(actual, reference, *, atol=1e-3, rtol=1e-3):
    """Compare only aligned, valid observations. A difference does not prove
    which implementation is correct; the reference must be independently vetted.
    """
    if not np.isfinite(atol) or not np.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    alignment = ("layer", "group", "kind", "req_ids", "blocks", "block_size", "input_digest", "request_fingerprints")
    if not actual.get("input_digest_complete") or not reference.get("input_digest_complete"):
        return {"status": "inconclusive", "reason": "input_evidence_missing"}
    if not actual.get("request_context_complete") or not reference.get("request_context_complete"):
        return {"status": "inconclusive", "reason": "request_history_unverified"}
    if any(
        (tuple(actual.get(key, ())) != tuple(reference.get(key, ())))
        if key in ("req_ids", "blocks", "request_fingerprints")
        else actual.get(key) != reference.get(key)
        for key in alignment
    ):
        return {"status": "inconclusive", "reason": "batch_or_page_alignment_mismatch"}
    comparisons, skipped = {}, {}
    actual_tensors = actual["tensors"]
    reference_tensors = reference["tensors"]
    names = sorted(
        {
            name
            for name in actual_tensors.keys() | reference_tensors.keys()
            if name == "raw_logits" or name.startswith("kv.after.")
        }
    )
    for name in names:
        if name not in actual_tensors or name not in reference_tensors:
            skipped[name] = "missing_tensor"
            continue
        left, right = actual_tensors[name], reference_tensors[name]
        if name == "raw_logits" and tuple(actual.get("scalars", {}).get("logits_selection", ())) != tuple(
            reference.get("scalars", {}).get("logits_selection", ())
        ):
            skipped[name] = "logits_selection_mismatch"
            continue
        try:
            a, b = decode_tensor(left), decode_tensor(right)
        except ValueError as error:
            skipped[name] = str(error)
            continue
        if a.shape != b.shape:
            skipped[name] = "shape_mismatch"
            continue
        if name.startswith("kv.") and not (left.get("whole_state_valid") and right.get("whole_state_valid")):
            offsets = left.get("checked_token_offsets")
            if (
                offsets is None
                or offsets != right.get("checked_token_offsets")
                or left.get("token_axis") != right.get("token_axis")
            ):
                skipped[name] = "valid_token_region_unverified"
                continue
            a = np.take(a, offsets, axis=left["token_axis"])
            b = np.take(b, offsets, axis=right["token_axis"])
        if not a.size:
            skipped[name] = "empty_valid_region"
            continue
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            comparisons[name] = {"match": False, "reason": "nonfinite_valid_values"}
            continue
        error = np.abs(a - b)
        mismatch = error > atol + rtol * np.abs(b)
        comparisons[name] = {
            "match": not bool(mismatch.any()),
            "elements": int(a.size),
            "mismatches": int(mismatch.sum()),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    mismatch = any(not result["match"] for result in comparisons.values())
    return {
        "status": "mismatch" if mismatch else "inconclusive" if skipped or not comparisons else "observed_match",
        "comparisons": comparisons,
        "skipped": skipped,
        "atol": atol,
        "rtol": rtol,
        "scope": "aligned_observed_values_not_full_model_accuracy",
    }


def compare_traces(actual, reference, *, atol=1e-3, rtol=1e-3):
    if actual.get("manifest") is None or reference.get("manifest") is None:
        return {"status": "inconclusive", "reason": "runtime_manifest_missing"}
    for key in ("model", "revision", "parallel"):
        if actual["manifest"].get(key) != reference["manifest"].get(key):
            return {"status": "inconclusive", "reason": "model_or_topology_mismatch"}
    for key in ("rank", "dp_rank"):
        if actual["identity"].get(key) != reference["identity"].get(key):
            return {"status": "inconclusive", "reason": "rank_mismatch"}
    reference_records = {}
    for record in reference["records"]:
        if record["kind"] == "device_snapshot" and record.get("payload") is not None:
            payload = record["payload"]
            key = (record["execution_id"], payload["layer"], payload["kind"])
            if key in reference_records:
                return {"status": "inconclusive", "reason": "ambiguous_reference_step"}
            reference_records[key] = payload
    results = []
    for record in actual["records"]:
        if record["kind"] != "device_snapshot" or record.get("payload") is None:
            continue
        payload = record["payload"]
        key = (record["execution_id"], payload["layer"], payload["kind"])
        result = (
            compare_device_snapshots(payload, reference_records[key], atol=atol, rtol=rtol)
            if key in reference_records
            else {
                "status": "inconclusive",
                "reason": "reference_step_missing",
            }
        )
        results.append({"execution_id": key[0], "layer": key[1], **result})
    return {
        "status": "mismatch"
        if any(item["status"] == "mismatch" for item in results)
        else "observed_match"
        if results and all(item["status"] == "observed_match" for item in results)
        else "inconclusive",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--execution-id", type=int)
    parser.add_argument("--request-id")
    parser.add_argument(
        "--reference", type=Path, help="Compare aligned snapshots against an independently vetted trace"
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    trace = read_trace(args.trace, args.execution_id, args.request_id)
    if args.reference:
        trace = compare_traces(trace, read_trace(args.reference), atol=args.atol, rtol=args.rtol)
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
