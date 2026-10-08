"""Opt-in, same-service AISBench TTFT experiment for the GLM-5.1 CI case.

Keep both rounds even when throughput is below the regular performance gate.
Success here requires exact token counts and all requests succeeding; this is
a diagnostic experiment, not a replacement performance baseline.
"""

import ast
import gzip
import hashlib
import io
import json
import math
import shutil
import sqlite3
import statistics
import struct
import tempfile
import urllib.request
import uuid
from contextlib import closing
from pathlib import Path

DATASET_SHA256 = "a05b561721da92e35b921a2ade43d290c6070201848fc681ecbbe4bcbe35e203"
INPUT_TOKENS = 3500
OUTPUT_TOKENS = 1500
CONCURRENCY = 8
NUM_PROMPTS = 32


def summarize(values):
    values = sorted(values)

    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "min": values[0],
        "max": values[-1],
    }


def decode_time_points(blob):
    """Read AISBench's one-dimensional float64 NPY storage without pickle."""
    stream = io.BytesIO(blob)
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid NPY data")
    version = stream.read(2)
    size_format = "<H" if version[0] == 1 else "<I"
    size = struct.unpack(size_format, stream.read(struct.calcsize(size_format)))[0]
    header = ast.literal_eval(stream.read(size).decode())
    if header["descr"] != "<f8" or len(header["shape"]) != 1:
        raise ValueError(f"Unexpected timestamp array: {header}")
    return list(struct.unpack("<" + "d" * header["shape"][0], stream.read()))


def analyze_round(detail_path, questions):
    records = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines()]
    if sorted(row["id"] for row in records) != list(range(len(questions))):
        raise ValueError("Missing or duplicate request IDs")
    result = []
    all_itls = []
    for row in sorted(records, key=lambda row: row["id"]):
        index = row["id"]
        if not row["success"] or row["input_tokens"] != INPUT_TOKENS or row["output_tokens"] != OUTPUT_TOKENS:
            raise ValueError(f"Request {index} failed or has unexpected token counts")
        messages = ast.literal_eval(row["input"]) if isinstance(row["input"], str) else row["input"]
        if messages != [{"role": "user", "content": questions[index]}]:
            raise ValueError(f"Request {index} does not match the fixed dataset")
        points = row["time_points"]
        if isinstance(points, dict):
            database = detail_path.parent / "db_data" / Path(row["db_name"]).name
            with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
                blob = connection.execute(
                    "SELECT arr_blob FROM numpy_store WHERE id = ?", (points["__db_ref__"],)
                ).fetchone()[0]
            points = decode_time_points(blob)
        if len(points) < 3 or any(b < a for a, b in zip(points, points[1:])):
            raise ValueError(f"Invalid stream timestamps for request {index}")
        itls = [(b - a) * 1000 for a, b in zip(points[1:], points[2:])]
        all_itls.extend(itls)
        result.append(
            {
                "id": index,
                "prompt_sha256": hashlib.sha256(questions[index].encode()).hexdigest(),
                "request_uuid": row.get("uuid"),
                "start_time": points[0],
                "ttft_ms": (points[1] - points[0]) * 1000,
                "tpot_ms": (points[-1] - points[1]) * 1000 / (OUTPUT_TOKENS - 1),
                "mean_itl_ms": statistics.mean(itls),
                "sse_events": len(points) - 1,
            }
        )
    return {
        "requests": result,
        "ttft_ms": summarize([row["ttft_ms"] for row in result]),
        "tpot_ms": summarize([row["tpot_ms"] for row in result]),
        "itl_ms": summarize(all_itls),
    }


def post_json(base_url, path, body):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def prepare_dataset(model, base_url, output_dir):
    fixture = Path(__file__).resolve().parents[1] / "tests/e2e/cases/models/data/glm51-total3500-num32.jsonl.gz"
    data = gzip.decompress(fixture.read_bytes())
    if hashlib.sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError("Fixed dataset checksum mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != NUM_PROMPTS:
        raise ValueError("Expected exactly 32 prompts")
    counts = []
    for row in rows:
        tokenized = post_json(
            base_url,
            "/tokenize",
            {"model": model, "messages": [{"role": "user", "content": row["question"]}], "add_generation_prompt": True},
        )
        counts.append(tokenized["count"])
    if counts != [INPUT_TOKENS] * NUM_PROMPTS:
        raise ValueError(f"Expected exactly 3500 tokens after chat templating: {counts}")
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir()
    # GSM8KDataset loads both splits. Only the test split is benchmarked.
    for split in ("train", "test"):
        (dataset_dir / f"{split}.jsonl").write_bytes(data)
    return dataset_dir, [row["question"] for row in rows]


def service_identity(server_process):
    import psutil

    if server_process.poll() is not None:
        raise RuntimeError("The original service process exited")
    parent = psutil.Process(server_process.pid)
    return sorted((process.pid, process.create_time()) for process in [parent, *parent.children(recursive=True)])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def execute_phase(model, host, port, case, questions, phase, work_dir, output_dir):
    from tools.aisbench import AisbenchRunner

    phase_dir = output_dir / phase
    phase_dir.mkdir()
    options = {**case, "num_warmups": 0, "num_prompts": len(questions), "work_dir": str(work_dir / phase)}
    try:
        with AisbenchRunner(model=model, port=port, host_ip=host, aisbench_config=options, verify=False) as runner:
            runner._get_result_performance()
            details = list(Path(runner.exp_folder).rglob("gsm8k_details.jsonl"))
            if len(details) != 1:
                raise ValueError(f"Expected one details file, found {details}")
            analysis = analyze_round(details[0].resolve(), questions)
            analysis["aisbench_summary"] = runner.result_json
            save_json(phase_dir / "analysis.json", analysis)
            print(f"TTFT_DIAGNOSTIC_PHASE {phase} {json.dumps(analysis)}", flush=True)
            return analysis
    finally:
        # Export raw request records, SQLite timestamps, configs and client logs
        # even if validation fails. CI collects this PVC directory on failure too.
        if (work_dir / phase).exists():
            shutil.copytree(work_dir / phase, phase_dir / "aisbench", dirs_exist_ok=True)
        if Path("output_performance.txt").exists():
            shutil.copy2("output_performance.txt", phase_dir / "client.log")


def run_ttft_diagnostic(*, model, host, port, case, server_process, output_dir):
    if (case["batch_size"], case["num_prompts"], case["max_out_len"], case["request_rate"]) != (
        CONCURRENCY,
        NUM_PROMPTS,
        OUTPUT_TOKENS,
        0,
    ):
        raise ValueError("This diagnostic requires C8, 32 requests, 1500 output tokens and request_rate=0")
    output_dir = Path(output_dir) / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)
    base_url = f"http://{host}:{port}"
    identity = service_identity(server_process)
    manifest = {"dataset_sha256": DATASET_SHA256, "service_processes": identity, "case": case, "phases": []}
    save_json(output_dir / "manifest.json", manifest)
    with tempfile.TemporaryDirectory(prefix="ttft-diagnostic-") as temporary:
        work_dir = Path(temporary)
        dataset_dir, questions = prepare_dataset(model, base_url, work_dir)
        shutil.copy2(dataset_dir / "test.jsonl", output_dir / "exact3500.jsonl")
        options = {**case, "dataset_path_local": str(dataset_dir)}
        rounds = []
        for phase, prompts in (("warmup", questions[:CONCURRENCY]), ("round1", questions), ("round2", questions)):
            if not set(identity).issubset(service_identity(server_process)):
                raise RuntimeError("Service process identity changed between phases")
            with urllib.request.urlopen(base_url + "/metrics", timeout=30) as response:
                (output_dir / f"{phase}-metrics-before.txt").write_bytes(response.read())
            analysis = execute_phase(model, host, port, options, prompts, phase, work_dir, output_dir)
            if not set(identity).issubset(service_identity(server_process)):
                raise RuntimeError("Service process identity changed during the experiment")
            manifest["phases"].append(phase)
            save_json(output_dir / "manifest.json", manifest)
            if phase != "warmup":
                rounds.append(analysis)
        with urllib.request.urlopen(base_url + "/metrics", timeout=30) as response:
            (output_dir / "round2-metrics-after.txt").write_bytes(response.read())
    paired = [
        {
            "id": first["id"],
            "prompt_sha256": first["prompt_sha256"],
            "round1_ttft_ms": first["ttft_ms"],
            "round2_ttft_ms": second["ttft_ms"],
            "delta_ms": second["ttft_ms"] - first["ttft_ms"],
        }
        for first, second in zip(rounds[0]["requests"], rounds[1]["requests"])
    ]
    comparison = {"round1": rounds[0], "round2": rounds[1], "paired_requests": paired}
    save_json(output_dir / "comparison.json", comparison)
    print(f"TTFT_DIAGNOSTIC_COMPARISON {json.dumps(comparison)}", flush=True)
