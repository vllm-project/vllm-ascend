# SPDX-License-Identifier: Apache-2.0
"""GLM reduced-model nightly integration and explicit baseline calibration."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

import regex as re

from .prepare import prepare_checkpoint, read_json
from .profiles import get_profile
from .reducer import required_shards
from .serving_gate import (
    CALIBRATION_REPEATS,
    CALIBRATION_STARTS,
    FORMAT,
    NIGHTLY_REPEATS,
    aggregate_perf,
    compare_output,
    compare_perf,
    digest,
    normalized_completion,
    validate_baseline,
    validate_perf,
)

DATA = Path(__file__).parent / "data" / "glm52"
SERVED_MODEL = "glm52-reduced8"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compile_requests(suite: dict, tokenizer) -> list[dict]:
    requests = []
    for prompt in suite["prompts"]:
        if "text" in prompt:
            tokens = tokenizer.encode(prompt["text"], add_special_tokens=False)
        else:
            prefix = tokenizer.encode(prompt["prefix"], add_special_tokens=False)
            suffix = tokenizer.encode(prompt["suffix"], add_special_tokens=False)
            filler = tokenizer.encode(prompt["filler"], add_special_tokens=False)
            remaining = prompt["input_tokens"] - len(prefix) - len(suffix)
            if remaining <= 0 or not filler:
                raise ValueError("Invalid retrieval prompt budget")
            tokens = prefix + (filler * (remaining // len(filler) + 1))[:remaining] + suffix
        requests.append({"id": prompt["id"], "request": {"model": SERVED_MODEL, "prompt": tokens, **suite["sampling"]}})
    return requests


def canonical_engine(config) -> dict:
    args = list(config.server_cmd)
    port = args.index("--port")
    args[port + 1] = "<PORT>"
    return {
        "args": args,
        "env": {k: str(v) for k, v in config.envs.items() if not k.endswith("PORT")},
    }


def runtime_provenance() -> dict:
    import vllm

    import vllm_ascend

    result = {}
    for name, module in (("vllm", vllm), ("vllm-ascend", vllm_ascend)):
        root = Path(module.__file__).resolve().parent.parent
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        changes = subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True)
        if commit.returncode or changes.returncode:
            raise ValueError(f"Cannot identify calibration checkout for {name}: {root}")
        result[name] = {
            "version": importlib.metadata.version(name),
            "source": str(root),
            "commit": commit.stdout.strip(),
            "tracked_diff_sha256": digest(changes.stdout.hex()),
        }
    return result


def runtime_identity(config, model_id: str, requests: list[dict], suite: dict, hardware: str) -> dict:
    import torch_npu

    machine = platform.machine()
    if "aarch64" in hardware and machine != "aarch64":
        raise ValueError(f"Hardware mismatch: runner {hardware}, actual CPU architecture {machine}")
    if "x86_64" in hardware and machine != "x86_64":
        raise ValueError(f"Hardware mismatch: runner {hardware}, actual CPU architecture {machine}")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "transformers")}
    toolkit_info = list(Path("/usr/local/Ascend/ascend-toolkit/latest").glob("*/ascend_toolkit_install.info"))
    versions = set()
    for path in toolkit_info:
        match = re.search(r"^version=(.+)$", path.read_text(), re.MULTILINE)
        if match:
            versions.add(match[1].strip().strip('"'))
    if not versions:
        raise ValueError("Cannot identify CANN toolkit; refusing an unqualified baseline")
    driver_path = Path("/usr/local/Ascend/driver/version.info")
    driver = driver_path.read_text().splitlines()
    driver_versions = sorted(line.strip() for line in driver if "=" in line)
    if not any(line.startswith("Version=") for line in driver_versions):
        raise ValueError("Cannot identify Ascend driver")
    devices = []
    for index in range(torch_npu.npu.device_count()):
        properties = torch_npu.npu.get_device_properties(index)
        devices.append({"name": properties.name, "total_memory": properties.total_memory})
    if len(devices) != 8:
        raise ValueError(f"The TP8 gate requires exactly eight visible NPUs, found {len(devices)}")
    return {
        "checkpoint_id": model_id,
        "source_descriptor": read_json(DATA / "source.json"),
        "hardware": {"runner": hardware, "machine": machine, "devices": devices},
        "dependencies": {**packages, "cann": sorted(versions), "driver": driver_versions},
        "engine": canonical_engine(config),
        "requests_sha256": digest(requests),
        "suite_sha256": digest(suite),
    }


def prepare_case(config, *, calibrating: bool = False, report_dir: Path | None = None, hardware: str | None = None):
    """Resolve the fixed source and reduced model before RemoteOpenAIServer starts."""
    from filelock import FileLock
    from transformers import AutoTokenizer

    from tools.aisbench import maybe_download_from_modelscope

    options = config.extra_config["reduced_model_gate"]
    descriptor_path = DATA / "source.json"
    descriptor = read_json(descriptor_path)
    if config.model != descriptor["model"]:
        raise ValueError("Configured model differs from pinned source")
    source_dir = options.get("source_dir")
    if source_dir is None:
        source_dir = maybe_download_from_modelscope(
            config.model,
            revision=descriptor["revision"],
            allow_patterns=["config.json", "quant_model_weights.safetensors.index.json"],
        )
        shards = required_shards(
            str(Path(source_dir) / "config.json"),
            str(Path(source_dir) / "quant_model_weights.safetensors.index.json"),
            get_profile(options["profile"]),
            options["layers"],
        )["shards"]
        auxiliary = [name for name in descriptor["files"] if not name.startswith("quant_model_weights-")]
        source_dir = maybe_download_from_modelscope(
            config.model, revision=descriptor["revision"], allow_patterns=sorted(set(shards + auxiliary))
        )
    cache_dir = Path(options.get("cache_dir", "~/.cache/vllm-ascend/glm-reduced")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_dir / ".prepare.lock"), timeout=1800):
        model, checkpoint_id = prepare_checkpoint(
            descriptor_path, source_dir, cache_dir, profile_name=options["profile"], layers=options["layers"]
        )
    suite = read_json(DATA / "suite.json")
    generated = compile_requests(suite, AutoTokenizer.from_pretrained(str(model), trust_remote_code=True))
    request_path = Path(options.get("requests", DATA / "requests.json"))
    if calibrating:
        requests = generated
    else:
        requests = read_json(request_path)["requests"]
        if generated != requests:
            raise ValueError("Fixed prompt tokens differ from tokenizer/suite; explicit recalibration required")
    identity = runtime_identity(config, checkpoint_id, requests, suite, hardware or options["hardware"])
    report_dir = report_dir or Path("benchmark_results") / config.name / uuid.uuid4().hex
    report_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "identity": identity,
        "requests": requests,
        "suite": suite,
        "model": str(model),
        "report_dir": report_dir,
        "runtime": runtime_provenance(),
    }
    write_json(report_dir / "identity.json", {"identity": identity, "runtime": state["runtime"]})
    write_json(
        report_dir / "configuration.json",
        {
            "name": config.name,
            "source_model": config.model,
            "server_cmd": config.server_cmd,
            "envs": config.envs,
            "reduced_model_gate": options,
            "suite": suite,
            "requests": requests,
        },
    )
    if not calibrating:
        baseline = read_json(options["baseline"])
        validate_baseline(baseline, identity, suite)
        state["baseline"] = baseline
    config.model = str(model)
    config.server_cmd.extend(["--served-model-name", SERVED_MODEL])
    return state


def curl_completion(url: str, request: dict, destination: Path) -> dict:
    write_json(destination.with_suffix(".request.json"), request)
    proc = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--max-time",
            "300",
            "--header",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            url,
        ],
        input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=310,
    )
    destination.write_bytes(proc.stdout)
    destination.with_suffix(".stderr.txt").write_bytes(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"curl failed with exit {proc.returncode}; see {destination}")
    return json.loads(proc.stdout)


def run_round(state: dict, server, *, calibrating: bool = False) -> dict:
    from tools.vllm_bench import VllmbenchRunner

    root = state["report_dir"]
    errors = []
    outputs = {}
    for prompt in state["requests"]:
        texts = []
        for repeat in range(CALIBRATION_REPEATS if calibrating else 2):
            destination = root / f"{prompt['id']}-{repeat}.response.json"
            try:
                response = curl_completion(server.url_for("v1", "completions"), prompt["request"], destination)
                text = normalized_completion(response, prompt["request"])
                texts.append(text)
                if not calibrating:
                    compare_output(text, state["baseline"]["outputs"][prompt["id"]])
            except Exception as exc:
                errors.append(f"{prompt['id']} repeat {repeat}: {exc}")
        outputs[prompt["id"]] = texts
        if calibrating and len(set(texts)) > 1:
            errors.append(f"Unstable output within one server start: {prompt['id']}")

    performance = {}
    for name, workload in state["suite"]["workloads"].items():
        samples = []
        rounds = CALIBRATION_REPEATS if calibrating else NIGHTLY_REPEATS
        for iteration in range(rounds + 1):
            try:
                run_dir = root / f"{name}-{'warmup' if iteration == 0 else iteration}"
                run_dir.mkdir()
                with VllmbenchRunner(
                    SERVED_MODEL,
                    server.port,
                    {**state["suite"]["performance_args"], **workload},
                    baseline=0,
                    verify=False,
                    model_path=state["model"],
                    host_ip="127.0.0.1",
                    backend="vllm",
                    endpoint="/v1/completions",
                    result_dir=str(run_dir.resolve()),
                    timeout_seconds=1800,
                ) as runner:
                    result = runner.result
                validate_perf(result, workload)
                if iteration:
                    samples.append(result)
            except Exception as exc:
                errors.append(f"{name} round {iteration}: {exc}")
        performance[name] = samples
        try:
            aggregate = aggregate_perf(samples, workload, repeats=rounds, calibrating=calibrating)
            if not calibrating:
                compare_perf(aggregate["median"], state["baseline"]["performance"][name]["median"])
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    report = {
        "format": FORMAT,
        "identity": state["identity"],
        "runtime": state["runtime"],
        "outputs": outputs,
        "performance": performance,
        "errors": errors,
        "passed": not errors,
    }
    write_json(root / "report.json", report)
    if errors:
        raise ValueError("GLM reduced gate failed:\n" + "\n".join(errors))
    return report


def finalize_calibration(reports: list[dict], suite: dict) -> dict:
    if len(reports) != CALIBRATION_STARTS or any(not r.get("passed") or r.get("errors") for r in reports):
        raise ValueError("Calibration requires three successful independent server starts")
    if (
        any(not isinstance(r.get("server_run_id"), str) or not r["server_run_id"] for r in reports)
        or len({r["server_run_id"] for r in reports}) != CALIBRATION_STARTS
    ):
        raise ValueError("Duplicate/missing server run identities")
    identity = reports[0]["identity"]
    if any(r["identity"] != identity for r in reports):
        raise ValueError("Calibration configurations changed between starts")
    if any(r["runtime"] != reports[0]["runtime"] for r in reports):
        raise ValueError("Calibration software changed between starts")
    outputs = {}
    for prompt in suite["prompts"]:
        texts = [text for report in reports for text in report["outputs"][prompt["id"]]]
        if len(texts) != CALIBRATION_STARTS * CALIBRATION_REPEATS or len(set(texts)) != 1 or not texts[0].strip():
            raise ValueError(f"Unstable/incomplete output: {prompt['id']}")
        outputs[prompt["id"]] = texts[0]
    performance = {}
    for name, workload in suite["workloads"].items():
        for report in reports:
            aggregate_perf(report["performance"][name], workload, repeats=CALIBRATION_REPEATS, calibrating=True)
        results = [sample for report in reports for sample in report["performance"][name]]
        performance[name] = aggregate_perf(
            results, workload, repeats=CALIBRATION_STARTS * CALIBRATION_REPEATS, calibrating=True
        )
    return {
        "format": FORMAT,
        "identity": identity,
        "outputs": outputs,
        "performance": performance,
        "calibration": {
            "starts": CALIBRATION_STARTS,
            "repeats": CALIBRATION_REPEATS,
            "runtime": [report["runtime"] for report in reports],
            "server_run_ids": [report["server_run_id"] for report in reports],
        },
    }


def run_nightly(config, server_factory) -> None:
    """Publish an explicit failure even when preparation or server startup fails."""
    report = {"name": config.name, "passed": False}
    report_path = Path("benchmark_results") / f"{config.name}.json"
    try:
        state = prepare_case(config)
        with server_factory(
            model=config.model,
            vllm_serve_args=config.server_cmd,
            server_host="127.0.0.1",
            server_port=config.server_port,
            env_dict=config.envs,
            auto_port=False,
            max_wait_seconds=1800,
        ) as server:
            report = run_round(state, server)
    except Exception as exc:
        report["error"] = str(exc)
        message = str(exc).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title=GLM reduced nightly::{message}")
        raise
    finally:
        write_json(report_path, report)
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(f"\nGLM reduced nightly: {'PASS' if report['passed'] else 'FAIL'}\n")
                handle.write(f"\nDetails: `{report_path}` and `benchmark_results/{config.name}/`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--output", required=True, help="New calibration directory; never overwrites a baseline")
    parser.add_argument("--baseline", help="Check an existing baseline instead of calibrating")
    parser.add_argument("--requests", help="Fixed token requests paired with --baseline")
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.requests):
        parser.error("--baseline and --requests must be supplied together")
    from tests.e2e.conftest import RemoteOpenAIServer
    from tests.e2e.nightly.single_node.models.scripts.single_node_config import SingleNodeConfigLoader

    config = SingleNodeConfigLoader.from_yaml_cases(str(Path(args.config).resolve()))[0]
    options = config.extra_config["reduced_model_gate"]
    if args.source_dir:
        options["source_dir"] = args.source_dir
    if args.cache_dir:
        options["cache_dir"] = args.cache_dir
    if args.baseline:
        options["baseline"] = args.baseline
        options["requests"] = args.requests
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    calibrating = not args.baseline
    for start in range(CALIBRATION_STARTS if calibrating else 1):
        current = copy.deepcopy(config)
        state = prepare_case(current, calibrating=calibrating, report_dir=root / str(start), hardware=args.hardware)
        write_json(root / "requests.json", {"requests": state["requests"]})
        with RemoteOpenAIServer(
            current.model,
            current.server_cmd,
            server_host="127.0.0.1",
            server_port=current.server_port,
            env_dict=current.envs,
            auto_port=False,
            max_wait_seconds=1800,
        ) as server:
            report = run_round(state, server, calibrating=calibrating)
            report["server_run_id"] = uuid.uuid4().hex
            write_json(state["report_dir"] / "report.json", report)
            reports.append(report)
    if not calibrating:
        return
    baseline = finalize_calibration(reports, state["suite"])
    validate_baseline(baseline, state["identity"], state["suite"])
    write_json(root / "requests.json", {"requests": state["requests"]})
    write_json(root / "baseline.json", baseline)


if __name__ == "__main__":
    main()
