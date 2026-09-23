# SPDX-License-Identifier: Apache-2.0
"""Run the registered 11-layer GLM5.x logits gate once, without recalibration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

from .logits_gate import DATA, MAIN_LAYERS, SAMPLE_COUNT, TP_SIZE, compare, load_case, read_json, sha256, validate_model


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def provision(options: dict, model_id: str) -> Path:
    # Lazy provisioning dependencies keep the comparator usable without vLLM/NPU.
    from filelock import FileLock

    from .prepare import prepare_checkpoint

    descriptor = DATA.parent / "source.json"
    if model_id != read_json(descriptor)["model"]:
        raise ValueError("No per-model baseline registered for this model")
    cache = Path(options.get("cache_dir", "~/.cache/vllm-ascend/glm-reduced")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(options.get("source_dir", cache / "glm52-source"))
    with FileLock(str(cache / ".prepare.lock"), timeout=1800):
        model, _ = prepare_checkpoint(
            descriptor, source, cache, layers=MAIN_LAYERS, download="source_dir" not in options
        )
    return model


def run_nightly(config) -> None:
    """SingleNodeConfigLoader entry point; never starts an OpenAI server."""
    options = config.extra_config["glm5x_logits_gate"]
    case, _, _ = load_case()
    if config.model != case["model"]:
        raise ValueError("This model requires its own registered baseline")
    report = Path("benchmark_results") / config.name / uuid.uuid4().hex
    report.mkdir(parents=True, exist_ok=False)
    try:
        model = provision(options, config.model)
        command = [
            sys.executable,
            "-m",
            "tools.glm_reduced.run_logits_gate",
            "--model",
            str(model),
            "--report-dir",
            str(report),
        ]
        with (report / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                env={**os.environ, **{k: str(v) for k, v in config.envs.items()}},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
    except Exception as exc:
        write_json(report / "failure.json", {"error": str(exc), "retry": "disabled"})
        raise


def validate_checkpoint(model: Path, case: dict) -> None:
    validate_model(read_json(model / "config.json"), case["model"])
    manifest = model / "reduction_manifest.json"
    if manifest.exists():
        from .reducer import resolve_index_filename, verify_reduced

        report = verify_reduced(str(model))
        source = read_json(manifest)["source"]
        expected = case["source_metadata_sha256"]
        if not report["ok"] or report["keep_layers"] != MAIN_LAYERS:
            raise ValueError("Reduced checkpoint failed integrity validation")
        for field, filename in (
            ("config_sha256", "config.json"),
            ("index_sha256", resolve_index_filename(expected)),
            ("quant_description_sha256", "quant_model_description.json"),
        ):
            if source[field] != expected[filename]:
                raise ValueError("Reduced checkpoint is from another source")
    else:
        # The independently audited B1/B2 artifact predates reduction_manifest.json.
        for name, expected in case["audited_model_metadata_sha256"].items():
            if sha256(model / name) != expected:
                raise ValueError(f"Unrecognized pre-provisioned checkpoint: {name}")


def collect(model: Path, destination: Path) -> dict:
    # Worker/runtime imports are deliberately isolated from CPU-only comparison.
    from vllm import LLM, SamplingParams

    case, dataset, reference = load_case()
    validate_checkpoint(model, case)
    settings = dict(
        model=str(model),
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=True,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        quantization="ascend",
        enforce_eager=True,
        block_size=128,
        num_gpu_blocks_override=64,
        seed=1024,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        worker_extension_cls="tools.glm_reduced.logits_probe.LogitsGateWorkerExtension",
        additional_config={"enable_reduce_sample": False},
    )
    provenance = {
        "settings": settings,
        "baseline": case,
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")
        },
    }
    for name in ("vllm", "vllm_ascend"):
        module = sys.modules.get(name)
        if module and module.__file__:
            root = Path(module.__file__).resolve().parent.parent
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
            provenance[name] = {"path": str(root), "commit": commit.stdout.strip() if commit.returncode == 0 else None}
    write_json(destination / "identity.json", provenance)
    llm = None
    results = []
    try:
        llm = LLM(**settings)
        workers = llm.collective_rpc("baseline_install", args=(MAIN_LAYERS,))
        if sorted(w["rank"] for w in workers) != list(range(TP_SIZE)):
            raise ValueError("Expected all eight TP workers")
        for sample in dataset["samples"]:
            llm.collective_rpc("baseline_arm", args=(sample,))
            llm.generate(
                [{"prompt_token_ids": sample["input_token_ids"]}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
            workers = llm.collective_rpc("baseline_finish")
            if sorted(w["rank"] for w in workers) != list(range(TP_SIZE)):
                raise ValueError("Missing worker capture")
            result = next(w for w in workers if w["rank"] == 0)
            results.append(result)
            with (destination / "progress.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(result) + "\n")
            print(f"Captured {len(results)}/{SAMPLE_COUNT}: {sample['id']}", flush=True)
        completed = llm.collective_rpc("baseline_remove")
        if len(completed) != TP_SIZE or any(
            w["captures"] != SAMPLE_COUNT or w["last_layer_calls"] != SAMPLE_COUNT for w in completed
        ):
            raise ValueError("Incomplete worker execution")
        report = compare(reference["samples"], results, dataset["samples"])
        write_json(destination / "candidate.json", {"samples": results})
        write_json(destination / "result.json", report)
        if not report["passed"]:
            raise AssertionError(f"Argmax agreement {report['agreement']:.6%} < 90%")
        print(f"GLM5X_LOGITS_GATE_OK {report['matched']}/{report['total']} ({report['agreement']:.6%})", flush=True)
        return report
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if any((args.report_dir / name).exists() for name in ("identity.json", "result.json", "failure.json")):
        raise ValueError("Refusing to overwrite evidence; use a fresh report directory")
    try:
        collect(args.model, args.report_dir)
    except Exception as exc:
        write_json(args.report_dir / "failure.json", {"error": str(exc), "retry": "disabled"})
        raise


if __name__ == "__main__":
    main()
