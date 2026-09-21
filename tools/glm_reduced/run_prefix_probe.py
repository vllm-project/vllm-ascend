# SPDX-License-Identifier: Apache-2.0
"""Instrument one eager GLM run with a fixed continuation; never retries.

Uses a real full or reduced checkpoint without changing its layer count.
This diagnostic disables MTP and graph capture on BOTH sides. Instrumented
timings are not performance measurements. Outputs are observations until a
separate comparison and reference qualification have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def install_probe(worker, output_dir, prompt_length, continuation):
    """Install per-request instrumentation after engine warmup on every rank."""
    import numpy as np
    import torch
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather

    from tools.glm_reduced.prefix_probe import project_boundary

    model = worker.model_runner.model
    if hasattr(model, "_prefix_probe"):
        raise ValueError("previous probe was not removed")
    if model.model.start_layer != 0 or model.model.end_layer < 8:
        raise ValueError("probe requires PP1 and at least eight decoder layers")
    original_logits = model.compute_logits
    state = {"steps": 0, "original_logits": original_logits, "projecting": False}
    rank = get_tp_group().rank_in_group
    destination = Path(output_dir) / f"rank-{rank}"
    destination.mkdir(parents=True, exist_ok=False)

    def capture(layer, args, outputs):
        step = state["steps"]
        if step >= len(continuation):
            raise ValueError("unexpected extra forward in fixed-continuation probe")
        positions = args[0]
        hidden, residual = outputs
        row = prompt_length - 1 if step == 0 else 0
        if int(positions[row].item()) != prompt_length - 1 + step:
            raise ValueError("captured position does not match fixed request history")
        # Call the unmodified logits processor, avoiding forced-token feedback.
        proxy = type("Projection", (), {})()
        proxy.model = model.model
        proxy.compute_logits = original_logits
        state["projecting"] = True
        try:
            with torch.inference_mode():
                values = project_boundary(
                    proxy,
                    hidden,
                    residual,
                    token_count=positions.numel(),
                    row_indices=[row],
                    gather_rows=lambda tensor: tensor_model_parallel_all_gather(tensor, 0),
                )
        finally:
            state["projecting"] = False
        arrays = {key: tensor.detach().float().cpu().numpy() for key, tensor in values.items() if tensor is not None}
        arrays["position"] = np.array([prompt_length - 1 + step])
        np.savez(destination / f"step-{step:03d}.npz", **arrays)
        state["steps"] += 1

    def capture_actual_norm(module, args, outputs):
        if state["projecting"] or model.model.end_layer != 8:
            return
        step = state["steps"] - 1
        if not 0 <= step < len(continuation):
            raise ValueError("normalization without matching boundary capture")
        row = prompt_length - 1 if step == 0 else 0
        normalized = outputs[0][[row]].detach().float().cpu().numpy()
        np.savez(destination / f"actual-norm-{step:03d}.npz", normalized=normalized)

    def fixed_logits(hidden):
        logits = original_logits(hidden)
        if logits is None:
            return None
        step = state["steps"] - 1
        if not 0 <= step < len(continuation):
            raise ValueError("logits computation without matching boundary capture")
        if logits.shape[0] != 1:
            raise ValueError("probe requires one serial request and one sampled row")
        np.savez(destination / f"final-{step:03d}.npz", logits=logits.detach().float().cpu().numpy())
        forced = torch.full_like(logits, -torch.inf)
        forced[:, continuation[step]] = 0
        return forced

    state["handle"] = model.model.layers[7].register_forward_hook(capture)
    state["norm_handle"] = model.model.norm.register_forward_hook(capture_actual_norm)
    model.compute_logits = fixed_logits
    model._prefix_probe = state
    return {"rank": rank, "layers": model.model.end_layer}


def remove_probe(worker):
    """Restore the instance even when request execution failed."""
    model = worker.model_runner.model
    state = model._prefix_probe
    state["handle"].remove()
    state["norm_handle"].remove()
    model.compute_logits = state["original_logits"]
    del model._prefix_probe
    return {"steps": state["steps"]}


class PrefixProbeWorkerExtension:
    """Expose named RPC methods without enabling pickle-based RPC serialization."""

    def install_prefix_probe(self, output_dir, prompt_length, continuation):
        return install_probe(self, output_dir, prompt_length, continuation)

    def remove_prefix_probe(self):
        return remove_probe(self)


def main():
    """Collect each fixed request once, failing on incomplete observations."""
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine-json", required=True)
    parser.add_argument("--reference-evidence", required=True)
    parser.add_argument("--role", required=True, choices=("reference", "candidate"))
    args = parser.parse_args()
    engine = json.loads(Path(args.engine_json).read_text())
    extension = "tools.glm_reduced.run_prefix_probe.PrefixProbeWorkerExtension"
    if engine.get("worker_extension_cls", extension) != extension:
        raise ValueError("a different worker extension cannot be combined with this probe")
    engine["worker_extension_cls"] = extension
    required = {
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": 1,
        "pipeline_parallel_size": 1,
        "async_scheduling": False,
    }
    if any(engine.get(key) != value for key, value in required.items()) or engine.get("speculative_config"):
        raise ValueError("probe requires eager, PP1, batch1, no chunked prefill/prefix cache/MTP")
    requests = json.loads(Path(args.requests).read_text())["requests"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    continuation = [17, 18, 19, 20]  # Identical teacher-forced history on both sides.
    metadata = {
        "model": args.model,
        "role": args.role,
        "engine": engine,
        "requests": requests,
        "continuation": continuation,
        "reference_evidence": args.reference_evidence,
        "status": "STARTED",
        "scope": "eager prefix8 prefill plus three decode steps; no MTP",
    }
    from tools.glm_reduced.nightly import runtime_provenance

    metadata["runtime"] = runtime_provenance()
    metadata["probe_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("prefix_probe.py", "run_prefix_probe.py")
    }
    controlled_env = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "VLLM_USE_V2_MODEL_RUNNER",
        "HCCL_OP_EXPANSION_MODE",
        "HCCL_BUFFSIZE",
        "LCCL_DETERMINISTIC",
        "HCCL_DETERMINISTIC",
        "ATB_MATMUL_SHUFFLE_K_ENABLE",
        "ATB_LLM_LCOC_ENABLE",
        "PYTORCH_NPU_ALLOC_CONF",
    )
    metadata["environment"] = {name: os.environ.get(name) for name in controlled_env}
    metadata_path = output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    llm = LLM(model=args.model, **engine)
    sampling = SamplingParams(temperature=0, max_tokens=len(continuation), ignore_eos=True, detokenize=False)
    for item in requests:
        tokens = item["request"]["prompt"]
        workers = llm.collective_rpc("install_prefix_probe", args=(str(output / item["id"]), len(tokens), continuation))
        try:
            expected_layers = 78 if args.role == "reference" else 8
            if any(record["layers"] != expected_layers for record in workers):
                raise ValueError(f"{args.role} must execute {expected_layers} real decoder layers")
            result = llm.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)
        finally:
            counts = llm.collective_rpc("remove_prefix_probe")
        if any(record["steps"] != len(continuation) for record in counts):
            raise ValueError("incomplete forward captures")
        if list(result[0].outputs[0].token_ids) != continuation:
            raise ValueError("fixed continuation was not respected")
    metadata["status"] = "COLLECTED_NOT_COMPARED"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
