# SPDX-License-Identifier: Apache-2.0
"""Mixed-batch and MTP execution probe. Not an acceptance-rate performance test."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tools.ci.glm53flash_collect import LOADER, FlashDummyLoader  # noqa: F401
from tools.ci.glm53flash_launch import effective_settings


def run(args):
    if args.forced_logits and not args.deterministic_batches:
        raise ValueError("--forced-logits requires --deterministic-batches for request-major row alignment")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings, _, environment = effective_settings("text_tp4")
    os.environ.update(environment)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if args.batch_invariant:
        # Strict cross-batch equality requires invariant GEMM/reduction paths.
        # This is an accuracy profile, not the production performance profile.
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    deterministic_environment = {}
    if args.deterministic_kernels:
        # Match the existing GLM5.2 precision/spec-decode CI controls. Keep
        # these separate from the production/performance deployment profile.
        deterministic_environment = {
            "LCCL_DETERMINISTIC": "1",
            "HCCL_DETERMINISTIC": "true",
            "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
            "CLOSE_MATMUL_K_SHIFT": "1",
        }
        os.environ.update(deterministic_environment)
    if args.deterministic_batches:
        # Queue every request before the first engine step. An out-of-process
        # engine may begin a B=1 prefill while generate() is still submitting
        # the rest of a B=3/4 batch, confounding a state-reuse comparison.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        settings["async_scheduling"] = False
    settings.update(
        load_format=LOADER,
        worker_extension_cls="tools.ci.glm53flash_collect.FlashWorker",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        disable_log_stats=False,
    )
    if args.mtp:
        settings["speculative_config"] = {"method": "deepseek_mtp", "num_speculative_tokens": 3, "enforce_eager": True}
        settings["compilation_config"]["cudagraph_capture_sizes"] = [4, 8, 16]
    if args.eager:
        settings.pop("compilation_config", None)
        settings["enforce_eager"] = True
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "deterministic_batches": args.deterministic_batches,
                "deterministic_environment": deterministic_environment,
                "forced_logits": args.forced_logits,
                "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
            },
            indent=2,
        )
    )
    with VllmRunner(args.model, **settings) as runner:
        llm = runner.model
        llm.collective_rpc("inspect_flash")
        (output / "weights.json").write_text(json.dumps(llm.collective_rpc("fingerprint"), indent=2))
        llm.collective_rpc("start_glm51_replay_count")
        rows = []
        for lengths in ((3, 129, 513), (128, 128, 128, 128), (3, 129, 513)):
            inputs = [
                {"prompt_token_ids": [10 + (i + r * 137) % 1000 for i in range(n)]} for r, n in enumerate(lengths)
            ]
            results = llm.generate(
                inputs, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, detokenize=False), use_tqdm=False
            )
            assert len(results) == len(lengths)
            assert all(r.finished and len(r.outputs[0].token_ids) == 16 for r in results)
            rows.append([r.outputs[0].token_ids for r in results])
            (output / "observations.json").write_text(json.dumps({"outputs": rows}, indent=2))
        metrics = {
            m.name: m.value for m in llm.get_metrics() if m.name.startswith("vllm:spec_decode") and hasattr(m, "value")
        }
        if args.mtp:
            assert metrics.get("vllm:spec_decode_num_drafts", 0) > 0, metrics
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if args.forced_logits:
            accepted_before = metrics.get("vllm:spec_decode_num_accepted_tokens", 0)
            for batch in (3, 4):
                for repeat in range(2):
                    llm.collective_rpc("begin_capture")
                    inputs = [
                        {"prompt_token_ids": [10 + (i + r * 137) % 1000 for i in range(128)]} for r in range(batch)
                    ]
                    forced = llm.generate(
                        inputs,
                        SamplingParams(
                            temperature=0, max_tokens=8, ignore_eos=True, detokenize=False, allowed_token_ids=[42]
                        ),
                        use_tqdm=False,
                    )
                    assert all(r.outputs[0].token_ids == [42] * 8 for r in forced)
                    accepted_after = {
                        m.name: m.value
                        for m in llm.get_metrics()
                        if m.name == "vllm:spec_decode_num_accepted_tokens" and hasattr(m, "value")
                    }
                    assert accepted_after.get("vllm:spec_decode_num_accepted_tokens", 0) == accepted_before
                    key = f"forced-b{batch}-r{repeat}"
                    capture = llm.collective_rpc("end_flash_forced_capture", args=(str(output / f"{key}.npy"), batch))
                    (output / f"{key}.json").write_text(json.dumps(capture, indent=2))
            for batch in (3, 4):
                first = np.load(output / f"forced-b{batch}-r0.npy", allow_pickle=False)
                repeated = np.load(output / f"forced-b{batch}-r1.npy", allow_pickle=False)
                assert np.array_equal(first, repeated), f"B{batch} logits changed after state reuse"
            if os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1":
                smaller = np.load(output / "forced-b3-r0.npy", allow_pickle=False)
                larger = np.load(output / "forced-b4-r0.npy", allow_pickle=False)
                assert np.array_equal(smaller, larger[:, :3]), "Batch-invariant B3/B4 logits differ"
        assert rows[0] == rows[2], "Output changed after state-slot reuse"
        replays = llm.collective_rpc("finish_glm51_replay_count")
        assert len(replays) == 4 and all(n == 0 if args.eager else n > 0 for n in replays)
        (output / "result.json").write_text(
            json.dumps(
                {"status": "PASS", "mtp": args.mtp, "outputs": rows, "replays": replays, "metrics": metrics}, indent=2
            )
        )
        print("FLASH_MIXED_BATCH_PASS", args.mtp, replays, metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--forced-logits", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--deterministic-batches", action="store_true")
    parser.add_argument("--deterministic-kernels", action="store_true")
    parser.add_argument(
        "--batch-invariant", action="store_true", help="Strict accuracy profile; not for performance baselines"
    )
    run(parser.parse_args())
