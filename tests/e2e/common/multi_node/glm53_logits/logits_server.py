# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Test-only engine endpoint: fixed decode traces and full raw-logit capture."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import regex as re
from aiohttp import web
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


async def serve(config_path, port, output_dir):
    config = json.loads(Path(config_path).read_text())
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = engine.get_tokenizer()
    vocab = engine.model_config.get_vocab_size()
    per_engine = await engine.engine_core.call_utility_all_async(
        "collective_rpc", "logits_probe_features", None, (), {}
    )
    features = [feature for group in per_engine for feature in group]
    (output / "features.json").write_text(json.dumps(features, indent=2))
    print("EFFECTIVE_FEATURES=" + json.dumps(features), flush=True)
    finished = asyncio.Event()
    outcome = {}

    async def health(request):
        return web.json_response({"ready": True, "vocab_size": vocab, "config": config, "features": features})

    async def tokenize(request):
        data = await request.json()
        return web.json_response({"ids": tokenizer.encode(data["text"], add_special_tokens=False)})

    async def artifact(request):
        name = request.match_info["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", name):
            raise web.HTTPBadRequest(text="Invalid artifact name")
        return web.FileResponse(output / name)

    async def finish(request):
        outcome.update(await request.json())
        (output / "completion.json").write_text(json.dumps(outcome, indent=2))
        asyncio.get_running_loop().call_later(0.5, finished.set)
        return web.json_response({"received": True})

    async def capture(request):
        data = await request.json()
        name = data["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise web.HTTPBadRequest(text="Invalid capture ID")
        ids = data["prompt_ids"]
        trace = data.get("trace")
        count = len(trace) if trace is not None else data.get("max_tokens", 8)
        extra = {}
        if data.get("kv_transfer_params"):
            extra["kv_transfer_params"] = data["kv_transfer_params"]
        if trace is not None:
            extra["regression_trace"] = trace
        params = SamplingParams(
            temperature=0,
            seed=1024,
            max_tokens=count,
            min_tokens=count,
            ignore_eos=True,
            logprobs=-1,
            detokenize=False,
            extra_args=extra or None,
        )
        result = None
        async for item in engine.generate({"prompt_token_ids": ids}, params, name):
            result = item
        assert result is not None and result.finished
        sequence = result.outputs[0]
        assert len(sequence.token_ids) == count
        if trace is not None and list(sequence.token_ids) != trace:
            raise RuntimeError("Trace replay did not preserve the requested token sequence")
        scores = np.empty((count, vocab), dtype=np.float32)
        for pos, values in enumerate(sequence.logprobs):
            if set(values) != set(range(vocab)):
                raise RuntimeError(f"Incomplete vocabulary at {pos}: {len(values)} != {vocab}")
            for token_id, value in values.items():
                scores[pos, token_id] = value.logprob
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite raw logits")
        # Scores must be unnormalized; a log_softmax output would hide the
        # absolute model logits and is not accepted as a raw-logit capture.
        maxima = scores.max(axis=-1)
        log_z = maxima + np.log(np.exp(scores - maxima[:, None]).sum(axis=-1))
        if np.all(np.abs(log_z) < 1e-4):
            raise RuntimeError("Received normalized logprobs instead of raw logits")
        path = output / f"{name}.npz"
        np.savez_compressed(
            path,
            logits=scores,
            prompt_ids=np.asarray(ids, np.int64),
            token_ids=np.asarray(sequence.token_ids, np.int64),
        )
        metadata = {
            "id": name,
            "file": str(path),
            "shape": list(scores.shape),
            "prompt_sha256": hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest(),
            "token_ids": list(sequence.token_ids),
            "logits_mode": config["logprobs_mode"],
            "kv_transfer_params": result.kv_transfer_params,
            "num_cached_tokens": getattr(result, "num_cached_tokens", None),
            "config": config,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return web.json_response(metadata)

    app = web.Application(client_max_size=16 * 1024**2)
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/tokenize", tokenize),
            web.post("/capture", capture),
            web.get("/artifacts/{name}", artifact),
            web.post("/finish", finish),
        ]
    )
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        await web.TCPSite(runner, os.environ["VLLM_HOST_IP"], port).start()
        print(f"LOGITS_READY port={port}", flush=True)
        await finished.wait()
    finally:
        await runner.cleanup()
        engine.shutdown()
    if outcome.get("passed") is not True:
        raise RuntimeError("Leader reported a failed logits regression; see completion.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    asyncio.run(serve(options.config, options.port, options.output_dir))
