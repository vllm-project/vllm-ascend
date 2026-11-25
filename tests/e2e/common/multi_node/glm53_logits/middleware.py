# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Test-only capture routes on the standard vllm serve API server.

The engine, model configuration, distributed workers and lifecycle remain owned
by vLLM. One API process per external DP rank keeps completion state local.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import regex as re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, JSONResponse
from vllm import SamplingParams


class LogitsCaptureMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.output = Path(os.environ.get("LOG_PREFIX", "./logits_results")) / "captures" / str(os.getpid())
        self.output.mkdir(parents=True, exist_ok=False)
        self.outcome = {}

    async def dispatch(self, request, call_next):
        prefix = "/glm53_logits/"
        if not request.url.path.startswith(prefix):
            return await call_next(request)
        route = request.url.path[len(prefix) :]
        engine = request.app.state.engine_client
        if route == "health" and request.method == "GET":
            groups = await engine.engine_core.call_utility_all_async(
                "collective_rpc", "logits_probe_features", None, (), {}
            )
            features = [feature for group in groups for feature in group]
            return JSONResponse({"ready": True, "features": features})
        if route == "capture" and request.method == "POST":
            return await self.capture(request)
        if route == "finish" and request.method == "POST":
            self.outcome = await request.json()
            (self.output / "completion.json").write_text(json.dumps(self.outcome, indent=2))
            return JSONResponse({"received": True})
        if route == "completion" and request.method == "GET":
            return JSONResponse(self.outcome)
        if route.startswith("artifacts/") and request.method == "GET":
            name = route.removeprefix("artifacts/")
            if not re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", name):
                return JSONResponse({"error": "Invalid artifact name"}, status_code=400)
            path = self.output / name
            if not path.is_file():
                return JSONResponse({"error": "Missing artifact"}, status_code=404)
            return FileResponse(path)
        return JSONResponse({"error": "Unknown capture route"}, status_code=404)

    async def capture(self, request):
        engine = request.app.state.engine_client
        vocab = request.app.state.vllm_config.model_config.get_vocab_size()
        output = self.output
        data = await request.json()
        name = data["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            return JSONResponse({"error": "Invalid capture ID"}, status_code=400)
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
        if sequence.logprobs is None or len(sequence.logprobs) != count:
            raise RuntimeError("Incomplete raw-logit positions")
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
            "logits_mode": request.app.state.vllm_config.model_config.logprobs_mode,
            "kv_transfer_params": result.kv_transfer_params,
            "num_cached_tokens": getattr(result, "num_cached_tokens", None),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return JSONResponse(metadata)
