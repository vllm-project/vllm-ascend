# SPDX-License-Identifier: Apache-2.0
"""Flash text calibration smoke: checkpoint-derived quant metadata, synthetic weights."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from vllm import SamplingParams
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from tests.e2e.conftest import VllmRunner
from tools.ci.glm53flash_launch import effective_settings
from tools.ci.glm53flash_protocol import precision_lengths
from tools.ci.glm53flash_worker import BaselineWorker, prompts

LOADER = "glm53flash_ci_dummy"


@register_model_loader(LOADER)
class FlashDummyLoader(DummyModelLoader):
    def load_weights(self, model, model_config):
        with torch.no_grad():
            for name, value in model.named_parameters():
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                generator = torch.Generator(device=value.device).manual_seed(seed)
                if name.endswith("weight") and value.dtype == torch.int8:
                    value.random_(-8, 9, generator=generator)
                elif not value.is_floating_point() or "A_log" in name:
                    value.zero_()
                elif "dt_bias" in name:
                    value.fill_(-2.0)
                elif "hc_" in name:
                    if name.endswith("scale"):
                        value.fill_(1.0)
                    elif name.endswith("base"):
                        value.zero_()
                    else:
                        value.uniform_(-0.001, 0.001, generator=generator)
                elif "scale" in name:
                    value.fill_(0.01)
                elif "offset" in name or name.endswith("bias"):
                    value.zero_()
                elif "norm" in name and name.endswith("weight"):
                    value.fill_(1.0)
                else:
                    value.uniform_(-0.01, 0.01, generator=generator)


class FlashWorker(BaselineWorker):
    def end_flash_forced_capture(self, path, batch_size):
        """Compare equal-length batches with no accepted speculative tokens.

        Verification rows are request-major: the first row of each request
        predicts its next committed token. The caller must check zero accepts.
        """
        self.model_runner.model.compute_logits = self._original_compute_logits
        if self.rank != 0:
            return None
        shapes = [list(x.shape) for x in self._captured_logits]
        assert len(shapes) == 8 and all(s[0] % batch_size == 0 for s in shapes), shapes
        data = np.stack([x.reshape(batch_size, -1, x.shape[-1])[:, 0] for x in self._captured_logits])
        assert data.shape == (8, batch_size, 154880) and np.isfinite(data).all()
        np.save(path, data, allow_pickle=False)
        self._captured_logits = []
        return {"shape": list(data.shape), "call_shapes": shapes}

    def inspect_flash(self):
        model = self.model_runner.model
        counts = {}
        schemes = set()
        for module in model.modules():
            name = type(module).__name__
            counts[name] = counts.get(name, 0) + 1
            method = getattr(module, "quant_method", None)
            if method is not None:
                schemes.add(type(getattr(method, "quant_method", method)).__name__)
        layers = counts.get("Glm5NextDecoderLayer", 0)
        kda = counts.get("Glm5NextLinearAttention", 0)
        dsa = counts.get("Glm5NextMLAAttention", 0)
        mhc = sum(hasattr(m, "hc_attn_fn") for m in model.modules())
        assert layers in (5, 9) and kda + dsa == layers and mhc == layers, (layers, kda, dsa, mhc)
        assert dsa == (2 if layers == 9 else 1)
        assert "AscendW8A8DynamicFusedMoEMethod" in schemes, schemes
        return {"layers": layers, "kda": kda, "dsa": dsa, "mhc": mhc, "quant_methods": sorted(schemes)}


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
    }.items():
        os.environ[name] = value
    settings, changes, environment = effective_settings("text_tp4")
    os.environ.update(environment)
    settings.update(
        dict(
            load_format=LOADER,
            worker_extension_cls="tools.ci.glm53flash_collect.FlashWorker",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            distributed_executor_backend="mp",
            enforce_eager=args.mode == "eager",
            disable_log_stats=False,
        )
    )
    if args.mode == "eager":
        settings.pop("compilation_config", None)
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    (output / "community-to-ci.json").write_text(json.dumps(changes, indent=2))
    with VllmRunner(str(Path(args.model).resolve()), **settings) as runner:
        llm = runner.model
        config = llm.llm_engine.vllm_config
        block_size = config.cache_config.block_size
        runtime = {
            "block_size": block_size,
            "cache_dtype": config.cache_config.cache_dtype,
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        }
        (output / "runtime-settings.json").write_text(json.dumps(runtime, indent=2))
        evidence = llm.collective_rpc("inspect_flash")
        (output / "path-evidence.json").write_text(json.dumps(evidence, indent=2))
        (output / "weights.json").write_text(json.dumps(llm.collective_rpc("fingerprint"), indent=2))
        llm.collective_rpc("start_glm51_replay_count")
        lengths = (128,) if args.smoke else precision_lengths(block_size)
        for length in lengths:
            llm.collective_rpc("begin_capture")
            results = llm.generate(
                [{"prompt_token_ids": [10 + i % 1000 for i in range(length)]}],
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, detokenize=False, allowed_token_ids=[42]),
                use_tqdm=False,
            )
            assert results[0].outputs[0].token_ids == [42] * 8
            llm.collective_rpc("end_capture", args=(str(output / f"n{length}-cold.npy"),))
        replays = llm.collective_rpc("finish_glm51_replay_count")
        assert all(r > 0 for r in replays) if args.mode == "graph" else all(r == 0 for r in replays)
        (output / "result.json").write_text(json.dumps({"status": "PASS", "replays": replays, "lengths": lengths}))
        print("FLASH_TEXT_PASS", evidence, replays, flush=True)
        if args.mode == "graph" and not args.smoke:
            rows = []
            params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, detokenize=False)
            for iteration in range(18):
                begin = time.perf_counter()
                outputs = llm.generate(prompts(4, 128), params, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                assert all(len(r.outputs[0].token_ids) == 64 and r.num_cached_tokens == 0 for r in outputs)
                requests = []
                for result in outputs:
                    metrics = result.metrics
                    assert metrics is not None and not metrics.is_corrupted
                    assert metrics.num_generation_tokens == 64 and metrics.num_preemptions == 0
                    requests.append(
                        {
                            "ttft_ms": metrics.first_token_latency * 1000,
                            "tpot_ms": (metrics.last_token_ts - metrics.first_token_ts) * 1000 / 63,
                            "preemptions": metrics.num_preemptions,
                        }
                    )
                if iteration >= 3:
                    rows.append(
                        {
                            "iteration": iteration - 3,
                            "wall_ms": elapsed * 1000,
                            "output_tokens_s": 256 / elapsed,
                            "requests": requests,
                        }
                    )
                (output / "performance.json").write_text(json.dumps(rows, indent=2))
            print("FLASH_PERFORMANCE_PASS", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
