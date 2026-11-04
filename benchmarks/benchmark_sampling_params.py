# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare P/D parameter construction and EngineCore MessagePack decoding.

[Role] Microbenchmark the transfer-only validation fast path.
[Invariants] Compare the same connector-shaped payload and vLLM decoder.
[Timeouts & Locks] Five bounded samples per case; no waits or locks.
[Side-Effect Policy] Patch SamplingParams only in this benchmark process;
never load model weights or start an inference service.

Run with VLLM_PLUGINS="" to measure the upstream validator before loading
Ascend platform patches. The payload is generated, not a production capture.
"""

import json
import runpy
import statistics
import time
from pathlib import Path

import msgspec
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest


def median_ms(operation, *args, **kwargs):
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        operation(*args, **kwargs)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main():
    original = SamplingParams._verify_extra_args
    for groups, blocks_per_group in [(4, 8192), (40, 8192)]:
        extra_args = {
            "kv_transfer_params": {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "remote_block_ids": [list(range(blocks_per_group)) for _ in range(groups)],
                "remote_engine_id": "prefill-engine",
                "remote_request_id": "prefill-request",
                "remote_host": "127.0.0.1",
                "remote_port": 36011,
                "tp_size": 4,
            }
        }
        SamplingParams._verify_extra_args = original
        params = SamplingParams(extra_args=extra_args)
        request = EngineCoreRequest(
            request_id="decode-request",
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        wire = msgspec.msgpack.encode(request)
        decoder = msgspec.msgpack.Decoder(EngineCoreRequest)
        result = {"groups": groups, "block_ids": groups * blocks_per_group, "wire_bytes": len(wire)}
        for label in ["baseline", "patched"]:
            if label == "patched":
                runpy.run_path(
                    str(Path(__file__).resolve().parents[1] / "vllm_ascend/patch/platform/patch_sampling_params.py")
                )
            result[label] = {
                "construct_ms": median_ms(SamplingParams, extra_args=extra_args),
                "decode_ms": median_ms(decoder.decode, wire),
            }
            assert decoder.decode(wire).sampling_params.extra_args == extra_args
        print(json.dumps(result), flush=True)
    SamplingParams._verify_extra_args = original


if __name__ == "__main__":
    main()
