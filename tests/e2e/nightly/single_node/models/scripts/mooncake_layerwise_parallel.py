# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Manual NPU smoke test; requires a running Mooncake range-session backend.

Run each topology in a fresh process with MOONCAKE_CONFIG_PATH configured.
The pool and workers stay alive between cold and warm requests. Reset only
vLLM's local prefix cache so a successful warm hit must come from Mooncake.

The engine writes to and reads from the pool itself, so no P2P connector takes
part: what this exercises is the pool's key space under the requested topology.
With ``--pp > 1`` every stage has to agree on that key space, so run with
``--debug`` and compare the ``Mooncake hybrid layout id`` lines (one per rank,
they must all be identical) and the ``Layerwise PP key space`` window each
stage reports against the partition you configured.
"""

import argparse
import json
import os
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prefetch-layers", type=int, default=1)
    parser.add_argument(
        "--engine-args",
        default="{}",
        help="JSON object of extra engine kwargs, for example the model's "
        "quantization or dtype flags. Keys given here override the defaults "
        "below; the connector config is owned by this script.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log at DEBUG so the per-rank layout identity lines are visible.",
    )
    args = parser.parse_args()

    if args.debug:
        # vLLM reads this at import time and the workers inherit it.
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    from vllm import LLM, SamplingParams

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "distributed_executor_backend": "mp",
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": args.enforce_eager,
        "seed": 42,
    }
    engine_kwargs.update(json.loads(args.engine_args))
    # Applied last: the smoke is only meaningful with this connector.
    engine_kwargs["kv_transfer_config"] = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "use_layerwise": True,
            "layerwise_prefetch_layers": args.prefetch_layers,
        },
    }

    llm = LLM(**engine_kwargs)
    # Unique first tokens prevent earlier runs from satisfying the cold lookup.
    prompt = f"Session {uuid.uuid4().hex}.\n" + "Explain how a distributed key-value cache works.\n" * 256
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    cold = llm.generate([prompt], sampling)[0]
    assert (cold.num_cached_tokens or 0) == 0, "The first request must have a cold prefix"
    assert cold.prompt_token_ids is not None
    assert len(cold.prompt_token_ids) > args.max_num_batched_tokens, "Prompt must exercise chunked prefill"
    for iteration in range(2):
        assert llm.reset_prefix_cache(reset_connector=False), "Failed to clear the local prefix cache"
        warm = llm.generate([prompt], sampling)[0]
        assert (warm.num_cached_tokens or 0) >= args.block_size, "No complete remote block was loaded"
        assert warm.outputs[0].token_ids == cold.outputs[0].token_ids, "Cold/warm generation differs"
        print(f"Warm round {iteration + 1}: cached_tokens={warm.num_cached_tokens}, token IDs match")
    print(f"PASS: Mooncake layerwise TP={args.tp}, PP={args.pp}")
    if args.pp > 1 and not args.debug:
        print("note: rerun with --debug to compare the per-rank layout identity lines")


if __name__ == "__main__":
    main()
