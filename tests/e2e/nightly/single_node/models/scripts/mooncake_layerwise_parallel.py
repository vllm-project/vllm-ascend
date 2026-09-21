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
import time
import uuid


def reset_local_prefix_cache(llm, timeout: float = 30.0, interval: float = 0.5) -> None:
    """Clear only the local prefix cache, so a warm hit must come from the pool.

    The layerwise save path keeps the prompt's blocks until its final layer
    commits, so a request that has just returned can still have live blocks and
    the reset reports failure. Retry briefly rather than failing the run on that
    race.
    """
    deadline = time.monotonic() + timeout
    while True:
        if llm.reset_prefix_cache(reset_connector=False):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Could not clear the local prefix cache; cannot validate a remote hit")
        time.sleep(interval)


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
        "--prompt-tokens",
        type=int,
        default=32768,
        help="Prompt length in tokens. It has to exceed the model's cache "
        "transfer granularity (the LCM of the group page sizes) or nothing is "
        "stored and nothing can be hit -- 16384 for a DeepSeek-V4 hybrid layout, "
        "so the default covers two of those blocks.",
    )
    parser.add_argument(
        "--engine-args",
        default="{}",
        help="JSON object of extra engine kwargs, for example the model's "
        "quantization or dtype flags. Keys given here override the defaults "
        "below; the connector config is owned by this script.",
    )
    parser.add_argument(
        "--kv-role",
        default="kv_both",
        choices=("kv_both", "kv_producer", "kv_consumer"),
        help="Pool role of the engine. kv_producer saves without ever loading, "
        "which is the control for 'does the save path perturb the caches it "
        "writes': with no load, the two rounds must still match.",
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
        # Has to fit the prompt (see --prompt-tokens) plus the sampled tokens.
        "max_model_len": max(8192, args.prompt_tokens + 256),
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
        "kv_role": args.kv_role,
        "kv_connector_extra_config": {
            "backend": "mooncake",
            "use_layerwise": True,
            "layerwise_prefetch_layers": args.prefetch_layers,
        },
    }

    llm = LLM(**engine_kwargs)
    # A hybrid layout can have a page far larger than the scheduler block size,
    # so build the prompt from tokens and size it past that granularity.
    tokenizer = llm.get_tokenizer()
    salt = tokenizer.encode(f"Mooncake layerwise validation {uuid.uuid4()}. ", add_special_tokens=False)
    body = tokenizer.encode(
        "Explain how a sliding window and compressed attention retain context. ", add_special_tokens=False
    )
    token_ids = (salt + body * (args.prompt_tokens // max(1, len(body)) + 1))[: args.prompt_tokens]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    cold = llm.generate([{"prompt_token_ids": token_ids}], sampling)[0]
    assert (cold.num_cached_tokens or 0) == 0, "The first request must have a cold prefix"
    assert cold.prompt_token_ids is not None
    assert len(cold.prompt_token_ids) > args.max_num_batched_tokens, "Prompt must exercise chunked prefill"
    can_load = args.kv_role != "kv_producer"
    for iteration in range(2):
        reset_local_prefix_cache(llm)
        warm = llm.generate([{"prompt_token_ids": token_ids}], sampling)[0]
        if can_load:
            assert (warm.num_cached_tokens or 0) >= args.block_size, (
                f"No remote block was loaded (cached_tokens={warm.num_cached_tokens}); is the prompt "
                f"({args.prompt_tokens} tokens) longer than the model's transfer granularity?"
            )
        else:
            print(f"control run (kv_producer): cached_tokens={warm.num_cached_tokens}")
        warm_ids = warm.outputs[0].token_ids
        cold_ids = cold.outputs[0].token_ids
        if warm_ids != cold_ids:
            # A failed load is recomputed, so a mismatch means the restored KV
            # reached the wrong place rather than not reaching any.
            first_diff = next((i for i, (a, b) in enumerate(zip(cold_ids, warm_ids)) if a != b), None)
            raise AssertionError(
                f"Cold/warm generation differs (cached_tokens={warm.num_cached_tokens}, "
                f"first differing index={first_diff}, cold={len(cold_ids)} warm={len(warm_ids)} tokens)\n"
                f"  cold: {tokenizer.decode(cold_ids[:24])!r}\n"
                f"  warm: {tokenizer.decode(warm_ids[:24])!r}"
            )
        print(f"Warm round {iteration + 1}: cached_tokens={warm.num_cached_tokens}, token IDs match")
    print(f"PASS: Mooncake layerwise TP={args.tp}, PP={args.pp}")
    if args.pp > 1 and not args.debug:
        print("note: rerun with --debug to compare the per-rank layout identity lines")


if __name__ == "__main__":
    main()
