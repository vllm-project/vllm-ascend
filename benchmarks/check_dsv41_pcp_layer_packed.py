# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check G64 prefix reuse against recomputation on eight local A3 devices with PCP4/TP2."""

import argparse
import random


def run(model, prefix_caching):
    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    rng = random.Random(37)
    # Unequal private prefixes cross block and prefill chunk boundaries.
    prefixes = [[rng.randrange(1000, 18000) for _ in range(n)] for n in (2051, 1539, 1285)]
    first = [{"prompt_token_ids": ids} for ids in prefixes]
    revisit = [{"prompt_token_ids": ids + [1001, 1002, 1003]} for ids in reversed(prefixes)]
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    results = []
    with VllmRunner(
        model,
        quantization="ascend",
        tokenizer_mode="deepseek_v41",
        tensor_parallel_size=2,
        prefill_context_parallel_size=4,
        max_model_len=4096,
        block_size=128,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        enable_prefix_caching=prefix_caching,
        enable_expert_parallel=True,
        seed=37,
        engram_config={"cpu_offload": True},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        additional_config={"dsv41_int8_long_kv": True},
    ) as runner:
        for prompts in (first, revisit):
            outputs = runner.model.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == len(prompts)
            for output in outputs:
                ids = list(output.outputs[0].token_ids)
                assert output.finished and len(ids) == params.max_tokens
                if prefix_caching and prompts is revisit:
                    assert output.num_cached_tokens > 0, "Revisited prefix missed"
                results.append(ids)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local DSV4.1 Flash ModelSlim W4A8 checkpoint")
    model_path = parser.parse_args().model
    baseline = run(model_path, prefix_caching=False)
    packed = run(model_path, prefix_caching=True)
    for request, (expected, actual) in enumerate(zip(baseline, packed, strict=True)):
        assert actual == expected, (request, expected, actual)
    print("PCP4/TP2: all six completions match, including three prefix-cache revisits.")
