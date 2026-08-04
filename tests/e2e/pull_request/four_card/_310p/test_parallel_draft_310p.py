#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash / DSpark parallel drafting on Atlas 300I (310P), TP=4, drafter eager.

Mirrors the A2 tests (one_card/spec_decode/test_dflash.py, test_dspark.py):
same checkpoints, prompt, chat template, max_tokens, acceptance helper and
per-position tolerance. Acceptance moves more with prompt formatting than with
anything this port changes, so comparing against BASELINES only means something
if those match. Only what 310P forces differs -- fp16 (the custom FIA kernel is
fp16-only), block_size 128 (the only KV page size its block selection covers),
enforce_eager (the drafter cannot be captured; see parallel_draft_attention),
and TP=4, which shards heads without changing the numerics.
"""

from __future__ import annotations

import os

import pytest

# 310P adaptation lives on Model Runner V1.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.v1.metrics.reader import Counter, Vector  # noqa: E402

from tests.e2e.conftest import VllmRunner  # noqa: E402
from tests.e2e.pull_request.one_card.spec_decode.utils import (  # noqa: E402
    BASELINES,
    DFLASH,
    DSPARK,
    calculate_acceptance_per_pos,
)

# (method, num_speculative_tokens), matching the A2 tests' parametrisation.
CASES = [("dspark", DSPARK, 7), ("dflash", DFLASH, 8)]

# The A2 tolerance, applied one-sided: upstream uses abs(a - b) < 0.1, which
# also fails when acceptance improves, and a port gate only cares about the
# downside.
TOLERANCE = 0.1


def _resolve(name, env):
    """Allow a pre-fetched local copy so four TP ranks do not race to download."""
    return os.environ.get(env, name)


@pytest.mark.parametrize(("method", "registry", "num_speculative_tokens"), CASES)
def test_parallel_draft_acceptance(method, registry, num_speculative_tokens):
    main_model = _resolve(registry[method]["main"], "QWEN3_8B_PATH")
    spec_model = _resolve(registry[method]["spec"], f"{method.upper()}_SPEC_PATH")
    for path in (main_model, spec_model):
        if path.startswith("/") and not os.path.isdir(path):
            pytest.skip(f"model path not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello, your name is"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    ]
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=256)

    with VllmRunner(
        main_model,
        max_model_len=4096,
        dtype="float16",
        tensor_parallel_size=4,
        block_size=128,
        enforce_eager=True,
        distributed_executor_backend="mp",
        enable_prefix_caching=False,
        disable_log_stats=False,
        max_num_seqs=256,
        gpu_memory_utilization=0.8,
        speculative_config={
            "method": method,
            "model": spec_model,
            "num_speculative_tokens": num_speculative_tokens,
            "draft_tensor_parallel_size": 4,
        },
    ) as llm:
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")

    num_drafts = sum(m.value for m in metrics if m.name == "vllm:spec_decode_num_drafts")
    acceptance = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = BASELINES[method]
    print(f"{method}: num_drafts={num_drafts} acceptance_per_pos={acceptance} golden={golden}")

    # num_drafts is the denominator of every rate above and this prompt hits EOS
    # well inside max_tokens, so it is small -- BASELINES is itself quantised to
    # fifths. Assert it rather than leave it invisible: a change that stopped
    # generation after one step would pass or fail for an unrelated reason.
    assert num_drafts >= 5, f"only {num_drafts} draft steps; the rates are too coarse to compare"

    # Every position, not just position 0: the 310P-specific machinery (context
    # KV precompute, per-layer drafting RoPE, query slot mapping, non-causal FIA
    # over the query block) shows up as a decaying tail while position 0 -- the
    # target's own bonus token -- still looks healthy.
    low = [i for i, (a, b) in enumerate(zip(acceptance, golden)) if a < b - TOLERANCE]
    assert not low, (
        f"{method} acceptance below the A2 baseline at positions {low}: "
        f"got {[round(a, 4) for a in acceptance]}, golden {golden}, tolerance {TOLERANCE}"
    )
