# SPDX-License-Identifier: Apache-2.0
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def test_qwen36_dmtp_matches_target_greedy():
    target = "Qwen/Qwen3.6-27B"
    draft = "Piggy12/Qwen3.6-27B-DMTP"
    prompts = ["The capital of France is", "Continue the sequence of even integers: 2, 4, 6, 8,"]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    common = dict(
        revision="1cda14dd71a3f47115850eb790f416f165da1cb5",
        tensor_parallel_size=2,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        language_model_only=True,
        max_model_len=3072,
        max_num_seqs=2,
        disable_log_stats=False,
    )
    with VllmRunner(target, **common) as llm:
        reference = [r.outputs[0].token_ids for r in llm.model.generate(prompts, sampling)]
    spec = {
        "method": "dmtp",
        "model": draft,
        "revision": "d3d8d02162c64b88a9000735d260ef479825863e",
        "num_speculative_tokens": 7,
    }
    with VllmRunner(target, speculative_config=spec, **common) as llm:
        stop_after_first = SamplingParams(temperature=0, max_tokens=32, stop_token_ids=[reference[0][0]])
        stopped = llm.model.generate(prompts[:1], stop_after_first)[0].outputs[0]
        assert stopped.finish_reason == "stop" and len(stopped.token_ids) <= 1
        for batch in ([prompts[0]], prompts):
            actual = [r.outputs[0].token_ids for r in llm.model.generate(batch, sampling)]
            assert actual == reference[: len(batch)]
        metrics = {m.name: m for m in llm.model.get_metrics()}
        assert metrics["vllm:spec_decode_num_drafts"].value > 0
        # A verifier that silently rejects every draft can still pass output parity.
        accepted = metrics["vllm:spec_decode_num_accepted_tokens_per_pos"].values
        assert len(accepted) == 7 and sum(accepted[1:]) > 0
