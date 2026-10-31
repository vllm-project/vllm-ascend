# SPDX-License-Identifier: Apache-2.0
import pytest
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal
from tests.e2e.pull_request.two_card.test_kvpp import token_prompt

MODEL = "Eco-Tech/GLM-5.2-w4a8"
DRAFT_MODEL = "RedHatAI/GLM-5.2-speculator.dspark"
BLOCK_SIZE = 128
MAX_TOKENS = 64

pytestmark = pytest.mark.e2e_model(MODEL)


@pytest.mark.parametrize("model_runner_v2", ["0", "1"], ids=["v1", "v2"])
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,dspark,chunked_prefill,prefix_caching",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_glm_dspark_kvpp_outputs(monkeypatch, model_runner_v2):
    """Keep GLM DSpark outputs unchanged across KVPP and repeated mixed batches."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", model_runner_v2)
    results = []
    for enabled in (False, True):
        with VllmRunner(
            maybe_model_redirect(MODEL),
            quantization="ascend",
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            async_scheduling=True,
            enforce_eager=True,
            distributed_executor_backend="mp",
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=BLOCK_SIZE,
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=64,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.92,
            seed=42,
            disable_log_stats=False,
            speculative_config={
                "method": "dspark",
                "model": maybe_model_redirect(DRAFT_MODEL),
                "num_speculative_tokens": 7,
                "enforce_eager": True,
            },
            additional_config={"enable_kvpp": enabled, "enable_fused_mc2": 0},
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            prefix = token_prompt(tokenizer, "Explain how computers store historical information. ", 3 * BLOCK_SIZE)
            long_prompt = prefix + token_prompt(tokenizer, "First answer: ", 16)
            other_long = prefix + token_prompt(tokenizer, "Second answer: ", 16)
            short_prompt = token_prompt(tokenizer, "What is one plus one? ", 16)
            outputs = []
            for batch in ([long_prompt], [long_prompt, short_prompt, other_long], [other_long, short_prompt]):
                generated = runner.model.generate(
                    [{"prompt_token_ids": prompt} for prompt in batch],
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS),
                    use_tqdm=False,
                )
                for output in generated:
                    assert output.finished
                    assert len(output.outputs[0].token_ids) == MAX_TOKENS
                    outputs.append((list(output.outputs[0].token_ids), output.outputs[0].text))
            assert (
                sum(
                    metric.value
                    for metric in runner.model.get_metrics()
                    if metric.name == "vllm:spec_decode_num_draft_tokens"
                )
                > 0
            )
            results.append(outputs)
    check_outputs_equal(
        outputs_0_lst=results[0], outputs_1_lst=results[1], name_0="DSpark KVPP off", name_1="DSpark KVPP on"
    )
