# SPDX-License-Identifier: Apache-2.0
import pytest
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
BLOCK_SIZE = 128
NUM_BLOCKS = 64
TOKEN_BUDGET = BLOCK_SIZE
PREFIX_LENGTH = 3 * BLOCK_SIZE
SUFFIX_LENGTH = 16
MAX_TOKENS = 16

pytestmark = pytest.mark.e2e_model(MODEL)


def token_prompt(tokenizer, text, length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert tokens
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching,mtp",
    parallel="TP,PCP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@pytest.mark.parametrize("tp_size,pcp_size", [(2, 1), (1, 2)])
@wait_until_npu_memory_free()
def test_kvpp_combined_features(monkeypatch, tp_size, pcp_size):
    """Compare KVPP off/on with TP or V2 PCP, EP, chunk, prefix and MTP."""
    if pcp_size > 1:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    results = []
    for enabled in (False, True):
        with VllmRunner(
            maybe_model_redirect(MODEL),
            dtype="auto",
            quantization="ascend",
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=pcp_size,
            enable_expert_parallel=True,
            enforce_eager=True,
            async_scheduling=True,
            distributed_executor_backend="mp",
            max_model_len=4 * BLOCK_SIZE,
            max_num_seqs=4,
            max_num_batched_tokens=TOKEN_BUDGET,
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=NUM_BLOCKS,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.8,
            seed=42,
            disable_log_stats=False,
            speculative_config={"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": True},
            additional_config={"enable_kvpp": enabled},
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            prefix = token_prompt(tokenizer, "Explain how computers store historical information. ", PREFIX_LENGTH)
            prompts = [
                prefix + token_prompt(tokenizer, suffix, SUFFIX_LENGTH)
                for suffix in ("First answer: ", "Second answer: ")
            ]
            # The uncached prompt exceeds the token budget and requires chunking.
            outputs = []
            for prompt in prompts:
                (output,) = runner.model.generate(
                    [{"prompt_token_ids": prompt}],
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS),
                    use_tqdm=False,
                )
                assert output.finished
                assert len(output.outputs) == 1
                assert len(output.outputs[0].token_ids) == MAX_TOKENS
                outputs.append(output)
            results.append(outputs)
    assert [output.prompt_token_ids for output in results[0]] == [output.prompt_token_ids for output in results[1]]
    check_outputs_equal(
        outputs_0_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in results[0]],
        outputs_1_lst=[(list(output.outputs[0].token_ids), output.outputs[0].text) for output in results[1]],
        name_0="KVPP off",
        name_1="KVPP on",
    )
